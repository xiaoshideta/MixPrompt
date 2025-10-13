import os
import torch 
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss, MscCrossEntropyLoss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from val_mm import evaluate
from torch.optim.lr_scheduler import LambdaLR, StepLR


def main(cfg, gpu, save_dir):
    start = time.time()
    miou = 0
    best_mIoU = 0.0
    best_epoch = 0
    num_workers = 8
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    resume_path = cfg['MODEL']['RESUME']
    gpus = int(os.environ['WORLD_SIZE'])

    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'])
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'])
    model = eval(model_cfg['NAME'])(classes=trainset.n_classes)

    pretrained = "/home/xiaozhongyu/pretrained/segformer/segformer.b5.640x640.ade.160k.pth"
    pretrained2 = "/home/xiaozhongyu/pretrained/segformer/mit_b1.pth"
    
    model.init_pretrained(pretrained, pretrained2)
    resume_checkpoint = None
    model = model.to(device)

    if gpus > 1:
        rank = dist.get_rank()
    else:
        rank = -1

    if args.wandb and rank <=0:        
        import wandb
        wandb.init(project="Prompt",config=cfg,name="adme_v1")

    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        sampler_val = None
        model = DDP(model, device_ids=[gpu], output_device=0, find_unused_parameters=True)
    else:
        sampler = RandomSampler(trainset)
        sampler_val = None

    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE'] // gpus

    start_epoch = 0
    params_list = model.parameters()
    
    optimizer = torch.optim.SGD(params_list, lr=optim_cfg['LR'], weight_decay=optim_cfg['WEIGHT_DECAY'], momentum=0.9)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / epochs) ** 0.9)

    # optimizer = get_optimizer(model, optim_cfg['NAME'], optim_cfg['LR'], optim_cfg['WEIGHT_DECAY'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=4e-5)
    # scheduler = get_scheduler(sched_cfg['NAME'], optimizer, int((epochs+1)*iters_per_epoch), sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])
    
    if resume_checkpoint:
        start_epoch = resume_checkpoint['epoch'] - 1
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        loss = resume_checkpoint['loss']        
        best_mIoU = resume_checkpoint['best_miou']
           
    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=False, sampler=sampler)
    valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=False, sampler=sampler_val)

    scaler = GradScaler()

    if hasattr(trainset, 'class_weight'):
        print('using classweight in dataset')
        class_weight = trainset.class_weight
    else:
        print('using classweight in')
        classweight = ClassWeight(cfg['class_weight'])
        class_weight = classweight.get_weight(train_loader, cfg['n_classes'])
    class_weight = torch.from_numpy(class_weight).float().to(device)

    criterion = MscCrossEntropyLoss(ignore_index=dataset_cfg['IGNORE_LABEL']).to(device)

    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer = SummaryWriter(str(save_dir))
        logger.info('================== model structure =====================')
        logger.info(model)
        logger.info('================== training config =====================')
        logger.info(cfg)

    extra_index = dataset_cfg['MODALS'].index('depth')
    img_index = dataset_cfg['MODALS'].index('img')
    print(img_index)

    for epoch in range(start_epoch, epochs):
        model.train()
        if train_cfg['DDP']: 
            sampler.set_epoch(epoch)

        train_loss = 0.0        
        lr = scheduler.get_lr()
        lr = sum(lr) / len(lr)
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")

        for iter, sample in pbar:
            optimizer.zero_grad(set_to_none=True)
            depth = sample['depth'].to(device)
            image = sample['image'].to(device)
            lbl = sample['label'].to(device)
            
            with autocast():

                logits = model(image, depth)
                loss = criterion(logits, lbl)
            
            if args.wandb and rank <=0:
                wandb.log({"loss":loss, "lr":lr, "train_loss":train_loss / (iter+1)})
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            torch.cuda.synchronize()

            lr = scheduler.get_lr()
            lr = sum(lr) / len(lr)
            if lr <= 1e-8:
                lr = 1e-8
            train_loss += loss.item()

            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")

        scheduler.step()

        train_loss /= iter+1

        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()

        if ((epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch+1)>train_cfg['EVAL_START']) or (epoch+1) == epochs:
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                acc, acc_cls, miou, fw_iou = evaluate(model, valloader, device)
                writer.add_scalar('val/mIoU', miou, epoch)

                if miou > best_mIoU:
                    prev_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    prev_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    if os.path.isfile(prev_best): os.remove(prev_best)
                    if os.path.isfile(prev_best_ckp): os.remove(prev_best_ckp)
                    best_mIoU = miou
                    best_epoch = epoch+1
                    cur_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    cur_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), cur_best)
                    torch.save({'epoch': best_epoch,
                                'model_state_dict': model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': train_loss,
                                'scheduler_state_dict': scheduler.state_dict(),
                                'best_miou': best_mIoU,
                                }, cur_best_ckp)
                logger.info(f"Current epoch:{epoch} mIoU: {miou} Best mIoU: {best_mIoU}")

                if args.wandb and rank <=0:
	                wandb.log({"miou":miou,'best_mIoU':best_mIoU})

    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    logger.info(tabulate(table, numalign='right'))

    if args.wandb and rank <=0:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/deliver_rgbdel.yaml', help='Configuration file to use')
    parser.add_argument('--wandb', default=0, type=int, help='whether use wandb')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    exp_name = '_'.join([cfg['DATASET']['NAME'], model, modals])
    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(save_dir / 'train.log')
    main(cfg, gpu, save_dir)
    cleanup_ddp()