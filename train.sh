CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=25035 --nproc_per_node=4 --use_env train_mm.py --cfg configs/nyu.yaml --wandb 0


