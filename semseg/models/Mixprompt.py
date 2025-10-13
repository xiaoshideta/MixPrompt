import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.nn as nn
import torch
from semseg.models.backbones.Mixprompt_transformer import MITB5, MITB1
from semseg.models.backbones.Mixprompt_transformer import OverlapPatchEmbed
from functools import partial
from semseg.models.heads import SegFormerHead
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50

def load_model():
    model = resnet50(pretrained=True)
    for n, p in model.named_parameters():
        # if "prompt" not in n:
        p.requires_grad = True

    return model

class ADA(nn.Module):
    def __init__(self, dim, down_ratio=4, n=4, mix_rgb=True, mix_x=False):
        super().__init__()
        self.n = n
        assert dim // down_ratio % n == 0
        self.down_rgb = nn.Linear(dim, dim // down_ratio)
        self.down_x = nn.Linear(dim, dim // down_ratio)

        if mix_rgb:
            self.mixer_rgb = nn.Linear(n, n, bias=False)
            nn.init.kaiming_uniform_(self.mixer_rgb.weight)
        else:
            self.mixer_rgb = nn.Identity()

        if mix_x:
            self.mixer_x = nn.Linear(n, n, bias=False)
            nn.init.kaiming_uniform_(self.mixer_x.weight)
        else:
            self.mixer_x = nn.Identity()

        self.up = nn.Linear(dim // down_ratio, dim)

    def forward(self, rgb, x): # rgb [1, 19200, 64]
        bsz, N, C = rgb.shape # bsz 1
        # breakpoint()
        rgb_sub = torch.stack(self.down_rgb(rgb).chunk(self.n, dim=-1), dim=-1) # rgb_sub [1, 19200, 4, 4]
        x_sub = torch.stack(self.down_x(x).chunk(self.n, dim=-1), dim=-1) # x_sub [1, 19200, 4, 4]

        rgb_mixed = self.mixer_rgb(rgb_sub).transpose(-1, -2).reshape(bsz, N, -1) # rgb_mixed [1, 19200, 16]
        x_mixed = self.mixer_x(x_sub).transpose(-1, -2).reshape(bsz, N, -1) # x_mixed [1, 19200, 16]

        prompt = self.up(rgb_mixed + x_mixed) # prompt [1, 19200, 64]

        return prompt


class Mixprompt(nn.Module):
    def __init__(self, in_chans=3, img_size=224, patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, classes=41):
        super(Mixprompt, self).__init__()

        self.rgb = MITB5(pretrained=True)
        # self.depth = MITB1(pretrained=True)
        # self.depth = resnet50(pretrained=True)
        self.depth = load_model()
        self.linear1 = nn.Linear(256, 64)

        # self.d_p = 0.01
        # self.conv1x1 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1)
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(64)
        self.norm2 = norm_layer(128)
        self.norm3 = norm_layer(320)
        self.norm4 = norm_layer(512)

        # self.normm1 = norm_layer(64)
        # self.normm2 = norm_layer(128)
        # self.normm3 = norm_layer(320)
        # self.normm4 = norm_layer(512)
        self.prompt_norms1 = norm_layer(64)
        self.prompt_norms2 = norm_layer(128)
        self.prompt_norms3 = norm_layer(320)
        self.prompt_norms4 = norm_layer(512)

        self.head = SegFormerHead(4)

        # self.learnable_prompt = nn.Parameter(torch.randn(1, 30, 32))

        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        self.ada1 = ADA(dim=64)
        self.ada2 = ADA(dim=128)
        self.ada3 = ADA(dim=320)
        self.ada4 = ADA(dim=512)

        self.conv1 = nn.Conv2d(150, classes, kernel_size=1)

    def forward(self, rgb, depth):
        
        orisize = rgb.shape
        B = rgb.shape[0] #[1, 3, 480, 640]
        outs = []
        x1, H, W = self.rgb.patch_embed1(rgb)  # [1, 19200, 64]
        
        d1 = self.depth.conv1(depth) # [1, 64, 240, 320]
        d1 = self.depth.bn1(d1) # [1, 64, 240, 320]
        d1 = self.depth.relu(d1) # [1, 64, 240, 320]
        d1 = self.depth.maxpool(d1) # [1, 64, 120, 160]
        d1 = self.depth.layer1(d1) # [1, 256, 120, 160]
        # breakpoint()
        d1 = d1.flatten(2).transpose(1, 2) # [1, 19200, 256]
        d1 = self.linear1(d1) # [1, 19200, 64]

        prompted = self.ada1(x1, d1)
        prompted = self.prompt_norms1(prompted)

        x1 = x1 + prompted
        
        for i, blk in enumerate(self.rgb.block1):
            x1 = blk(x1, H, W)
        # breakpoint()
        x1 = self.rgb.norm1(x1)
        
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        prompted = prompted.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x1)
        
        x2, H, W = self.rgb.patch_embed2(x1)
        d2, Hd, Wd = self.patch_embed2(prompted)

        prompted = self.ada2(x2, d2)
        prompted = self.prompt_norms2(prompted)
 
        x2 = x2 + prompted

        for i, blk in enumerate(self.rgb.block2):
            x2 = blk(x2, H, W)
        x2 = self.rgb.norm2(x2)
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        prompted = prompted.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x2)

        x3, H, W = self.rgb.patch_embed3(x2)
        d3, Hd, Wd = self.patch_embed3(prompted)

        prompted = self.ada3(x3, d3)
        prompted = self.prompt_norms3(prompted)

        x3 = x3 + prompted

        for i, blk in enumerate(self.rgb.block3):
            x3 = blk(x3, H, W)
        x3 = self.rgb.norm3(x3)
        x3 = x3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        prompted = prompted.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x3)

        

        x4, H, W = self.rgb.patch_embed4(x3)
        d4, Hd, Wd = self.patch_embed4(prompted)

        prompted = self.ada4(x4, d4)
        prompted = self.prompt_norms4(prompted)

        x4 = x4 + prompted

        for i, blk in enumerate(self.rgb.block4):
            x4 = blk(x4, H, W)
        x4 = self.rgb.norm4(x4)
        x4 = x4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x4)

        x = self.head(outs)
        x = F.interpolate(x, size=orisize[2:], mode='bilinear', align_corners=False)
        x = self.conv1(x)

        return x
    
    def init_pretrained(self, pretrained: str = None, pretrained2: str = None) -> None:
        if pretrained:
            model_dict = self.rgb.state_dict()
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            new_state_dict = OrderedDict()
            # for k, v in pretrained_dict['state_dict'].items():
            for k, v in pretrained_dict['state_dict'].items():
                if k[:8] == 'backbone':
                    name = k[9:]
                    new_state_dict[name] = v
                if k[:11] == 'decode_head':
                    name = k[12:]
                    name = 'head.' + name
                    new_state_dict[name] = v
            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
            msg = self.rgb.load_state_dict(pretrained_dict, strict=False)
            print(msg)

if __name__ == '__main__':
    # from thop import profile
    x = torch.randn(1, 3, 480, 640)
    ir = torch.randn(1, 3, 480, 640)
    edge = torch.randn(1, 480, 640)
    net = Mixprompt()
    s = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(s)

    x = net(x, ir)
    print(x.shape)
