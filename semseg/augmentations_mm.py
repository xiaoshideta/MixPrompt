import torchvision.transforms.functional as TF 
import random
import math
import torch
from torch import Tensor
from typing import Tuple, List, Union, Tuple, Optional
from PIL import Image
import sys
import collections
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import numbers

# class Compose:
#     def __init__(self, transforms: list) -> None:
#         self.transforms = transforms

#     def __call__(self, sample: list) -> list:
#         img, mask = sample['img'], sample['mask']
#         if mask.ndim == 2:
#             assert img.shape[1:] == mask.shape
#         else:
#             assert img.shape[1:] == mask.shape[1:]

#         for transform in self.transforms:
#             sample = transform(sample)

#         return sample
__all__ = ["Compose",
           "Resize",  # 尺寸缩减到对应size, 如果给定size为int,尺寸缩减到(size * height / width, size)
           "RandomScale",  # 尺寸随机缩放
           "RandomCrop",  # 随机裁剪,必要时可以进行padding
           "RandomHorizontalFlip",  # 随机水平翻转
           "ColorJitter",  # 亮度,对比度,饱和度,色调
           ]

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',   #最邻近
    Image.BILINEAR: 'PIL.Image.BILINEAR', #双线性
    Image.BICUBIC: 'PIL.Image.BICUBIC',   #三次样条插值
    Image.LANCZOS: 'PIL.Image.LANCZOS',   #多相位图像插值
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}
  #查看版本信息
if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """
 #callable:检查一个对象是否可以调用
    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

class Normalize:
    def __init__(self, mean: list = (0.485, 0.456, 0.406), std: list = (0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, sample: list) -> list:
        for k, v in sample.items():
            if k == 'mask':
                continue
            elif k == 'img':
                sample[k] = sample[k].float()
                sample[k] /= 255
                sample[k] = TF.normalize(sample[k], self.mean, self.std)
            else:
                sample[k] = sample[k].float()
                sample[k] /= 255
        
        return sample


class RandomColorJitter:
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            self.brightness = random.uniform(0.5, 1.5)
            sample['img'] = TF.adjust_brightness(sample['img'], self.brightness)
            self.contrast = random.uniform(0.5, 1.5)
            sample['img'] = TF.adjust_contrast(sample['img'], self.contrast)
            self.saturation = random.uniform(0.5, 1.5)
            sample['img'] = TF.adjust_saturation(sample['img'], self.saturation)
        return sample


class AdjustGamma:
    def __init__(self, gamma: float, gain: float = 1) -> None:
        """
        Args:
            gamma: Non-negative real number. gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter.
            gain: constant multiplier
        """
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.adjust_gamma(img, self.gamma, self.gain), mask


class RandomAdjustSharpness:
    def __init__(self, sharpness_factor: float, p: float = 0.5) -> None:
        self.sharpness = sharpness_factor
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            sample['img'] = TF.adjust_sharpness(sample['img'], self.sharpness)
        return sample


class RandomAutoContrast:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            sample['img'] = TF.autocontrast(sample['img'])
        return sample


class RandomGaussianBlur:
    def __init__(self, kernel_size: int = 3, p: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample: list) -> list:
        if random.random() < self.p:
            sample['img'] = TF.gaussian_blur(sample['img'], self.kernel_size)
            # img = TF.gaussian_blur(img, self.kernel_size)
        return sample

class RandomScale(object):
    def __init__(self, scale):
        assert isinstance(scale, Iterable) and len(scale) == 2
        assert 0 < scale[0] <= scale[1]
        self.scale = scale

    def __call__(self, sample):
        assert 'image' in sample.keys()
        assert 'label' in sample.keys()

        w, h = sample['image'].size
      #random.uniform:随机生成下一个实数，它在 [x, y] 范围内。
        scale = random.uniform(self.scale[0], self.scale[1])
        size = (int(round(h * scale)), int(round(w * scale)))

        # BILINEAR
        sample['image'] = F.resize(sample['image'], size, interpolation=InterpolationMode.BILINEAR)

        # NEAREST
        if 'depth' in sample.keys():
            sample['depth'] = F.resize(sample['depth'], size, interpolation=InterpolationMode.NEAREST)
        sample['label'] = F.resize(sample['label'], size, interpolation=InterpolationMode.NEAREST)

        return sample

class RandomCrop(object):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
       #random.randint:函数返回参数1和参数2之间的任意整数
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        img = sample['image']
        if self.padding is not None:
            for key in sample.keys():
                #pad:参数1：四维或者五维的tensor Variabe;参数2：不同tensor的填充方式(上，下，左右等,数值代表填充次数)；参数3:填充的数值
                #在"contant"模式下默认填充0，mode="reflect" or "replicate"时没有value参数
                #参数4：constant‘, ‘reflect’ or ‘replicate’三种模式，指的是常量，反射，复制三种模式
                sample[key] = F.pad(sample[key], self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            for key in sample.keys():
                sample[key] = F.pad(sample[key], (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            for key in sample.keys():
                sample[key] = F.pad(sample[key], (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(sample['image'], self.size)
        for key in sample.keys():
            sample[key] = F.crop(sample[key], i, j, h, w)

        return sample


# class RandomHorizontalFlip:
#     def __init__(self, p: float = 0.5) -> None:
#         self.p = p

#     def __call__(self, sample: list) -> list:
#         if random.random() < self.p:
#             for k, v in sample.items():
#                 sample[k] = TF.hflip(v)
#             return sample
#         return sample


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            return TF.vflip(img), TF.vflip(mask)
        return img, mask


class RandomGrayscale:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if random.random() < self.p:
            img = TF.rgb_to_grayscale(img, 3)
        return img, mask


class Equalize:
    def __call__(self, image, label):
        return TF.equalize(image), label


class Posterize:
    def __init__(self, bits=2):
        self.bits = bits # 0-8
        
    def __call__(self, image, label):
        return TF.posterize(image, self.bits), label


class Affine:
    def __init__(self, angle=0, translate=[0, 0], scale=1.0, shear=[0, 0], seg_fill=0):
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.seg_fill = seg_fill
        
    def __call__(self, img, label):
        return TF.affine(img, self.angle, self.translate, self.scale, self.shear, TF.InterpolationMode.BILINEAR, 0), TF.affine(label, self.angle, self.translate, self.scale, self.shear, TF.InterpolationMode.NEAREST, self.seg_fill) 


class RandomRotation:
    def __init__(self, degrees: float = 10.0, p: float = 0.2, seg_fill: int = 0, expand: bool = False) -> None:
        """Rotate the image by a random angle between -angle and angle with probability p

        Args:
            p: probability
            angle: rotation angle value in degrees, counter-clockwise.
            expand: Optional expansion flag. 
                    If true, expands the output image to make it large enough to hold the entire rotated image.
                    If false or omitted, make the output image the same size as the input image. 
                    Note that the expand flag assumes rotation around the center and no translation.
        """
        self.p = p
        self.angle = degrees
        self.expand = expand
        self.seg_fill = seg_fill

    def __call__(self, sample: list) -> list:
        random_angle = random.random() * 2 * self.angle - self.angle
        if random.random() < self.p:
            for k, v in sample.items():
                if k == 'mask':                
                    sample[k] = TF.rotate(v, random_angle, TF.InterpolationMode.NEAREST, self.expand, fill=self.seg_fill)
                else:
                    sample[k] = TF.rotate(v, random_angle, TF.InterpolationMode.BILINEAR, self.expand, fill=0)
            # img = TF.rotate(img, random_angle, TF.InterpolationMode.BILINEAR, self.expand, fill=0)
            # mask = TF.rotate(mask, random_angle, TF.InterpolationMode.NEAREST, self.expand, fill=self.seg_fill)
        return sample
    

class CenterCrop:
    def __init__(self, size: Union[int, List[int], Tuple[int]]) -> None:
        """Crops the image at the center

        Args:
            output_size: height and width of the crop box. If int, this size is used for both directions.
        """
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        return TF.center_crop(img, self.size), TF.center_crop(mask, self.size)


# class RandomCrop:
#     def __init__(self, size: Union[int, List[int], Tuple[int]], p: float = 0.5) -> None:
#         """Randomly Crops the image.

#         Args:
#             output_size: height and width of the crop box. If int, this size is used for both directions.
#         """
#         self.size = (size, size) if isinstance(size, int) else size
#         self.p = p

#     def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
#         H, W = img.shape[1:]
#         tH, tW = self.size

#         if random.random() < self.p:
#             margin_h = max(H - tH, 0)
#             margin_w = max(W - tW, 0)
#             y1 = random.randint(0, margin_h+1)
#             x1 = random.randint(0, margin_w+1)
#             y2 = y1 + tH
#             x2 = x1 + tW
#             img = img[:, y1:y2, x1:x2]
#             mask = mask[:, y1:y2, x1:x2]
#         return img, mask


class Pad:
    def __init__(self, size: Union[List[int], Tuple[int], int], seg_fill: int = 0) -> None:
        """Pad the given image on all sides with the given "pad" value.
        Args:
            size: expected output image size (h, w)
            fill: Pixel fill value for constant fill. Default is 0. This value is only used when the padding mode is constant.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        padding = (0, 0, self.size[1]-img.shape[2], self.size[0]-img.shape[1])
        return TF.pad(img, padding), TF.pad(mask, padding, self.seg_fill)


class ResizePad:
    def __init__(self, size: Union[int, Tuple[int], List[int]], seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        Args:
            size: Desired output size. 
                If size is a sequence, the output size will be matched to this. 
                If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
        """
        self.size = size
        self.seg_fill = seg_fill

    def __call__(self, img: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        H, W = img.shape[1:]
        tH, tW = self.size

        # scale the image 
        scale_factor = min(tH/H, tW/W) if W > H else max(tH/H, tW/W)
        # nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)

        # pad the image
        padding = [0, 0, tW - nW, tH - nH]
        img = TF.pad(img, padding, fill=0)
        mask = TF.pad(mask, padding, fill=self.seg_fill)
        return img, mask 

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class Resize(object):
    def __init__(self, size):
        #isinstance:判断两个类型是否相同
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, sample):
        assert 'image' in sample.keys()
        assert 'label' in sample.keys()


        sample['image'] = F.resize(sample['image'], self.size, interpolation=InterpolationMode.BILINEAR)
        # NEAREST
        if 'depth' in sample.keys():
            sample['depth'] = F.resize(sample['depth'], self.size, interpolation=InterpolationMode.NEAREST)
        sample['label'] = F.resize(sample['label'], self.size, interpolation=InterpolationMode.NEAREST)

        return sample

class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            for key in sample.keys():
                sample[key] = F.hflip(sample[key])

        return sample

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):

        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, sample):
        assert 'image' in sample.keys()
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        sample['image'] = transform(sample['image'])
        return sample

# class Resize:
#     def __init__(self, size: Union[int, Tuple[int], List[int]]) -> None:
#         """Resize the input image to the given size.
#         Args:
#             size: Desired output size. 
#                 If size is a sequence, the output size will be matched to this. 
#                 If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio.
#         """
#         self.size = size

#     def __call__(self, sample:list) -> list:
#         H, W = sample['img'].shape[1:]

#         # scale the image 
#         scale_factor = self.size[0] / min(H, W)
#         nH, nW = round(H*scale_factor), round(W*scale_factor)
#         for k, v in sample.items():
#             if k == 'mask':                
#                 sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
#             else:
#                 sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)
#         # img = TF.resize(img, (nH, nW), TF.InterpolationMode.BILINEAR)
#         # mask = TF.resize(mask, (nH, nW), TF.InterpolationMode.NEAREST)

#         # make the image divisible by stride
#         alignH, alignW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        
#         for k, v in sample.items():
#             if k == 'mask':                
#                 sample[k] = TF.resize(v, (alignH, alignW), TF.InterpolationMode.NEAREST)
#             else:
#                 sample[k] = TF.resize(v, (alignH, alignW), TF.InterpolationMode.BILINEAR)
#         # img = TF.resize(img, (alignH, alignW), TF.InterpolationMode.BILINEAR)
#         # mask = TF.resize(mask, (alignH, alignW), TF.InterpolationMode.NEAREST)
#         return sample


class RandomResizedCrop:
    def __init__(self, size: Union[int, Tuple[int], List[int]], scale: Tuple[float, float] = (0.5, 2.0), seg_fill: int = 0) -> None:
        """Resize the input image to the given size.
        """
        self.size = size
        self.scale = scale
        self.seg_fill = seg_fill

    def __call__(self, sample: list) -> list:
        # img, mask = sample['img'], sample['mask']
        H, W = sample['img'].shape[1:]
        tH, tW = self.size

        # get the scale
        ratio = random.random() * (self.scale[1] - self.scale[0]) + self.scale[0]
        # ratio = random.uniform(min(self.scale), max(self.scale))
        scale = int(tH*ratio), int(tW*4*ratio)
        # scale the image 
        scale_factor = min(max(scale)/max(H, W), min(scale)/min(H, W))
        nH, nW = int(H * scale_factor + 0.5), int(W * scale_factor + 0.5)
        # nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        for k, v in sample.items():
            if k == 'mask':                
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.NEAREST)
            else:
                sample[k] = TF.resize(v, (nH, nW), TF.InterpolationMode.BILINEAR)

        # random crop
        margin_h = max(sample['img'].shape[1] - tH, 0)
        margin_w = max(sample['img'].shape[2] - tW, 0)
        y1 = random.randint(0, margin_h+1)
        x1 = random.randint(0, margin_w+1)
        y2 = y1 + tH
        x2 = x1 + tW
        for k, v in sample.items():
            sample[k] = v[:, y1:y2, x1:x2]

        # pad the image
        if sample['img'].shape[1:] != self.size:
            padding = [0, 0, tW - sample['img'].shape[2], tH - sample['img'].shape[1]]
            for k, v in sample.items():
                if k == 'mask':                
                    sample[k] = TF.pad(v, padding, fill=self.seg_fill)
                else:
                    sample[k] = TF.pad(v, padding, fill=0)

        return sample



def get_train_augmentation(size: Union[int, Tuple[int], List[int]], seg_fill: int = 0):
    return Compose([
        ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5),
        RandomHorizontalFlip(0.5),  # p=0.5
        RandomScale(scale=(0.5, 2.0)),
        RandomCrop(size, pad_if_needed=True)
    ])

    # return Compose([
    #     RandomColorJitter(p=0.2), # 
    #     RandomHorizontalFlip(p=0.5), #
    #     RandomGaussianBlur((3, 3), p=0.2), #
    #     RandomResizedCrop(size, scale=(0.5, 2.0), seg_fill=seg_fill), #
    #     Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])

def get_val_augmentation(size: Union[int, Tuple[int], List[int]]):
    return Compose([
        Resize(size)
        # Resize(size),
        # Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


if __name__ == '__main__':
    h = 230
    w = 420
    sample = {}
    sample['img'] = torch.randn(3, h, w)
    sample['depth'] = torch.randn(3, h, w)
    sample['lidar'] = torch.randn(3, h, w)
    sample['event'] = torch.randn(3, h, w)
    sample['mask'] = torch.randn(1, h, w)
    aug = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomResizedCrop((512, 512)),
        Resize((224, 224)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    sample = aug(sample)
    for k, v in sample.items():
        print(k, v.shape)