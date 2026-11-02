import os
import pdb

import torch
import torchvision.transforms.functional as tf
import random
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

from models import utils


# fix randomness on DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# https://github.com/rwightman/pytorch-image-models/blob/d72ac0db259275233877be8c1d4872163954dfbb/timm/data/loader.py
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

# 实现了无限次地返回数据集中的元素
class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def is_image(src):
    return True if os.path.splitext(src)[1].lower() in ['.jpg', '.png', '.tif', '.ppm'] else False

# 加载和处理图像的类
# 类Image2ImageLoader_resize(Dataset)和类Image2ImageLoader_zero_pad(Dataset)除了transform函数内容不一样，其他完全一样
class Image2ImageLoader_resize(Dataset):

    def __init__(self,
                 x_path,
                 y_path,
                 mode,
                 **kwargs):

        self.mode = mode
        self.args = kwargs['args']

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

        x_img_name = os.listdir(x_path)
        y_img_name = os.listdir(y_path)
        x_img_name = filter(is_image, x_img_name)
        y_img_name = filter(is_image, y_img_name)

        self.img_x_path = []
        self.img_y_path = []

        x_img_name = sorted(x_img_name)
        y_img_name = sorted(y_img_name)
        # 将两个可迭代对象（例如两个列表）中的元素按顺序“配对”起来，生成一个迭代器，每个元素是一个元组 (x, y)，分别来自两个原始列表中的相同位置。
        img_paths = zip(x_img_name, y_img_name)
        for item in img_paths:
            self.img_x_path.append(x_path + os.sep + item[0])
            self.img_y_path.append(y_path + os.sep + item[1])
        # 确保图像路径 (img_x_path) 和标签路径 (img_y_path) 的数量是一样的
        assert len(self.img_x_path) == len(self.img_y_path), 'Images in directory must have same file indices!!'

        print(f'{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.mode}{utils.Colors.END}')
        self.img_x = []
        self.img_y = []
        for index in range(len(self.img_x_path)):
            x_path = self.img_x_path[index]
            y_path = self.img_y_path[index]
            self.img_x.append(Image.open(x_path).convert('RGB'))
            self.img_y.append(Image.open(y_path).convert('L'))


    # 图像增强(每次执行增强后就立即送入网络，叠加在线增强，不保留中间文件，数量上没有扩增，且原图没有送入到网络过，都是叠加增强后送入网络）
    def transform(self, image, target):
        # 无论是validation模式还是train模式，都进行统一的缩放，直接将图像和标签缩放到指定的输入大小self.args.input_size。
        # 从 self.args.input_size 中获取目标尺寸
        resize_h = self.args.input_size[0]
        resize_w = self.args.input_size[1]
        #  图像和标签缩放，改变图像的比例,图像使用默认插值方法进行缩放,标签则使用最近邻插值方法
        image = tf.resize(image, [resize_h, resize_w])
        target = tf.resize(target, [resize_h, resize_w], interpolation=InterpolationMode.NEAREST)

        # 下面的内容和Image2ImageLoader_zero_pad(Dataset)的transform函数完全一样
        # 如果当前不是验证模式（即是训练模式），则执行后续的数据增强操作。
        if not self.mode == 'validation':
            random_gen = random.Random()  # thread-safe random,随机生成器

            # 以 80% 的概率执行 CutMix 操作，将图像的部分区域与另一张随机图像合成。
            if (random_gen.random() < 0.8) and self.args.transform_cutmix:
                rand_n = random_gen.randint(0, self.__len__() - 1)     # randomly generates reference image on dataset
                image_refer = Image.open(self.img_x_path[rand_n]).convert('RGB')
                target_refer = Image.open(self.img_y_path[rand_n]).convert('L')
                image, target = utils.cut_mix(image, target, image_refer, target_refer)

            # 以 80% 的概率随机调整图像和标签的尺寸
            if (random_gen.random() < 0.8) and self.args.transform_rand_resize:
                rand_h = (random_gen.random() * 1.5) + 0.5  # [0.5, 2.0]
                rand_w = (random_gen.random() * 1.5) + 0.5
                resize_h = int((self.args.input_size[0] * rand_h).__round__())
                resize_w = int((self.args.input_size[1] * rand_w).__round__())

                image = tf.resize(image, [resize_h, resize_w])
                target = tf.resize(target, [resize_h, resize_w], interpolation=InterpolationMode.NEAREST)

            # 随机裁剪到尺寸self.args.transform_rand_crop（一定执行，无概率）
            if hasattr(self.args, 'transform_rand_crop'):
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(int(self.args.transform_rand_crop), int(self.args.transform_rand_crop)))
                image = tf.crop(image, i, j, h, w)
                target = tf.crop(target, i, j, h, w)

            # 以 50% 的概率对图像和标签进行水平翻转。
            if (random_gen.random() < 0.5) and self.args.transform_hflip:
                image = tf.hflip(image)
                target = tf.hflip(target)

            # 以 80% 的概率对图像进行颜色扰动
            if (random_gen.random() < 0.8) and self.args.transform_jitter:
                transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                image = transform(image)

            # 以 50% 的概率对图像应用高斯模糊，模糊核大小随机生成。
            if (random_gen.random() < 0.5) and self.args.transform_blur:
                kernel_size = int((random.random() * 10 + 2.5).__round__())    # random kernel size 3 to 11
                if kernel_size % 2 == 0:
                    kernel_size -= 1
                transform = transforms.GaussianBlur(kernel_size=kernel_size)
                image = transform(image)

            # recommend to use at the end.以 30% 的概率对图像和标签应用仿射变换（RandomPerspective），模拟拍摄角度的变化。
            if (random_gen.random() < 0.3) and self.args.transform_perspective:
                start_p, end_p = transforms.RandomPerspective.get_params(image.width, image.height, distortion_scale=0.5)
                image = tf.perspective(image, start_p, end_p)
                target = tf.perspective(target, start_p, end_p, interpolation=InterpolationMode.NEAREST)

        # 图像和标签转换为张量
        image_tensor = tf.to_tensor(image)
        target_tensor = torch.tensor(np.array(target))

        if self.args.input_space == 'GR':   # grey, red
            image_tensor_r = image_tensor[0].unsqueeze(0)
            image_tensor_grey = tf.to_tensor(tf.to_grayscale(image))

            image_tensor = torch.cat((image_tensor_r, image_tensor_grey), dim=0)

        # 'mean' and 'std' are acquired by cropped face from sense-time landmark
        # 图像归一化
        if self.args.input_space == 'RGB':
            image_tensor = tf.normalize(image_tensor,
                                        mean=self.image_mean,
                                        std=self.image_std)

        # 标签二值化
        if self.args.n_classes <= 2:  # for visualized binary GT
            target_tensor[target_tensor < 128] = 0
            target_tensor[target_tensor >= 128] = 1

        # 为了确保标签的维度和模型输出的维度匹配，将标签的维度升高，增加一个通道维度（unsqueeze(0)）
        target_tensor = target_tensor.unsqueeze(0)    # expand 'grey channel' for loss function dependency

        # 返回处理后的图像张量和标签张量。
        return image_tensor, target_tensor

    def __getitem__(self, index):
        img_x_tr, img_y_tr = self.transform(self.img_x[index], self.img_y[index])

        return (img_x_tr, self.img_x_path[index]), (img_y_tr, self.img_y_path[index])

    def __len__(self):
        return len(self.img_x_path)


class Image2ImageLoader_zero_pad(Dataset):

    def __init__(self,
                 x_path,
                 y_path,
                 mode,
                 **kwargs):

        self.mode = mode
        self.args = kwargs['args']

        # 使用了 ImageNet 预训练模型的默认归一化标准，这些值是根据 ImageNet 数据集的像素统计计算得出的
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

        # 获取图像和标签路径下的所有图片文件名（用is_image()过滤掉非图像文件）
        x_img_name = os.listdir(x_path)
        y_img_name = os.listdir(y_path)
        x_img_name = filter(is_image, x_img_name)
        y_img_name = filter(is_image, y_img_name)

        # 准备两个列表来保存每张图和其对应标签的完整路径
        self.img_x_path = []
        self.img_y_path = []

        # 字典序排序图像名，为了确保图像和标签对齐
        x_img_name = sorted(x_img_name)
        y_img_name = sorted(y_img_name)

        # 把图像文件名与标签文件名一一配对，生成完整路径
        img_paths = zip(x_img_name, y_img_name)
        for item in img_paths:
            self.img_x_path.append(x_path + os.sep + item[0])
            self.img_y_path.append(y_path + os.sep + item[1])

        # 检查图像和标签数量是否一致
        assert len(self.img_x_path) == len(self.img_y_path), 'Images in directory must have same file indices!!'

        # 把所有图像和标签一次性加载到内存中，原图像转为 RGB，标签转为灰度
        print(f'{utils.Colors.LIGHT_RED}Mounting data on memory...{self.__class__.__name__}:{self.mode}{utils.Colors.END}')
        self.img_x = []
        self.img_y = []
        for index in range(len(self.img_x_path)):
            x_path = self.img_x_path[index]
            y_path = self.img_y_path[index]
            self.img_x.append(Image.open(x_path).convert('RGB'))
            self.img_y.append(Image.open(y_path).convert('L'))

    # 图像增强(叠加在线增强,每次执行增强后就立即送入网络，不保留中间文件，数量上没有扩增，且原图没有送入到网络过，都是叠加增强后送入网络）
    def transform(self, image, target):
        # 在处理验证时不做增强，仅会执行中心填充（center_padding），没有改变图像的比例，确保图像大小与目标输入大小匹配。
        if self.mode == 'validation':
            image = utils.center_padding(image, [int(self.args.input_size[0]), int(self.args.input_size[1])])
            target = utils.center_padding(target, [int(self.args.input_size[0]), int(self.args.input_size[1])])

        # 下面的内容和类Image2ImageLoader_resize(Dataset)的transform函数完全一样
        # 如果当前不是验证模式（即是训练模式），则执行后续的数据增强操作。
        if not self.mode == 'validation':
            random_gen = random.Random()  # thread-safe random, 创建随机生成器

            # 以 80% 的概率执行 CutMix；将图像的一部分替换为另一张图的内容（含标签）
            if (random_gen.random() < 0.8) and self.args.transform_cutmix:
                rand_n = random_gen.randint(0, self.__len__() - 1)     # randomly generates reference image on dataset
                image_refer = Image.open(self.img_x_path[rand_n]).convert('RGB')
                target_refer = Image.open(self.img_y_path[rand_n]).convert('L')
                image, target = utils.cut_mix(image, target, image_refer, target_refer)

            # 以 80% 概率执行随机缩放
            if (random_gen.random() < 0.8) and self.args.transform_rand_resize:
                rand_h = (random_gen.random() * 0.5) + 0.75  # [0.75, 1.25]
                rand_w = (random_gen.random() * 0.5) + 0.75
                resize_h = int((self.args.input_size[0] * rand_h).__round__())
                resize_w = int((self.args.input_size[1] * rand_w).__round__())

                image = tf.resize(image, [resize_h, resize_w])
                target = tf.resize(target, [resize_h, resize_w], interpolation=InterpolationMode.NEAREST)

            # 随机裁剪到尺寸self.args.transform_rand_crop（一定执行，无概率）
            if hasattr(self.args, 'transform_rand_crop'):
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(int(self.args.transform_rand_crop), int(self.args.transform_rand_crop)))
                image = tf.crop(image, i, j, h, w)
                target = tf.crop(target, i, j, h, w)

            # 水平翻转
            if (random_gen.random() < 0.5) and self.args.transform_hflip:
                image = tf.hflip(image)
                target = tf.hflip(target)

            # 80% 概率对亮度、对比度、饱和度、色调做颜色扰动 Color Jitter,只影响图像，不影响标签
            if (random_gen.random() < 0.8) and self.args.transform_jitter:
                transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                image = transform(image)

            # 50% 概率执行模糊
            if (random_gen.random() < 0.5) and self.args.transform_blur:
                kernel_size = int((random.random() * 10 + 2.5).__round__())    # random kernel size 3 to 11
                if kernel_size % 2 == 0:
                    kernel_size -= 1
                transform = transforms.GaussianBlur(kernel_size=kernel_size)
                image = transform(image)

            # recommend to use at the end. 30% 概率执行仿射变换（模拟拍摄角度变化）,变换图像和标签；
            if (random_gen.random() < 0.3) and self.args.transform_perspective:
                start_p, end_p = transforms.RandomPerspective.get_params(image.width, image.height, distortion_scale=0.5)
                image = tf.perspective(image, start_p, end_p)
                target = tf.perspective(target, start_p, end_p, interpolation=InterpolationMode.NEAREST)

        # 图像转原换为Tensor，标签变成 LongTensor
        image_tensor = tf.to_tensor(image)
        target_tensor = torch.tensor(np.array(target))

        if self.args.input_space == 'GR':   # grey, red
            image_tensor_r = image_tensor[0].unsqueeze(0)
            image_tensor_grey = tf.to_tensor(tf.to_grayscale(image))

            image_tensor = torch.cat((image_tensor_r, image_tensor_grey), dim=0)

        # 标准的**图像归一化（Normalization）**操作
        # 'mean' and 'std' are acquired by cropped face from sense-time landmark
        if self.args.input_space == 'RGB':
            image_tensor = tf.normalize(image_tensor,
                                        mean=self.image_mean,
                                        std=self.image_std)

        # 标签二值化（针对二分类）
        if self.args.n_classes <= 2:  # for visualized binary GT
            target_tensor[target_tensor < 128] = 0
            target_tensor[target_tensor >= 128] = 1
        # 标签升维成 (1, H, W),保证标签 shape 和模型输出 shape 匹配
        target_tensor = target_tensor.unsqueeze(0)    # expand 'grey channel' for loss function dependency

        return image_tensor, target_tensor

    # 返回一个元组：图像张量 + 图像路径，标签张量 + 标签路径
    def __getitem__(self, index):
        # pdb.set_trace()
        img_x_tr, img_y_tr = self.transform(self.img_x[index], self.img_y[index])
        return (img_x_tr, self.img_x_path[index]), (img_y_tr, self.img_y_path[index])

    # 返回数据集大小
    def __len__(self):
        return len(self.img_x_path)


# Image2ImageDataLoader_resize和类Image2ImageDataLoader_zero_pad中，除了self.image_loader = Image2ImageLoader_zero_pad不一样，其他地方完全一样
class Image2ImageDataLoader_resize:

    def __init__(self,
                 x_path,
                 y_path,
                 mode,
                 batch_size=4,
                 num_workers=0,
                 pin_memory=True,
                 **kwargs):

        g = torch.Generator()
        g.manual_seed(3407)

        self.image_loader = Image2ImageLoader_resize(x_path,
                                                     y_path,
                                                     mode=mode,
                                                     **kwargs)

        # use your own data loader
        # shuffle=(not mode == 'validation') ⇒说明是验证集 ⇒ shuffle=False ⇒ 不打乱顺序，否则是训练集 ⇒ 打乱 ⇒ shuffle=True
        self.Loader = MultiEpochsDataLoader(self.image_loader,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=(not mode == 'validation'),
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            pin_memory=pin_memory)

    def __len__(self):
        return self.image_loader.__len__()


class Image2ImageDataLoader_zero_pad:

    def __init__(self,
                 x_path,
                 y_path,
                 mode,
                 batch_size=4,
                 num_workers=0,
                 pin_memory=True,
                 **kwargs):

        g = torch.Generator()
        g.manual_seed(3407)

        # 实例化底层图像加载器
        self.image_loader = Image2ImageLoader_zero_pad(x_path,
                                                       y_path,
                                                       mode=mode,
                                                       **kwargs)

        # use your own data loader
        # shuffle=(not mode == 'validation') ⇒说明是验证集 ⇒ shuffle=False ⇒ 不打乱顺序，否则是训练集 ⇒ 打乱 ⇒ shuffle=True
        # 构建多 epoch 支持的数据加载器
        self.Loader = MultiEpochsDataLoader(self.image_loader,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            shuffle=(not mode == 'validation'),
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            pin_memory=pin_memory)

    def __len__(self):
        return self.image_loader.__len__()
