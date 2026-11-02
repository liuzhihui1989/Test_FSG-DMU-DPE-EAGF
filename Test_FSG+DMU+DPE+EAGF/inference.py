import torch
import time
import numpy as np
import os

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from models import metrics
from models import utils
from models import dataloader as dataloader_hub
from models import model_implements
from train import Trainer_seg
from PIL import Image


class Inferencer:
    def __init__(self, args):
        self.start_time = time.time()
        self.args = args

        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        # 调用自定义函数加载验证数据。mode='validation' 告诉 数据加载器我要加载验证集来跑推理。
        self.loader_form = self.__init_data_loader(self.args.val_x_path,
                                                   self.args.val_y_path,
                                                   batch_size=1,
                                                   mode='validation')

        # self.model = Trainer_seg.init_model(self.args.model_name, self.device, self.args)
        # self.model.load_state_dict(torch.load(args.model_path))
        # self.model.eval()

        # 加载已训练好的模型权重，并设置为评估模式 eval()（不进行 dropout/batchnorm 的训练行为）
        self.device = torch.device('cuda' if self.args.cuda and torch.cuda.is_available() else 'cpu')
        self.model = Trainer_seg.init_model(self.args.model_name, self.device, self.args)
        self.model.load_state_dict(torch.load(args.model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        # 初始化评价指标
        self.metric = self._init_metric(self.args.task, self.args.n_classes)

        self.image_mean = self.loader_form.image_loader.image_mean
        self.image_std = self.loader_form.image_loader.image_std
        self.fn_list = []

    def plot_roc_curve(self, fpr, tpr, auc_value):
        # 绘制 ROC 曲线
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_value:.4f})')

        # 绘制对角线
        # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        # 添加标题、标签和图例
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')

        # 显示图像
        plt.tight_layout()
        plt.savefig('roc_curve.png')
        plt.show()

    # def visualize_feature_map(self, feature_map, name, save_dir):
    #     os.makedirs(save_dir, exist_ok=True)
    #
    #     if isinstance(feature_map, torch.Tensor):
    #         feature_map = feature_map.detach().cpu()
    #
    #     # 默认取第一张图
    #     fmap = feature_map[0]
    #
    #     # 对多通道特征图，做通道平均
    #     if fmap.ndim == 3:
    #         fmap = fmap.mean(0)
    #
    #     # 归一化到 0-1
    #     fmap -= fmap.min()
    #     fmap /= (fmap.max() + 1e-6)
    #
    #     # 保存图像
    #     plt.imshow(fmap, cmap='jet')
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(save_dir, f"{name}.png"), bbox_inches='tight', pad_inches=0)
    #     plt.close()

    def visualize_feature_map(self, feature_map, name, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        if isinstance(feature_map, torch.Tensor):
            feature_map = feature_map.detach().cpu()

        fmap = feature_map[0]

        if fmap.ndim == 3:
            fmap = fmap.mean(0)

        fmap -= fmap.min()
        fmap /= (fmap.max() + 1e-6)

        # 直接保存为 608×608
        plt.imsave(
            os.path.join(save_dir, f"{name}.png"),
            fmap.numpy(),
            cmap='jet'
        )

    def start_inference_segmentation(self):
        # 记录推理时间
        total_inference_time = 0
        num_images = 0

        # 初始化评估结果列表, 存放每张图像的指标结果
        f1_list = []
        acc_list = []
        auc_list = []
        sen_list = []
        spe_list = []
        pre_list = []
        mcc_list = []

        all_prob = []  # 保存所有图片的预测概率（flatten 后）
        all_target = []  # 保存所有图片的真实标签（flatten 后）

        # 遍历验证集，执行推理，执行for的时候会自动调用dataloader.py文件中的Image2ImageLoader_zero_pad的__getitem__函数，此时原图和标签的尺寸都变为self.args.input_size尺寸
        for batch_idx, (img, target) in enumerate(self.loader_form.Loader):
            # 调试用
            print(f"Batch {batch_idx}")

            with torch.no_grad():
                x_in, img_id = img
                target, origin_size = target

                x_in = x_in.to(self.device)
                x_img = x_in
                target = target.long().to(self.device)

                # 记录推理开始时间
                start_time = time.time()

                output = self.model(x_in) #此处会跳转到FSGNet模型。把图像送入模型，得到预测结果 output

                # 记录推理结束时间
                end_time = time.time()
                # 计算本次推理时间
                inference_time = end_time - start_time
                total_inference_time += inference_time
                num_images += 1

                # 将out_fused结果作为最终输出结果
                if isinstance(output, tuple) or isinstance(output, list):  # condition for deep supervision
                    output = output[0]

                # 调用后处理函数
                metric_result = self.post_process(output, target, x_img, img_id)
                # 返回评估指标，并将它们逐个保存下来。
                f1_list.append(metric_result['f1'])
                acc_list.append(metric_result['acc'])
                auc_list.append(metric_result['auc'])
                sen_list.append(metric_result['sen'])
                spe_list.append(metric_result['spe'])
                pre_list.append(metric_result['pre'])
                mcc_list.append(metric_result['mcc'])

                # 1. 取出预测的概率（output shape: [1,1,H,W]）
                output_np = output.cpu().numpy().squeeze()  # (H,W)
                prob = 1 / (1 + np.exp(-output_np))  # sigmoid
                # 2. 取出 ground truth
                target_np = target.cpu().numpy().squeeze()  # (H,W)
                # 3. flatten 到一维
                all_prob.append(prob.flatten())
                all_target.append(target_np.flatten())

                # ✅ 只在第一 batch 可视化特征
                if batch_idx == 0:
                    save_dir = './vis_features'
                    save_order = ['input', 'x1_down', 'x2_down', 'x3_down', 'x4', 'x3_up', 'x2_up', 'x1_up', 'output_1',
                                  'output_2', 'output_3', 'f1_prime', 'f2_prime', 'f3_prime', 'alpha','beta','gamma','out_fused']
                    for idx, name in enumerate(save_order):
                        # 兼容 DataParallel 和普通模型
                        model_core = self.model.module if hasattr(self.model, "module") else self.model
                        temp = model_core.FSGNet.feature_vis
                        if name in temp:
                            self.visualize_feature_map(temp[name], f"{idx + 1:02d}_{name}", save_dir)


        # 计算平均推理时间 (AIT)
        ait = total_inference_time / num_images
        print(f'Average Inference Time (AIT) per image: {ait:.6f} seconds')

        # 计算 ROC 曲线并绘制
        all_prob = np.concatenate(all_prob)  # 长度：所有像素数和
        all_target = np.concatenate(all_target)
        # 计算 FPR 和 TPR
        fpr, tpr, _ = roc_curve(all_target, all_prob)

        roc_auc = auc(fpr, tpr)# 计算 AUC

        # 保存 ROC 数据用于之后汇总绘图
        os.makedirs("roc_data", exist_ok=True)
        np.save(f"roc_data/{self.args.model_name}_fpr.npy", fpr)
        np.save(f"roc_data/{self.args.model_name}_tpr.npy", tpr)
        np.save(f"roc_data/{self.args.model_name}_auc.npy", np.array([roc_auc]))

        self.plot_roc_curve(fpr, tpr, roc_auc)
        print(f"Dataset-level AUC: {roc_auc:.4f}")


        metrics = self.metric.get_results()
        cIoU = [metrics['Class IoU'][i] for i in range(self.args.n_classes + 1)]
        mIoU = sum(cIoU) / (self.args.n_classes + 1)

        print('mean mIoU', mIoU)
        print('IoU', metrics['Class IoU'][1])
        print('mean F1 score:', sum(f1_list) / len(f1_list))
        print('mean Accuracy', sum(acc_list) / len(acc_list))
        print('mean AUC', sum(auc_list) / len(auc_list))
        print('mean Sensitivity', sum(sen_list) / len(sen_list))
        print('mean Specificity', sum(spe_list) / len(spe_list))
        print('mean Precision', sum(pre_list) / len(pre_list))
        print('mean MCC', sum(mcc_list) / len(mcc_list))

    def post_process(self, output, target, x_img, img_id):
        # 将一个 PyTorch 张量转换为一个 NumPy 数组，并进行一系列的标准化和反标准化操作，最后转换成一个原始图像格式，可以用来显示或保存。
        x_img = x_img.squeeze(0).data.cpu().numpy()
        x_img = np.transpose(x_img, (1, 2, 0))
        x_img = x_img * np.array(self.image_std)
        x_img = x_img + np.array(self.image_mean)
        x_img = x_img * 255.0
        x_img = x_img.astype(np.uint8)

        # 去除 padding
        output = utils.remove_center_padding(output)
        target = utils.remove_center_padding(target)

        # 预测结果二值化（大于 0.5 为 1，否则为 0），适用于二分类掩码分割。
        output_argmax = torch.where(output > 0.5, 1, 0).cpu().detach()
        self.metric.update(target.squeeze(1).cpu().detach().numpy(), output_argmax.numpy())

        path, fn = os.path.split(img_id[0])
        img_id, ext = os.path.splitext(fn)
        dir_path, fn = os.path.split(self.args.model_path)
        fn, ext = os.path.splitext(fn)
        save_dir = dir_path + '/' + fn + '/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        Image.fromarray(x_img).save(save_dir + img_id + '.png', quality=100)
        Image.fromarray((output_argmax.squeeze().numpy() * 255).astype(np.uint8)).save(save_dir + img_id + f'_argmax.png', quality=100)
        # Image.fromarray(output_heatmap.astype(np.uint8)).save(save_dir + img_id + f'_heatmap_overlay.png', quality=100)

        # 计算当前单张图像的指标（F1、ACC、AUC、MCC、敏感性等
        metric_result = metrics.metrics_np(output_argmax[None, :], target.squeeze(0).detach().cpu().numpy(), b_auc=True)
        print(f'{img_id} \t Done !!')

        #  将output输出为灰度明暗变化的图。output 是一个 PyTorch tensor
        # output 形状为 (1, 1, 512, 512)
        output_numpy = output.cpu().detach().numpy()
        # 去掉多余维度，得到 (512, 512)
        output_2d = output_numpy.squeeze()  # squeeze 会自动去掉长度为1的维度
        # 将概率值缩放到 0-255
        output_normalized = (output_2d * 255.0).astype(np.uint8)
        # 保存为灰度图
        Image.fromarray(output_normalized).save(save_dir + img_id + '_output_gray.png', quality=100)


        return metric_result

    # 在inference时候就没有调用。注释掉也没报错
    def __init_model(self, model_name):
        if model_name == 'UNet':
            model = model_implements.UNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'UNet2P':
            model = model_implements.UNet2P(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'UNet3P_Deep':
            model = model_implements.UNet3P_Deep(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'ResUNet':
            model = model_implements.ResUNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'ResUNet2P':
            model = model_implements.ResUNet2P(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'SAUNet':
            model = model_implements.SAUNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'ATTUNet':
            model = model_implements.ATTUNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'DCSAU_UNet':
            model = model_implements.DCSAU_UNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'AGNet':
            model = model_implements.AGNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'R2UNet':
            model = model_implements.R2UNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'ConvUNeXt':
            model = model_implements.ConvUNeXt(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'FRUNet':
            model = model_implements.FRUNet(n_classes=1, in_channels=self.args.input_channel)
            # 实例化 FSGNet_parts.FSGNet(...)，并传入这些参数：
            # in_channels = self.args.input_channel
            # n_classes = 1
            # 其余参数（depths, base_c, kernel_size）将使用默认值：depths=[3,3,9,3], base_c=64, kernel_size=3
        elif model_name == 'FSGNet':
            model = model_implements.FSGNet(n_classes=1, in_channels=self.args.input_channel)
        else:
            raise Exception('No model named', model_name)

        return torch.nn.DataParallel(model)

    def __init_data_loader(self,
                           x_path,
                           y_path,
                           batch_size,
                           mode):

        if self.args.dataloader == 'Image2Image_zero_pad':
            loader = dataloader_hub.Image2ImageDataLoader_zero_pad(x_path=x_path,
                                                                   y_path=y_path,
                                                                   batch_size=batch_size,
                                                                   num_workers=self.args.worker,
                                                                   pin_memory=self.args.pin_memory,
                                                                   mode=mode,
                                                                   args=self.args)
        elif self.args.dataloader == 'Image2Image_resize':
            loader = dataloader_hub.Image2ImageDataLoader_resize(x_path=x_path,
                                                                 y_path=y_path,
                                                                 batch_size=batch_size,
                                                                 num_workers=self.args.worker,
                                                                 pin_memory=self.args.pin_memory,
                                                                 mode=mode,
                                                                 args=self.args)
        else:
            raise Exception('No dataloader named', self.args.dataloader)

        return loader

    def _init_metric(self, task_name, num_class):
        if task_name == 'segmentation':
            metric = metrics.StreamSegMetrics_segmentation(num_class + 1)
        else:
            raise Exception('No task named', task_name)

        return metric
