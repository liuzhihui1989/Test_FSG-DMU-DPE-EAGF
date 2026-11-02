import torch
import time
import os
import wandb
import swanlab as wandb
from fvcore.nn import FlopCountAnalysis
import numpy as np
import sys
import math

from models import dataloader as dataloader_hub
from models import lr_scheduler
from models import model_implements
from models import losses as loss_hub
from models import metrics
from models import utils


class Trainer_seg:
    # 添加计算模型参数量的函数
    def count_params(self, model):
        # 计算模型中所有可训练的参数的总数
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_in_million = total_params / 1e6  # 将参数量转换为百万
        return total_params_in_million

        # 添加计算 FLOPs 的函数

    def count_flops(self, model, input_tensor):
        # 计算 FLOPs
        flops = FlopCountAnalysis(model, input_tensor)
        return flops.total() / 1e9  # 将 FLOPs 转换为 GFLOPs

    # 初始化
    def __init__(self, args, now_time=None):
        self.start_time = time.time()
        self.args = args

        # save hyper-parameters
        if not self.args.debug:
            with open(self.args.config_path, 'r', encoding='utf-8') as f_r:
                file_path = self.args.saved_model_directory + '/' + now_time
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                with open(os.path.join(file_path, self.args.config_path.split('/')[-1]), 'w', encoding='utf-8') as f_w:
                    f_w.write(f_r.read())

        if args.wandb:
            # os.environ['WANDB_MODE'] = 'offline'  # 添加这一行，设置为离线模式(用于国内恒源云运行的场景)
            # # wandb.login(key='871de999c5c87f7824ab65afd8af8cbf4ef53ee3')
            # wandb.init(project='{}'.format(args.project_name), config=args, name=now_time,
            #            settings=wandb.Settings(start_method="thread"))
            wandb.login(api_key='NQjlbMtKUQSCxurqaeFfW')
            wandb.init(project=args.project_name,
                       config=args,
                       name=now_time,
                       mode="online")

        # Check cuda available and assign to device
        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        # 'init' means that this variable must be initialized.
        # 'set' means that this variable is available of being set, not must.
        # 调用自定义的__init_data_loader函数来加载训练和验证数据。
        self.loader_train = self.__init_data_loader(self.args.train_x_path,
                                                    self.args.train_y_path,
                                                    self.args.batch_size,
                                                    mode='train')
        self.loader_val = self.__init_data_loader(self.args.val_x_path,
                                                  self.args.val_y_path,
                                                  batch_size=1,
                                                  mode='validation')

        #  初始化模型和优化器
        self.model = self.init_model(self.args.model_name, self.device, self.args)

        # 在模型初始化后计算并输出参数量
        self.model = Trainer_seg.init_model(self.args.model_name, self.device, self.args)  # 获取模型
        params = self.count_params(self.model)  # 通过self调用count_params
        print(f'Total Parameters: {params:.2f} Million')  # 输出参数量

        # 创建一个假输入，确保维度与模型输入一致
        input_tensor = torch.randn(1, 3, 608, 608).to(self.device)  # 假设输入为 3x608x608的图像
        flops = self.count_flops(self.model, input_tensor)  # 计算 FLOPs
        print(f'Total FLOPs: {flops:.2f} GFLOPs')  # 输出 FLOPs

        self.optimizer = self.__init_optimizer(self.args.optimizer, self.model, self.args.lr)
        self.scheduler = self.__set_scheduler(self.optimizer, self.args.scheduler, self.loader_train, self.args.batch_size)

        # 加载预训练模型（如指定了 model_path）
        if hasattr(self.args, 'model_path'):
            if self.args.model_path != '':
                self.model.load_state_dict(torch.load(self.args.model_path), strict=False)
                print('Model loaded successfully!!! (Custom)')
                self.model.to(self.device)
        # 定义损失函数
        self.criterion = self.__init_criterion(self.args.criterion)

        # wandb 观察模型
        if self.args.wandb and self.args.mode == 'train':
            if hasattr(wandb, "watch"):  # 只有 wandb 时才执行
                wandb.watch(self.model)

        # 设置模型保存路径。
        self.saved_model_directory = self.args.saved_model_directory + '/' + now_time

        self.metric_best = {'f1_score': 0}  # the 'value' follows the metric on validation
        self.model_post_path_dict = {}
        self.last_saved_epoch = 0

        self.__validate_interval = 1 if (self.loader_train.__len__() // self.args.train_fold) == 0 else self.loader_train.__len__() // self.args.train_fold

        # 设置训练和迭代的回调函数
        self.callback = utils.TrainerCallBack()
        if hasattr(self.model.module, 'train_callback'):
            self.callback.train_callback = self.model.module.train_callback
            print('train_callback adapted.')
        if hasattr(self.model.module, 'iteration_callback'):
            self.callback.iteration_callback = self.model.module.iteration_callback
            print('iteration_callback adapted.')

    # 负责执行当前轮的训练
    def _train(self, epoch):
        self.model.train()
        self.callback.train_callback()
        batch_losses = []
        f1_list = []

        print('Start Train')

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            num_gpus = 1  # 用1避免除零错误
        # 计算训练集数量
        total_samples = len(self.loader_train.Loader.dataset)
        print("读取原训练集得到总样本数：", total_samples)


        print('Start Train')
        # batch_idx：当前是第几个 batch（从 0 开始计数），按批次batch遍历训练数据集，每读取一个批次数据，都会自动调用dataloader的getitem函数
        # (x_in, target) 是 一个 batch 的训练数据，来自__getitem__() 方法。
        # __getitem__() 返回结构是：x_in = (image_tensor原图张量, image_path);  target = (target_tensor标签张量, label_path)
        # 由MultiEpochsDataLoader中的参数shuffle=(not mode == 'validation')决定：训练时乱序取图片
        for batch_idx, (x_in, target) in enumerate(self.loader_train.Loader):
            self.callback.iteration_callback()
            # if (x_in[0].shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():   # if has 1 batch per GPU
            #     break   # avoid BN issue
           # print(f"Batch {batch_idx}")
            # 只取张量部分，不取路径字符串
            x_in, _ = x_in
            target, _ = target
            x_in = x_in.to(self.device)
            target = target.long().to(self.device)

            # 前向传播，会跳转到model_implements.py代码中的class FSGNet的forward函数，调用FSGNet模型。
            # x_in是输入图像，output是输出预测分割结果。
            output = self.model(x_in)

            # # 如果 output 是一个元组（tuple）或者列表（list），说明模型输出了多个结果（通常是不同深度的输出）
            # if isinstance(output, tuple) or isinstance(output, list):  # condition for deep supervision
            #     # 对每一个输出 item 计算与目标 target 的损失（用 self.criterion() 计算），然后求平均。也就是说，把多个输出的损失取平均，作为最终的损失值。
            #
            #     loss = sum([self.criterion(item, target) for item in output]) / len(output)
            #     # 取输出中的第一个作为最终输出，深度监督的辅助输出只用于训练优化，最终的推理结果通常仍然只取最深层的主输出
            #     output = output[0]  # please check that the first element of tuple is not interpolated in network.
            if isinstance(output, (tuple, list)):
                out_fused, out_1, out_2, out_3 = output

                # 只用 out_1, out_2, out_3 计算 loss（深度监督）
                loss = sum([self.criterion(o, target) for o in [out_1, out_2, out_3]]) / 3

                # 推理/评估时使用融合后的最终输出
                output = out_fused
            else:
                # 如果 output 不是元组或列表（即模型只输出一个结果），直接计算这个输出与目标之间的损失
                loss = self.criterion(output, target)

            if not torch.isfinite(loss):
                raise Exception('Loss is NAN. End training.')

            # 每次反向传播之前都要调用 zero_grad() 清除上一步的梯度缓存
            self.optimizer.zero_grad()
            # 反向传播，计算梯度
            loss.backward()
            # 使用选定的优化器，根据刚才算出的梯度来更新参数
            self.optimizer.step()
            # 如果设置了调度器，根据当前的 step 或 epoch 动态调整学习率
            if self.scheduler is not None:
                self.scheduler.step()

            # output 是模型预测结果（通常是一个概率图），把大于 0.5 的概率当作正类，变成 1；否则变成 0。
            output_argmax = torch.where(output > 0.5, 1, 0).cpu()
            # 计算评价指标
            metric_result = metrics.metrics_np(output_argmax[:, 0], target.squeeze(0).detach().cpu().numpy(), b_auc=False)
            # 记录 F1 分数
            f1_list.append(metric_result['f1'])
            # 记录 loss
            batch_losses.append(loss.item())

            # 如果设置了train_fold参数（表示多少个batch验证一次）；就在指定间隔处做一次验证。
            if hasattr(self.args, 'train_fold'):
                if batch_idx != 0 and (batch_idx % self.__validate_interval) == 0:
                    self._validate(self.model, epoch)

            # 每隔一定步数，打印一次训练日志：当前 epoch，损失函数类型，当前平均 loss，当前学习率
            if (batch_idx != 0) and (batch_idx % (self.args.log_interval // self.args.batch_size) == 0):
                loss_mean = np.mean(batch_losses)  #当前 epoch 开始以来，从第一个 batch 到当前 batch 为止 所有累计的损失的平均值。
                print('{} epoch / Train Loss {} : {}, lr {}'.format(epoch,
                                                                    self.args.criterion,
                                                                    loss_mean,
                                                                    self.optimizer.param_groups[0]['lr']))

        loss_mean = np.mean(batch_losses) #当前epoch轮训练结束时，当前轮整体平均损失
        print('{} epoch / Train Loss {} : {}, lr {}'.format(epoch,
                                                            self.args.criterion,
                                                            loss_mean,
                                                            self.optimizer.param_groups[0]['lr']))

        f1_score = sum(f1_list) / len(f1_list) #当前 epoch 训练集上的 F1 分数
        print('{} epoch / Train f1_score: {}'.format(epoch, f1_score))

        # 如果开启了 Weights & Biases（wandb）日志记录功能，就把当前训练轮次（epoch）的指标上传到 wandb 平台。
        if self.args.wandb:
            wandb.log({'Train Loss {}'.format(self.args.criterion): loss_mean,
                       'Train f1_score': f1_score},
                      step=epoch)

    # 验证函数
    def _validate(self, model, epoch):
        model.eval() # 设置为验证模式（比如关闭 Dropout、BatchNorm 的更新）
        f1_list = []

        with torch.no_grad():
            # 遍历验证数据
            for batch_idx, (x_in, target) in enumerate(self.loader_val.Loader):
                x_in, _ = x_in
                target, _ = target
                x_in = x_in.to(self.device)
                target = target.long().to(self.device)

                output = model(x_in)

                if isinstance(output, tuple) or isinstance(output, list):  # condition for Deep supervision
                    output = output[0]

                output_argmax = torch.where(output > 0.5, 1, 0).cpu()
                metric_result = metrics.metrics_np(output_argmax[:, 0], target.squeeze(0).detach().cpu().numpy(), b_auc=False)
                # 计算当前 batch 的 F1 分数
                f1_list.append(metric_result['f1'])

        # 汇总所有 batch 的平均 F1 分数
        f1_score = sum(f1_list) / len(f1_list)
        print('{} epoch / Val f1_score: {}'.format(epoch, f1_score))

        if self.args.wandb:
            wandb.log({'Val f1_score': f1_score},
                      step=epoch)

        model_metrics = {'f1_score': f1_score}

        # 遍历所有指标（目前只有一个：f1_score），保存当前最优模型
        for key in model_metrics.keys():
            if model_metrics[key] - self.metric_best[key] > 0.001:
                self.metric_best[key] = model_metrics[key]
                self.save_model(model, self.args.model_name, epoch, model_metrics[key], best_flag=True, metric_name=key)

        # 连续self.args.cycles * 4轮训练对应的验证集上性能没有提升，就提前停止(先用 cycles=25~50 做个试验，看模型什么时候停止，视情况再调整。)
        if (epoch - self.last_saved_epoch) > self.args.cycles * 4:
            print('The model seems to be converged. Early stop training.')
            print(f'Best F1 -----> {self.metric_best["f1_score"]}')
            wandb.log({f'Best F1': self.metric_best['f1_score']},
                      step=epoch)
            sys.exit()  # safe exit

    # 启动训练
    def start_train(self):
        for epoch in range(1, self.args.epoch + 1):
            self._train(epoch) #负责执行当前轮训练
            self._validate(self.model, epoch)  #在每一轮训练后进行模型验证。

            print('### {} / {} epoch ended###'.format(epoch, self.args.epoch))

    # 保存模型
    def save_model(self, model, model_name, epoch, metric=None, best_flag=False, metric_name='metric'):
        file_path = self.saved_model_directory + '/'

        file_format = file_path + model_name + '-Epoch_' + str(epoch) + '-' + metric_name + '_' + str(metric) + '.pt'

        if not os.path.exists(file_path):
            os.mkdir(file_path)

        if best_flag:
            if metric_name in self.model_post_path_dict.keys():
                os.remove(self.model_post_path_dict[metric_name])
            self.model_post_path_dict[metric_name] = file_format

        torch.save(model.state_dict(), file_format)


        print(file_format + '\t model saved!!')
        self.last_saved_epoch = epoch

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

    @staticmethod
    def init_model(model_name, device, args):
        model = getattr(model_implements, model_name)(**vars(args)).to(device)

        return torch.nn.DataParallel(model)

    def __init_criterion(self, criterion_name):
        criterion = getattr(loss_hub, criterion_name)().to(self.device)

        return criterion

    def __init_optimizer(self, optimizer_name, model, lr):
        optimizer = None

        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.weight_decay)

        return optimizer

    def __set_scheduler(self, optimizer, scheduler_name, data_loader, batch_size):
        scheduler = None
        # 每个 epoch 的训练步数，通常等于 总训练样本数 / batch_size
        steps_per_epoch = math.ceil((data_loader.__len__() / batch_size))

        if hasattr(self.args, 'scheduler'):
            # self.args.warmup_epoch是定义的预热 epoch 数量，即模型在正式训练前使用较小学习率进行预热的 epoch 数。
            if scheduler_name == 'WarmupCosine':
                scheduler = lr_scheduler.WarmupCosineSchedule(optimizer=optimizer,
                                                              warmup_steps=steps_per_epoch * self.args.warmup_epoch,
                                                              t_total=self.args.epoch * steps_per_epoch,
                                                              cycles=self.args.epoch / self.args.cycles,
                                                              last_epoch=-1)
            elif scheduler_name == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.cycles, eta_min=self.args.lr / 100)   # min: lr / 100, max: lr
            elif scheduler_name == 'ConstantLRSchedule':
                scheduler = lr_scheduler.ConstantLRSchedule(optimizer, last_epoch=-1)
            elif scheduler_name == 'WarmupConstantSchedule':
                scheduler = lr_scheduler.WarmupConstantSchedule(optimizer, warmup_steps=steps_per_epoch * self.args.warmup_epoch)
            else:
                raise Exception('No scheduler named', scheduler_name)
        else:
            pass

        return scheduler
