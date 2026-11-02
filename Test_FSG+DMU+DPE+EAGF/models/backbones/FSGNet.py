import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F


class TriplePoolingAttention(nn.Module):
    def __init__(self, in_c1, in_c2, mid_c):
        """
        :param in_c1: 引导图 hr_x 的通道数（例如 3）
        :param in_c2: 输入图 lr_y 的通道数（例如 512）
        :param mid_c: 两者统一后的中间通道数（例如 64/128）
        """
        super(TriplePoolingAttention, self).__init__()

        self.conv_hr = nn.Sequential(
            nn.Conv2d(in_c1, mid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_c)
        )
        self.conv_lr = nn.Sequential(
            nn.Conv2d(in_c2, mid_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_c)
        )

        self.combine = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c * 2, mid_c * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_c * 2),
            nn.Sigmoid()  # 输出 shape: (B, 2C, H, W)
        )

    def forward(self, hr_x, lr_y):
        """
        hr_x: 引导图 (B, in_c1, H, W)，如 input_scale_3
        lr_y: 输入图 (B, in_c2, H, W)，如 X3_down
        return: I_a ∈ (B, 1, H, W)
        """
        hr_feat = self.conv_hr(hr_x)  # (B, mid_c, H, W)
        lr_feat = self.conv_lr(lr_y)  # (B, mid_c, H, W)

        feat = torch.cat([hr_feat, lr_feat], dim=1)  # (B, 2*mid_c, H, W)
        att = self.combine(feat)  # (B, 2*mid_c, H, W)

        att_max = torch.amax(att, dim=1, keepdim=True)  # (B, 1, H, W)
        att_mean = torch.mean(att, dim=1, keepdim=True)  # (B, 1, H, W)
        att_min = torch.amin(att, dim=1, keepdim=True)  # (B, 1, H, W)

        I_a = att_max + att_mean - att_min  # (B, 1, H, W)

        return I_a


class AdaptiveConvSelectionBlock(nn.Module):
    def __init__(self, in_channels, mid_channels=61, out_channels=64):
        super(AdaptiveConvSelectionBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=3, dilation=3)

        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.attn_conv = nn.Conv2d(mid_channels * 3, 3, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        # ✅ 新增统一输出通道数的 1×1 卷积
        self.output_conv = nn.Conv2d(in_channels + mid_channels, out_channels, kernel_size=1)

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(x)
        f3 = self.conv3(x)

        feats = torch.cat([f1, f2, f3], dim=1)              # [B, 3*mid_c, H, W]
        pooled = self.global_max_pool(feats)                # [B, 3*mid_c, 1, 1]
        attn = self.attn_conv(pooled)                       # [B, 3, 1, 1]
        attn = self.softmax(attn)

        a1, a2, a3 = torch.chunk(attn, 3, dim=1)

        f1 = f1 * a1
        f2 = f2 * a2
        f3 = f3 * a3
        fused = f1 + f2 + f3                                 # [B, mid_c, H, W]

        out = torch.cat([x, fused], dim=1)                  # [B, in_c + mid_c, H, W]

        # out = self.output_conv(out)                         # ✅ 压缩到 out_channels（默认64）
        return out


class DynamicMergeUnit(nn.Module):
    def __init__(self, in_channels=1, reduction=16):
        super(DynamicMergeUnit, self).__init__()

        # 输入为 1 通道特征图 (B,1,H,W)，先升维到 C
        self.conv1 = nn.Conv2d(in_channels, in_channels * 4, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.in_channels = in_channels * 4  # 升维后通道数

        # 通道注意力模块（类似 CBAM 中 CAM）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Linear(self.in_channels, self.in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(self.in_channels // reduction, self.in_channels, bias=False)

        # 通道注意力融合后的乘法结果
        self.sigmoid_c = nn.Sigmoid()

        # 用于生成空间注意力图 α(x,y)
        self.conv2 = nn.Conv2d(self.in_channels, 1, kernel_size=1)
        self.sigmoid_s = nn.Sigmoid()

    def forward(self, x):
        residual = x  # 原始输入 (B,1,H,W)

        # Step 1: 升维
        x = self.conv1(x)  # (B, C, H, W)

        # Step 2: 通道注意力分支（CAM）
        avg_out = self.avg_pool(x).view(x.size(0), -1)  # (B, C)
        max_out = self.max_pool(x).view(x.size(0), -1)  # (B, C)

        avg_fc = self.fc2(self.relu(self.fc1(avg_out)))  # (B, C)
        max_fc = self.fc2(self.relu(self.fc1(max_out)))  # (B, C)

        attention_c = self.sigmoid_c(avg_fc + max_fc).unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)

        x = x * attention_c  # 通道注意力加权

        # Step 3: 空间注意力图 α(x,y)
        spatial_weight = self.sigmoid_s(self.conv2(x))  # (B, 1, H, W)

        return spatial_weight


class AdaptiveWeightLearning(nn.Module):
    def __init__(self, num_outputs=3):
        super(AdaptiveWeightLearning, self).__init__()
        self.num_outputs = num_outputs
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1x1 = nn.Conv2d(num_outputs * 2, num_outputs, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, outputs):
        # outputs: List of [B, C, H, W] -> concat along channel: [B, num_outputs * C, H, W]
        x = torch.cat(outputs, dim=1)  # [B, 3*C, H, W]
        B, C_total, _, _ = x.size()
        C = C_total // self.num_outputs

        # reshape to list of [B, C, H, W]
        split_outputs = torch.chunk(x, self.num_outputs, dim=1)

        # Get global descriptors
        gap = [self.global_avg_pool(feat) for feat in split_outputs]
        gmp = [self.global_max_pool(feat) for feat in split_outputs]

        gap = torch.cat(gap, dim=1)  # [B, 3C, 1, 1]
        gmp = torch.cat(gmp, dim=1)  # [B, 3C, 1, 1]

        agg = torch.cat([gap, gmp], dim=1)  # [B, 6C, 1, 1]
        weight = self.conv1x1(agg)         # [B, 3, 1, 1]
        weight = self.sigmoid(weight)      # [B, 3, 1, 1]

        # split weights
        alpha, beta, gamma = torch.chunk(weight, self.num_outputs, dim=1)

        # apply weights
        weighted_outputs = [
            alpha * split_outputs[0],
            beta  * split_outputs[1],
            gamma * split_outputs[2]
        ]
        out_fused = sum(weighted_outputs)
        return out_fused, [alpha, beta, gamma]


# DropBlock是一种正则化技术，它随机丢弃一块连续的特征区域（block）
class DropBlock(nn.Module):
    # 定义了一个 2D DropBlock。
    # block_size: 每个被丢弃的区域是一个 block（方形）；
    # p是丢弃概率，通常在训练中使用。
    def __init__(self, block_size=5, p=0.1):
        super().__init__()
        self.block_size = block_size
        self.p = p

    # gamma 控制 有多少个 block 会被丢
    def calculate_gamma(self, x):
        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x):
        N, C, H, W = x.size() #获取输入张量的维度（N 是 batch_size，C 是通道数，H 和 W 是高和宽）
        # DropBlock 只在训练时起作用,测试时不进行 DropBlock，直接返回原始输入。
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
            # 为了后面能与原始图大小匹配，进行对称 padding。
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            # 对 mask 做最大池化，生成遮盖区域
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x

# Spatial Attention（空间注意力）
class SelfAttentionBlock(nn.Module):
    def __init__(self):
        super(SelfAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)
        x3 = torch.cat((x1, x2), dim=1)
        temp=self.conv(x3)
        x4 = torch.sigmoid(temp)
        x = x4 * x
        assert len(x.shape) == 4
        return x

# 输出是一个单通道的注意力权重图，形状为：[batch_size, 1, 高, 宽]。它反映了输入特征图中“哪些位置更重要”。值域是 [0, 1]，因为最后经过了 Sigmoid() 激活。
class CrossAttentionBlock(nn.Module):
    # in_channels 是输入特征图的通道数，要求输入的两个特征图 x 和 g 通道相同。
    def __init__(self, in_channels):
        super(CrossAttentionBlock, self).__init__()
        # in_channels: 表示主输入 x 的通道数。
        # gating_channels: 表示引导信号 g 的通道数。
        # inter_channels: 表示中间变换空间的通道数，决定注意力的维度。
        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels

        # 对输入 x 做变换
        self.theta = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            nn.BatchNorm2d(self.inter_channels),
        )

        # 对 gating signal g 做变换
        self.phi = nn.Sequential(
            nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.inter_channels),
        )

        # 输出注意力权重，降为单通道
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    # x: 当前阶段的高分辨率特征图（In_stage，尺寸较大）
    # g: 上阶段的输出（In_up，尺寸较小）
    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0) #获取输入尺寸并校验 batch 大小一致

        # 对 x 提取特征 theta(x)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # 对 g 提取特征phi(g)后上采样。theta_x_size[2:]表示取出从索引2开始的部分
        # 如果 tmp 的空间尺寸已经 等于 theta_x_size[2:]，那么这行代码会正常执行，但什么都不改变，没有实际插值运算
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=True)

        # 点加两个特征，融合来自 x 和 g 的信息
        f = F.relu(theta_x + phi_g, inplace=True)

        # 计算注意力权重图
        psi_f = self.psi(f)

        return psi_f

# 一个卷积块，包含三步操作，二维卷积操作-->批标准化-->激活函数
class M_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super(M_Conv, self).__init__()
        # pad_size = kernel_size // 2 是一种通用技巧，确保stride=1 的情况下，输出大小与输入一致
        pad_size = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=pad_size, stride=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    # 将 x 送入我们在 __init__ 中定义的卷积块 self.conv，并将结果输出。
    def forward(self, x):
        x = self.conv(x)
        return x

# 沿 x 方向（宽度方向）计算差分
# input: 4D Tensor  [B, C, H, W]，即图像数据
# r: 滤波半径（整数） → 滑动窗口大小为 (2*r+1)
# # “滑动窗口的和差值” 是一种优化滑动窗口求和操作的技术，常用于加速如 BoxFilter 或 Guided Filter
def diff_x(input, r):
    assert input.dim() == 4 #确保输入是 [B, C, H, W] 四维张量

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

# 沿 y 方向（高度方向）计算差分
# input: 4D Tensor  [B, C, H, W]，即图像数据
# r滤波半径（整数） → 滑动窗口大小为 (2*r+1)
# “滑动窗口的和差值” 是一种优化滑动窗口求和操作的技术，常用于加速如 BoxFilter 或 Guided Filter
def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output

#  基于 积分图+差分 实现滑动窗口求和（不是平均）
class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    # x 是输入图像（张量）
    def forward(self, x):
        assert x.dim() == 4 #确认 x 是一个 4D 张量
        # 获得 每个滑动窗口的总和
        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class EdgeAttention_GuidedFilter(nn.Module):
    def __init__(self, r=2, eps=1e-8, hr_c=3, lr_c=128, common_c=128):
        super().__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

        self.align_conv_hr = nn.Conv2d(hr_c, common_c, kernel_size=1)
        self.align_conv_lr = nn.Conv2d(lr_c, common_c, kernel_size=1)
        self.out_conv = nn.Conv2d(lr_c + lr_c, lr_c, kernel_size=1)

    # lr_y: 低分辨率输入（通常是 p），如粗糙预测
    # hr_x: 高分辨率引导图（通常是 I）
    # l_a: 注意力图（来自 cross attention 模块）
    def forward(self, lr_y, hr_x, l_a):
        # 获取尺寸信息
        # n_lrx：batch size（样本数量）
        # c_lrx：通道数（channels）
        # h_lrx：高度（height）
        # w_lrx：宽度（width）

        hr_x = self.align_conv_hr(hr_x)
        # lr_y = self.align_conv_lr(lr_y)

        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        # 将输入数据转换为 double 精度（float64），提高数值稳定性
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        #  N是BoxFilter计算的滑动窗口中像素个数（求和）
        #  N的shaple和原图一样，每个像素点的值表示以该像素为中心、滑动窗口ω覆盖的实际像素数量
        # 如果窗口完整（完全落在图像内部），比如中心区域，那就是(2r+1)x(2r+1)
        # 如果窗口靠近边界，被截断了，比如左上角、边缘位置，那就小于(2r+1)x(2r+1)
        N = self.boxfilter(Variable(hr_x.data.new().resize_((1, 1, h_hrx, w_hrx)).fill_(1.0)))

        # 对注意力图进行归一化处理，使得所有元素加起来为1，相当于空间加权。形成注意力权重图l_t，每个像素attention越大，对滤波影响越大
        l_a = torch.abs(l_a) + self.epss #先取 l_a 的绝对值（防止有负数），然后加上一个很小的常数 epss防止除以 0。
        t_all = torch.sum(l_a) #求所有元素的和
        l_t = l_a / t_all #归一化处理，即每个元素除以总和，使得结果张量 l_t 的所有元素加起来为 1

        mean_a = self.boxfilter(l_a) / N #加权平均注意力 E[la]。对l_a做一个窗口均值计算。得到每个像素在其邻域内的 attention 平均值。
        mean_a2xy = self.boxfilter(l_a * l_a * hr_x * lr_y) / N   #加权引导×输入图。对E|IP|加权 即E|lala·I·P|
        mean_ay = self.boxfilter(l_a * lr_y) / N  #加权平均输入图，对E|P|加权 即E|la·P|
        mean_a2x2 = self.boxfilter(l_a * l_a * hr_x * hr_x) / N  #加权平方引导图，对E|II|加权 即E|lala·II|
        mean_ax = self.boxfilter(l_a * hr_x) / N  #加权平均引导图，对E|I|加权 即E|la·I|
        mean_tax = self.boxfilter(l_t * l_a * hr_x) / N  # 归一注意力再乘引导，即E[lt·la·I]。用于拟合权重归一化的调整项

        # 计算线性系数 A 和 b
        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean，再做一次平滑（BoxFilter），增强鲁棒性
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        # 上采样回高分辨率并输出
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        output = mean_A * hr_x + mean_b  # 此时 output 是 double（float64）

        fused = torch.cat([output.float(), lr_y.float()], dim=1)
        out = self.out_conv(fused)  # fused 已是 float32
        return out

        # return (mean_A*hr_x+mean_b).float()

class FastGuidedFilter_attention(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(FastGuidedFilter_attention, self).__init__()

        self.r = r # 滑动窗口半径 r
        self.eps = eps # 稳定项，防止除以0（1e-8）
        self.boxfilter = BoxFilter(r) # 用于局部求和操作（加速计算）
        self.epss = 1e-12 # 一个非常小的值，用于避免除以 0


    # lr_x 低分辨率引导图像（通常是 I）
    # lr_y: 低分辨率输入（通常是 p），如粗糙预测
    # hr_x: 高分辨率引导图
    # l_a: 注意力图（来自 cross attention 模块）
    def forward(self, lr_x, lr_y, hr_x, l_a):
        # 获取尺寸信息
        # n_lrx：batch size（样本数量）
        # c_lrx：通道数（channels）
        # h_lrx：高度（height）
        # w_lrx：宽度（width）
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        # 将输入数据转换为 double 精度（float64），提高数值稳定性
        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        assert n_lrx == n_lry and n_lry == n_hrx #所有输入 batch size 一致
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry) #通道匹配
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1 #尺寸大于滑动窗口要求

        #  N是BoxFilter计算的滑动窗口中像素个数（求和）
        #  N的shaple和原图一样，每个像素点的值表示以该像素为中心、滑动窗口ω覆盖的实际像素数量
        # 如果窗口完整（完全落在图像内部），比如中心区域，那就是(2r+1)x(2r+1)
        # 如果窗口靠近边界，被截断了，比如左上角、边缘位置，那就小于(2r+1)x(2r+1)
        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        # 对注意力图进行归一化处理，使得所有元素加起来为1，相当于空间加权。形成注意力权重图l_t，每个像素attention越大，对滤波影响越大
        l_a = torch.abs(l_a) + self.epss #先取 l_a 的绝对值（防止有负数），然后加上一个很小的常数 epss防止除以 0。
        t_all = torch.sum(l_a) #求所有元素的和
        l_t = l_a / t_all #归一化处理，即每个元素除以总和，使得结果张量 l_t 的所有元素加起来为 1

        mean_a = self.boxfilter(l_a) / N #加权平均注意力 E[la]。对l_a做一个窗口均值计算。得到每个像素在其邻域内的 attention 平均值。
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N   #加权引导×输入图。对E|IP|加权 即E|lala·I·P|
        mean_ay = self.boxfilter(l_a * lr_y) / N  #加权平均输入图，对E|P|加权 即E|la·P|
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N  #加权平方引导图，对E|II|加权 即E|lala·II|
        mean_ax = self.boxfilter(l_a * lr_x) / N  #加权平均引导图，对E|I|加权 即E|la·I|
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N  # 归一注意力再乘引导，即E[lt·la·I]。用于拟合权重归一化的调整项

        # 计算线性系数 A 和 b
        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        b = (mean_ay - A * mean_ax) / (mean_a)

        # --------------------------------
        # Mean，再做一次平滑（BoxFilter），增强鲁棒性
        # --------------------------------
        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        # 上采样回高分辨率并输出
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)

        return (mean_A*hr_x+mean_b).float()

# 构建一个残差卷积模块
class ResidualConv(nn.Module):
    # input_dim：输入特征图的通道数
    # output_dim：输出特征图的通道数
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        # 主分支conv_block
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )

        # 跳跃分支：conv_skip，用于将输入 x 映射为与 conv_block 输出相同尺寸的特征图
        # 由于主分支可能改变特征图的大小（stride ≠ 1），这里用 Conv2d+BN 使跳跃路径与主路径保持 通道数和空间尺寸一致，方便相加
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)

# ConvNext 模块，这是一个“残差卷积模块”，受 Transformer 启发但基于卷积构建
class ConvNext(nn.Module):
    # dim: 输入/输出通道数（保持不变）
    # layer_scale_init_value: 可学习的缩放因子初始值（类似 Transformer 中的 LayerScale）
    # kernel_size: 用于 depthwise 卷积的核大小（一般为 7）
    def __init__(self, dim, layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__()
        # pad_size = kernel_size // 2 是一种通用技巧，确保stride=1 的情况下，输出大小与输入一致
        pad_size = kernel_size // 2
        # 深度卷积（Depthwise Convolution），groups=dim 表示“深度可分离卷积”，对每个通道独立做 7×7 卷积
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad_size, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm2d(dim)
        # 用 1×1 卷积将通道数从 dim 提升到 dim × 4，增加表示能力
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU()
        # 用 1×1 卷积把通道数降回原始维度 dim
        self.conv2 = nn.Conv2d(dim * 4, dim, kernel_size=1, stride=1, padding=0)
        # gamma 是一个每个通道一个的可学习参数，初始值是 1e-6，训练时自动学出合适的比例（防止初期扰动太大）
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_block = DropBlock(7, 0.5)

    def forward(self, x):
        _input = x #保留残差连接
        x = self.dwconv(x)  # 深度卷积（提取空间特征）
        x = self.norm(x)  # BatchNorm
        x = self.conv1(x)  # 升维
        x = self.act(x)  # 激活
        x = self.conv2(x)  # 降维
        # self.gamma 是一个 shape 为 [C] 的 可学习参数向量，即每个通道一个缩放因子。
        # 经过两次unsqueeze(-1)将self.gamma 的shape改为[C, 1, 1]
        x = self.gamma.unsqueeze(-1).unsqueeze(-1) * x
        x = _input + self.drop_block(x)  #残差连接：主路径的输出 + 输入

        return x

# 自定义的层归一化模块，支持两种数据格式：channels_last (B, H, W, C) 和 channels_first (B, C, H, W)
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # 创建可学习的缩放因子 weight 和偏置 bias，初始为1和0，表示不缩放、不平移
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps #防止除零的极小值
        self.data_format = data_format #输入张量格式，是 (B, C, H, W) 还是 (B, H, W, C)
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # 直接使用 PyTorch 内置 F.layer_norm 处理 (B, H, W, C) 格式
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # 自定义实现对 (B, C, H, W) 格式进行 LayerNorm
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# 定义一个上采样模块，用的是转置卷积（ConvTranspose2d）
class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    # input_encoder: 编码器特征图的通道数
    # input_decoder: 解码器特征图的通道数
    # output_dim: 注意力通道数（中间统一的特征维度）
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        # 编码器分支处理，BN + ReLU → 3×3 Conv → 下采样（尺寸减半）
        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        # 解码器分支处理，BN + ReLU + Conv（通道统一为 output_dim）
        # 注意这里可能写错了：BatchNorm2d(input_encoder) 应该是 input_decoder，否则与输入通道不匹配
        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        # 通过 1×1 Conv 压缩成单通道注意力图（权重）
        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        # 将两个分支分别处理后相加（融合），通道统一为 output_dim，空间尺寸相同
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        # 再经过注意力卷积层，生成注意力图 out（shape: [B,1,H,W]）
        out = self.conv_attn(out)
        return out * x2 #注意力图与 decoder 分支 x2 相乘，实现通道注意力增强

# 完整的 FSGNet 网络结构，实现了Encoder-Decoder + Attention + Fast Guided Filter + Cross Attention 设计。
class FSGNet(nn.Module):
    # depths: 每个阶段 ConvNeXt 块的数量，列表类型
    def __init__(self, channel, n_classes, base_c, depths, kernel_size):
        super(FSGNet, self).__init__()

        # 通过 M_Conv + depths[0]层 ConvNext，从 64通道提取为 base_c * 1通道；
        self.input_layer = nn.Sequential(
            M_Conv(64, base_c * 1, kernel_size=kernel_size),
            *[ConvNext(base_c * 1, kernel_size=kernel_size) for _ in range(depths[0])]
        )
        # 通过一个 M_Conv得到base_c * 1通道
        self.input_skip = nn.Sequential(
            M_Conv(64, base_c * 1, kernel_size=kernel_size),
        )
        # 原始输入 x（[B, 3, H, W]）通过ACS变成64通道，在通过 M_Conv变成 base_c * 1 通道的特征图
        self.conv1 = M_Conv(64, base_c * 1, kernel_size=3)

        # 分辨率减半、保持通道不变。先用 2×2 stride=2 卷积下采样，再用多个 ConvNext 模块提取中高层语义特征
        self.down_conv_2 = nn.Sequential(*[
            # 二维卷积层，保持通道数不变，但将特征图尺寸缩小一半 的卷积操作（常用于下采样）
            nn.Conv2d(base_c * 2, base_c * 2, kernel_size=2, stride=2),
            # 接下来通过 depths[1] 个 ConvNext 块对下采样后的特征图进行深度提取与特征变换；
            # 每个 ConvNext(base_c * 2, ...) 的输入输出通道数都是 base_c * 2，保持通道不变；
            # 起到增强语义特征提取能力的作用
            *[ConvNext(base_c * 2, kernel_size=kernel_size) for _ in range(depths[1])]
            ])
        # 原始输入 x（[B, 3, H/2, W/2]）通过ACS变成64通道，在通过 M_Conv 变成 base_c * 2 通道的特征图
        self.conv2 = M_Conv(64, base_c * 2, kernel_size=3)

        self.down_conv_3 = nn.Sequential(*[
            nn.Conv2d(base_c * 4, base_c * 4, kernel_size=2, stride=2),
            *[ConvNext(base_c * 4, kernel_size=kernel_size) for _ in range(depths[2])]
            ])
        # 原始输入 x（[B, 3, h/4, W/4]）通过ACS变成64通道，在通过 M_Conv 变成 base_c * 4 通道的特征图
        self.conv3 = M_Conv(64, base_c * 4, kernel_size=3)

        self.down_conv_4 = nn.Sequential(*[
            nn.Conv2d(base_c * 8, base_c * 8, kernel_size=2, stride=2),
            *[ConvNext(base_c * 8, kernel_size=kernel_size) for _ in range(depths[3])]
            ])
        self.attn = SelfAttentionBlock()

        # 对上采样后的特征图进行通道压缩；
        # 同时用残差结构增强特征表达能力；
        # 保持空间尺寸不变（stride=1）以便后续与浅层特征融合。
        self.up_residual_conv3 = ResidualConv(base_c * 8, base_c * 4, 1, 1)
        self.up_residual_conv2 = ResidualConv(base_c * 4, base_c * 2, 1, 1)
        self.up_residual_conv1 = ResidualConv(base_c * 2, base_c * 1, 1, 1)

        #  1×1 的2D卷积层，它的作用是将输入的特征图从 base_c * 4 个通道 压缩到最终类别数 n_classes 个通道。
        # [batch_size, base_c * 4, H, W] 转化为 [batch_size, n_classes, H, W]，变成一张分割概率图
        # Sigmoid() 的作用是把值压到 [0, 1]，表示像素属于前景的概率
        self.output_layer3 = nn.Sequential(
            nn.Conv2d(base_c * 4, n_classes, 1, 1),
            nn.Sigmoid(),
        )
        self.output_layer2 = nn.Sequential(
            nn.Conv2d(base_c * 2, n_classes, 1, 1),
            nn.Sigmoid(),
        )
        self.output_layer1 = nn.Sequential(
            nn.Conv2d(base_c * 1, n_classes, 1, 1),
            nn.Sigmoid(),
        )

        self.eagf3 = EdgeAttention_GuidedFilter(r=2, eps=1e-2, hr_c=3, lr_c=base_c * 8, common_c=base_c * 8)
        self.eagf2 = EdgeAttention_GuidedFilter(r=2, eps=1e-2, hr_c=3, lr_c=base_c * 4, common_c=base_c * 4)
        self.eagf1 = EdgeAttention_GuidedFilter(r=2, eps=1e-2, hr_c=3, lr_c=base_c * 2, common_c=base_c * 2)


        self.tpa_attention3 = TriplePoolingAttention(in_c1=3, in_c2=base_c * 8, mid_c=base_c)
        self.tpa_attention2 = TriplePoolingAttention(in_c1=3, in_c2=base_c * 4, mid_c=base_c)
        self.tpa_attention1 = TriplePoolingAttention(in_c1=3, in_c2=base_c * 2, mid_c=base_c)

        self.fgf = FastGuidedFilter_attention(r=2, eps=1e-2)
        self.attention_block3 = CrossAttentionBlock(in_channels=base_c * 8)
        self.attention_block2 = CrossAttentionBlock(in_channels=base_c * 4)
        self.attention_block1 = CrossAttentionBlock(in_channels=base_c * 2)

        # 输入通道数：base_c * 8 + base_c * 8 = base_c * 16，表示有两个 feature map 按照通道维度拼接（cat） 后送入。输出通道数：base_c * 8
        # 使用 1×1 卷积将通道数降维为一半。
        # kernel_size = 1：代表只在通道维度做变换，不影响空间尺寸（常用于压缩通道或调整维度）
        self.conv_cat_3 = M_Conv(base_c * 8 + base_c * 8, base_c * 8, kernel_size=1)
        self.conv_cat_2 = M_Conv(base_c * 8 + base_c * 4, base_c * 4, kernel_size=1)
        self.conv_cat_1 = M_Conv(base_c * 4 + base_c * 2, base_c * 2, kernel_size=1)

        self.awl = AdaptiveWeightLearning(num_outputs=3)

        self.dmu1 = DynamicMergeUnit(in_channels=1)
        self.dmu2 = DynamicMergeUnit(in_channels=1)
        self.dmu3 = DynamicMergeUnit(in_channels=1)

        self.acs1 = AdaptiveConvSelectionBlock(in_channels=3, mid_channels=61, out_channels=64)
        self.acs2 = AdaptiveConvSelectionBlock(in_channels=3, mid_channels=61, out_channels=64)
        self.acs3 = AdaptiveConvSelectionBlock(in_channels=3, mid_channels=61, out_channels=64)

    def forward(self, x):
        self.feature_vis = {}  
        self.feature_vis['input'] = x
        # Get multi-scale from input
        _, _, h, w = x.size()

        # 构建两个尺寸为 1/2 和 1/4 的原始图像副本，用于后续多尺度融合
        input_scale_1=x
        x_scale_2 = F.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        input_scale_2 = x_scale_2
        x_scale_3 = F.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)
        input_scale_3 = x_scale_3

        # 第一层编码器（原始+浅层）
        x=self.acs1(x)
        x1 = self.input_layer(x) + self.input_skip(x)  #两者逐元素相加，作为融合特征（类似残差连接）
        x1_conv = self.conv1(x)
        x1_down = torch.cat([x1_conv, x1], dim=1)  #把 x1_conv（来自原始输入）和 x1（深层残差特征）在通道维度拼接。
        eagf_out1 = self.eagf1(x1_down,input_scale_1, self.tpa_attention1(input_scale_1, x1_down))
        self.feature_vis['x1_down'] = x1_down

        # 第二层编码器
        x_scale_2 = self.acs2(x_scale_2)
        x2 = self.down_conv_2(x1_down)
        x2_conv = self.conv2(x_scale_2)
        x2_down = torch.cat([x2_conv, x2], dim=1)
        eagf_out2 = self.eagf2(x2_down,input_scale_2, self.tpa_attention2(input_scale_2, x2_down))
        self.feature_vis['x2_down'] = x2_down

        # 第三层编码器
        x_scale_3 = self.acs3(x_scale_3)
        x3 = self.down_conv_3(x2_down)
        x3_conv = self.conv3(x_scale_3)
        x3_down = torch.cat([x3_conv, x3], dim=1)
        eagf_out3 = self.eagf3(x3_down,input_scale_3, self.tpa_attention3(input_scale_3, x3_down))
        self.feature_vis['x3_down'] = x3_down

        # 编码器最后一层 + 注意力机制
        x4 = self.down_conv_4(x3_down)
        x4 = self.attn(x4)
        self.feature_vis['x4'] = x4

        # Decoder
        _, _, h, w = x3_down.size()
        x3_gf = torch.cat([eagf_out3, F.interpolate(x4, size=(h, w), mode='bilinear', align_corners=True)], dim=1) #对x4进行interpolate上采样，然后和x3_down进行拼接
        x3_gf_conv = self.conv_cat_3(x3_gf) #用 conv_cat_3（一个 M_Conv 1×1 卷积）将通道从 base_c * 8 + base_c * 8 压缩回 base_c * 8。
        # x3_small = F.interpolate(x3_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True) #对x3_gf_conv进行interpolate下采样得到x3_small，H、W减半，通道不变
        # fgf_out = self.fgf(x3_small, x4, x3_gf_conv, self.attention_block3(x3_small, x4))
        # x3_up = self.up_residual_conv3(fgf_out)
        x3_up = self.up_residual_conv3(x3_gf_conv)
        self.feature_vis['x3_up'] = x3_up

        _, _, h, w = x2_down.size()
        x2_gf = torch.cat([eagf_out2, F.interpolate(x3_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x2_gf_conv = self.conv_cat_2(x2_gf)
        # x2_small = F.interpolate(x2_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        # fgf_out = self.fgf(x2_small, x3_up, x2_gf_conv, self.attention_block2(x2_small, x3_up))
        # x2_up = self.up_residual_conv2(fgf_out)
        x2_up = self.up_residual_conv2(x2_gf_conv)
        self.feature_vis['x2_up'] = x2_up

        _, _, h, w = x1_down.size()
        x1_gf = torch.cat([eagf_out1, F.interpolate(x2_gf_conv, size=(h, w), mode='bilinear', align_corners=True)], dim=1)
        x1_gf_conv = self.conv_cat_1(x1_gf)
        # x1_small = F.interpolate(x1_gf_conv, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        # fgf_out = self.fgf(x1_small, x2_up, x1_gf_conv, self.attention_block1(x1_small, x2_up))
        # x1_up = self.up_residual_conv1(fgf_out)
        x1_up = self.up_residual_conv1(x1_gf_conv)
        self.feature_vis['x1_up'] = x1_up

        _, _, h, w = x.size() #取出原始输入图像的高宽
        out_3 = F.interpolate(x3_up, size=(h, w), mode='bilinear', align_corners=True) #将x3_up上采样到和输入图像一样大
        out_2 = F.interpolate(x2_up, size=(h, w), mode='bilinear', align_corners=True) #将x2_up上采样到和输入图像一样大

        out_3 = self.output_layer3(out_3)
        out_2 = self.output_layer2(out_2)
        out_1 = self.output_layer1(x1_up)
        self.feature_vis['output_1'] = out_1
        self.feature_vis['output_2'] = out_2
        self.feature_vis['output_3'] = out_3

        ##############下面是原来FSG-NET里的###########################
        # 返回的是三个分辨率一致、语义深度不同的预测图（多尺度输出）
        # return out_1, out_2, out_3
        ############################################################

        ##############下面是在FSG-NET基础上加了AWL模块###########################
        # 多尺度预测融合
        # out_fused, [alpha, beta, gamma] = self.awl([out_1, out_2, out_3])
        # self.feature_vis['out_fused'] = out_fused
        # self.feature_vis['awl_weights'] = [alpha, beta, gamma]#??????????????????????????什么样子的结果
        # return out_fused, out_1, out_2, out_3
        ####################################################################

        ###############下面是在FSG-NET基础上加了DMU模块##########################
        #  通过 DMU 获取初始 attention 权重图（未归一化）
        alpha = self.dmu1(out_1)  # shape: [B, 1, H, W]
        beta = self.dmu2(out_2)
        gamma = self.dmu3(out_3)

        # 进行逐像素 softmax 归一化（stack → softmax → split）
        # shape after stack: [B, 3, H, W]
        attention_stack = torch.cat([alpha, beta, gamma], dim=1)
        attention_softmax = torch.softmax(attention_stack, dim=1)
        alpha_norm, beta_norm, gamma_norm = torch.chunk(attention_softmax, chunks=3, dim=1)

        # 分别加权各分支输出
        f1_prime = out_1 * alpha_norm  # shape: [B, 1, H, W]
        f2_prime = out_2 * beta_norm
        f3_prime = out_3 * gamma_norm

        out_fused = f1_prime + f2_prime + f3_prime  # shape: (B,1,H,W)

        self.feature_vis['f1_prime'] = f1_prime
        self.feature_vis['f2_prime'] = f2_prime
        self.feature_vis['f3_prime'] = f3_prime
        self.feature_vis['alpha'] = alpha_norm
        self.feature_vis['beta'] = beta_norm
        self.feature_vis['gamma'] = gamma_norm
        self.feature_vis['out_fused'] = out_fused

        return out_fused, out_1, out_2, out_3
        ####################################################################

