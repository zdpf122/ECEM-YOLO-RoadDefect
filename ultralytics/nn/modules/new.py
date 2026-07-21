import torch
import torch.nn as nn
import torchvision
from .block import PSABlock, C2PSA  # 保持与原有模块的兼容性

# 1. 轻量化可变形卷积 - 减少偏移网络复杂度
class LightDeformConv(nn.Module):
    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1):
        super(LightDeformConv, self).__init__()
        self.kernel_size = kernel_size
        
        # 轻量化偏移网络：通过1x1卷积降维后再计算偏移量
        self.offset_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1, bias=True),  # 先降维
            nn.GELU(),
            nn.Conv2d(
                in_channels=in_channels // 2,
                out_channels=2 * kernel_size[0] * kernel_size[1],
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                bias=True
            )
        )
        
        # 保持可变形卷积核心功能
        self.deform_conv = torchvision.ops.DeformConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            stride=stride,
            dilation=dilation,
            bias=False
        )

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


# 2. 优化LKA模块 - 减少冗余计算
class LightDeformableLKA(nn.Module):
    def __init__(self, dim, reduction=2):
        super().__init__()
        # 动态感受野调整：根据输入维度自适应空洞率
        self.dilation = 2 if dim < 128 else 3
        
        # 使用轻量化可变形卷积
        self.conv_dynamic = LightDeformConv(
            dim, 
            kernel_size=(3,3), 
            padding=self.dilation,  # 动态适配空洞率
            groups=dim, 
            dilation=self.dilation
        )
        # 1. 边缘检测：用固定Sobel核提取裂缝边缘（无训练参数，计算量极小）
        self.sobel_x = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.sobel_y = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        # 初始化Sobel核（固定值，不参与训练）
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1,1,3,3)
        self.sobel_x.weight.data = sobel_x_kernel.repeat(dim, 1, 1, 1)
        self.sobel_y.weight.data = sobel_y_kernel.repeat(dim, 1, 1, 1)
        self.sobel_x.weight.requires_grad = False  # 固定核，不训练
        self.sobel_y.weight.requires_grad = False
        
        # 2. 动态掩码生成：基于边缘强度过滤背景（仅1个1x1卷积，参数极少）
        self.bg_mask = nn.Sequential(
            nn.Conv2d(dim, 1, 1, bias=False),  # 边缘特征压缩到单通道
            nn.BatchNorm2d(1),
            nn.Sigmoid()  # 生成0~1掩码（1=裂缝区域，0=背景区域）
        )
        # 简化能量检测器：减少通道转换
        self.energy_detector = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=3, padding=1, groups=dim//reduction, bias=False),
            nn.BatchNorm2d(dim // reduction),
            nn.GELU(),
            nn.Conv2d(dim // reduction, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # 保持通道整合功能但减少参数
        self.conv1 = nn.Conv2d(dim, dim, 1, groups=dim//32)  # 分组卷积进一步轻量化

    def forward(self, x):
        # 计算边缘强度（x+y方向）
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        edge_strength = torch.sqrt(edge_x**2 + edge_y**2)  # 边缘强度图
        # 生成背景掩码（边缘密集区域掩码值接近1，背景接近0）
        mask = self.bg_mask(edge_strength)
        # 用掩码过滤输入x：抑制背景区域特征
        x_filtered = x * mask  # 仅保留裂缝候选区域的特征
        # ------------------------------------------------------------------------------

        # 原有小目标增强逻辑（基于过滤后的x_filtered计算）
        energy = self.energy_detector(x_filtered)
        attn = self.conv_dynamic(x_filtered)  # 可变形卷积仅处理过滤后的特征
        attn = torch.where(energy < 0.4, attn * 1.1, attn)
        attn = self.conv1(attn)
        
        # 再次用背景掩码过滤注意力结果，彻底排除背景干扰
        attn = attn * mask
        return x * attn


# 3. 轻量化注意力模块 - 简化投影层
class LightDeformableLKA_Attention(nn.Module):
    def __init__(self, d_model, reduction=2):
        super().__init__()
        # 合并投影层和激活函数，减少层数量
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model // reduction, 1),  # 先降维
            nn.GELU(),
            nn.Conv2d(d_model // reduction, d_model, 1)   # 再升维
        )
        
        self.spatial_gating_unit = LightDeformableLKA(d_model)
        self.white_line_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化：计算每个通道的亮度均值（B, C, 1, 1）
            nn.Conv2d(d_model, 1, 1, bias=False),  # 压缩到单通道亮度评分
            nn.Sigmoid()  # 亮度评分：0=普通路面，1=高亮度（白线）
        )
        # 白线区域裂缝增强权重（可训练，但仅1个参数）
        self.white_line_gain = nn.Parameter(torch.tensor(1.2), requires_grad=True)
    def forward(self, x):
        shortcut = x
        x = self.proj(x)
        
        # -------------------------- 白线区域增强 --------------------------
        # 1. 识别白线区域：计算亮度评分（越高越可能是白线）
        white_line_score = self.white_line_detector(x)  # (B, 1, H, W)
        # 2. 生成增强权重：白线区域用1.2倍权重，普通区域用1.0倍
        enhance_weight = 1.0 + (self.white_line_gain - 1.0) * white_line_score
        # 3. 增强白线区域特征（让模型更关注白线上的裂缝）
        x = x * enhance_weight
        # ------------------------------------------------------------------------------

        # 原有空间门控（含背景抑制）
        x = self.spatial_gating_unit(x)
        return x + shortcut


# 4. 优化PSABlock - 减少FFN复杂度
class LightPSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=None, shortcut=True) -> None:
        super().__init__()
        self.num_heads = num_heads if num_heads else max(1, c // 128)  
        self.attn = LightDeformableLKA_Attention(c)
        
        # 关键修改：用 c*3//2 替代 c*1.5，确保通道数为整数
        self.ffn = nn.Sequential(
            Conv(c, c * 3 // 2, 1),  # 3//2 = 1.5，且结果必为整数（如c=31→31*3//2=46，c=32→48）
            Conv(c * 3 // 2, c, 1, act=False)
        )
        
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


# 5. 优化C2PSA_LKA - 调整通道比例和堆叠数量
class C2PSA_LKA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e * 0.8)  # 原有通道压缩
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        # 调用优化后的LightPSABlock（含背景抑制+白线增强）
        self.m = nn.Sequential(*(
            LightPSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) 
            for _ in range(max(1, n if c1 >= 128 else n // 2))
        ))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)  # 处理后的b已过滤背景+增强白线裂缝
        return self.cv2(torch.cat((a, b), 1))


# 保持原有的Conv和autopad定义，但稍作优化
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """优化的轻量级卷积层"""
    default_act = nn.SiLU()  # 保持与YOLO系列兼容的激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # 当输入输出通道相差较大时，使用分组卷积
        if g == 1 and c1 > c2 and c1 % c2 == 0:
            g = c2  # 自适应分组数
            
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
 
    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Bottleneck_DLKA(nn.Module):
    """Standard bottleneck."""
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = deformable_LKA_Attention(c_)
        self.add = shortcut and c1 == c2
 
    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
 
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
 
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
 
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
 
    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
 
class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
 
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))
 
    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
 
 
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""
 
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_DLKA(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2_DLKA(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
 
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in
            range(n)
        )

 
