class EfficientCrackEnhance(nn.Module):
    """高效裂缝增强模块 - 专门针对细长低对比度目标."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels

        # 轻量多尺度上下文提取（替代多尺度池化）
        self.dw_conv3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw_conv5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)

        # 高效注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 8), channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # 轻量空间增强（针对细长特征）
        self.spatial_enhance = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())

        # 背景抑制
        self.bg_suppress = nn.Sequential(
            nn.Conv2d(channels, max(channels // 4, 8), 1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(max(channels // 4, 8), channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # 多尺度特征提取
        feat3 = self.dw_conv3(x)
        feat5 = self.dw_conv5(x)
        multi_scale = torch.max(feat3, feat5)  # 取最大值增强细长特征

        # 通道注意力
        ca = self.channel_attention(x)
        x_enhanced = x * ca

        # 空间增强（针对裂缝的细长特性）
        avg_out = torch.mean(x_enhanced, dim=1, keepdim=True)
        max_out, _ = torch.max(x_enhanced, dim=1, keepdim=True)
        spatial_weights = self.spatial_enhance(torch.cat([avg_out, max_out], dim=1))
        x_enhanced = x_enhanced * spatial_weights

        # 多尺度特征融合
        x_enhanced = x_enhanced + multi_scale

        # 背景抑制
        bg_weights = self.bg_suppress(x_enhanced)
        return x_enhanced * (1 - bg_weights)  # 抑制背景


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (Tuple[int, int]): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Apply bottleneck with optional shortcut connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
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
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class EfficientBottleneckCE(Bottleneck):
    """高效裂缝增强Bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        # 只在第一个卷积后添加轻量增强
        self.crack_enhance = EfficientCrackEnhance(c_)

    def forward(self, x):
        y = self.cv1(x)
        y = self.crack_enhance(y)  # 裂缝增强
        y = self.cv2(y)
        return x + y if self.add else y


class C3k2CE(C3k2):
    """最终高效版C3k2CE - 比原始C3k2更轻量且性能更好."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)

        # 使用高效Bottleneck替代标准Bottleneck
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 1, shortcut, g) if c3k else EfficientBottleneckCE(self.c, self.c, shortcut, g)
            for _ in range(max(1, n // 2))  # 减少层数但增强每层能力
        )
