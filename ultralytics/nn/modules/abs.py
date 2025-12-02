class LightECEM(nn.Module):
    """轻量级裂缝增强注意力模块 - 仅保留核心注意力机制"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        
        # 1. 高效通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 8), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 8), channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 2. 轻量空间注意力（针对裂缝细长特性）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # 3. 轻量多尺度上下文（简化版）
        self.dw_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

    def forward(self, x):
        residual = x
        
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        
        # 多尺度上下文增强
        context = self.dw_conv(x)
        x = torch.max(x, context)  # 增强细长特征
        
        return x + residual  # 残差连接

class LightBottleneckCE(Bottleneck):
    """轻量级裂缝增强Bottleneck"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)
        self.light_ecem = LightECEM(c_)

    def forward(self, x):
        y = self.cv1(x)
        y = self.light_ecem(y)  # 轻量增强
        y = self.cv2(y)
        return x + y if self.add else y
    
class C3k2_LightCE(C3k2):
    """轻量版C3k2CE"""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        
        # 使用轻量Bottleneck
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 1, shortcut, g) if c3k else LightBottleneckCE(self.c, self.c, shortcut, g)
            for _ in range(max(1, n//2))
        )