import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class conv_bn_relu(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, padding_size):
        super(conv_bn_relu, self).__init__()
        in_ch       = int(in_ch)          # NEW: 强制转 int
        out_ch      = int(out_ch)         # NEW
        k_size      = int(k_size)         # NEW
        padding_size= int(padding_size)   # NEW
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=padding_size),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x): return self.conv(x)

class deconv_bn_relu(nn.Module):
    def __init__(self, in_ch, out_ch, last=False):
        super(deconv_bn_relu, self).__init__()
        in_ch  = int(in_ch)               # NEW
        out_ch = int(out_ch)              # NEW
        if last:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2)
            )
    def forward(self, x): return self.deconv(x)

class CausalAE(nn.Module):
    def __init__(self, ch=False):
        super(CausalAE, self).__init__()

        # NEW: 同时支持 bool（原来的 wide 开关）和 int（显式通道数）
        if isinstance(ch, bool):
            base = 640 if ch else 512
        elif type(ch) is int:             # 注意：排除 bool
            base = ch
        else:
            raise TypeError("ch must be bool or int")

        base = int(base)                  # NEW: 强制转 int
        self.nc = int(base)               # NEW
        self.k_list = 3                   # NEW: 确保是 int
        # NEW: 通道表基于 base 推导，避免硬编码 640/512
        self.c_list = [base, base, base*2, base*2, base, base]
        self.c_list = [int(c) for c in self.c_list]   # NEW

        self.encoder = nn.Sequential(
            conv_bn_relu(in_ch=self.nc,            out_ch=self.c_list[0], k_size=self.k_list, padding_size=1),
            conv_bn_relu(in_ch=self.c_list[0],     out_ch=self.c_list[1], k_size=self.k_list, padding_size=1),
            conv_bn_relu(in_ch=self.c_list[1],     out_ch=self.c_list[2], k_size=self.k_list, padding_size=1),
        )
        self.linear = nn.Sequential(
            nn.Linear(int(self.c_list[2]), int(self.c_list[3])),  # NEW: 显式 int
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            deconv_bn_relu(in_ch=self.c_list[3], out_ch=self.c_list[4]),
            deconv_bn_relu(in_ch=self.c_list[4], out_ch=self.c_list[5]),
            deconv_bn_relu(in_ch=self.c_list[5], out_ch=self.nc, last=True),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        x = self.encoder(x)
        _, _, h, w = x.shape
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(-1, int(self.c_list[2]))                # NEW
        x = self.linear(x)
        x = x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
        x = self.decoder(x)
        return x

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):  # NEW: 把转置卷积也覆盖
        init.kaiming_normal_(m.weight)         # NEW: 用带下划线版本
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1.)
        if m.bias is not None:
            m.bias.data.zero_()

def causal(dataset, ch=False):
    return CausalAE(ch)