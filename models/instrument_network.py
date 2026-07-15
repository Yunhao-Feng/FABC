import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class conv_bn_relu(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, padding_size):
        super(conv_bn_relu, self).__init__()
        # 强制转换，彻底避免 numpy.int64 / bool 与 torch 维度不兼容问题
        in_ch        = int(in_ch)
        out_ch       = int(out_ch)
        k_size       = int(k_size)
        padding_size = int(padding_size)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=padding_size),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class deconv_bn_relu(nn.Module):
    def __init__(self, in_ch, out_ch, last=False):
        super(deconv_bn_relu, self).__init__()
        in_ch  = int(in_ch)
        out_ch = int(out_ch)
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

    def forward(self, x):
        return self.deconv(x)

class CausalIV(nn.Module):
    """
    Instrument (Z) 自编码器。支持 ch 为 bool（旧逻辑：wide 与否）或 int（显式通道数）。
    当 ch 为 int 时，通道表按 base=ch 自适配：[base, base, 2*base, 2*base, base, base]
    """
    def __init__(self, ch=False):
        super(CausalIV, self).__init__()

        # 注意：先判断 bool，再判断 int（因为 bool 是 int 的子类）
        if isinstance(ch, bool):
            base = 640 if ch else 512
        elif type(ch) is int:
            base = ch
        else:
            raise TypeError("ch must be bool or int")

        base      = int(base)
        self.nc   = int(base)
        self.k_list = 3
        # 用 base 自动推导、而不是写死 640/512
        self.c_list = [base, base, base * 2, base * 2, base, base]
        self.c_list = [int(c) for c in self.c_list]

        self.encoder = nn.Sequential(
            conv_bn_relu(in_ch=self.nc,            out_ch=self.c_list[0], k_size=self.k_list, padding_size=1),
            conv_bn_relu(in_ch=self.c_list[0],     out_ch=self.c_list[1], k_size=self.k_list, padding_size=1),
            conv_bn_relu(in_ch=self.c_list[1],     out_ch=self.c_list[2], k_size=self.k_list, padding_size=1),
        )
        self.linear = nn.Sequential(
            nn.Linear(int(self.c_list[2]), int(self.c_list[3])),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            deconv_bn_relu(in_ch=self.c_list[3], out_ch=self.c_list[4]),
            deconv_bn_relu(in_ch=self.c_list[4], out_ch=self.c_list[5]),
            deconv_bn_relu(in_ch=self.c_list[5], out_ch=self.nc, last=True),
        )
        self.weight_init()

    def weight_init(self):
        # 原地初始化：覆盖 Conv2d / ConvTranspose2d / Linear / BN
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        x = self.encoder(x)
        _, _, h, w = x.shape
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = x.view(-1, int(self.c_list[2]))
        x = self.linear(x)
        x = x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, h, w)
        x = self.decoder(x)
        return x

def kaiming_init(m):
    # 覆盖 Conv2d / ConvTranspose2d / Linear，用 kaiming_normal_ 原地初始化
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()

def exogenous(dataset, ch=False):
    return CausalIV(ch)
