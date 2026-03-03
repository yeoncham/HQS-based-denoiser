import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import fourier_data_step
import torch
import numpy as np

# --- NAFNet Components ---
class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=(0, 2, 3)), grad_output.sum(dim=(0, 2, 3)), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, 1, 0, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, 1, 0, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma

class NAFNet(nn.Module):
    def __init__(self, img_channel=4, width=32, middle_blk_num=12,
                 enc_blk_nums=[2,2,4,8], dec_blk_nums=[2,2,2,2]):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, 1, 1, bias=True)
        self.ending = nn.Conv2d(width, 3, 3, 1, 1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2*chan, 2, 2))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan*2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp_pad = self.check_image_size(inp)
        x = self.intro(inp_pad)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp_pad[:, :3, :, :]
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))


class HQS_PnP:
    def __init__(self, nafnet_model, iterations=10, rho_range=(-3, 1)):
        self.nafnet = nafnet_model
        self.iterations = iterations
        self.rho_min = rho_range[0]
        self.rho_max = rho_range[1]

    def _get_rho_schedule(self):
        return np.exp(np.linspace(self.rho_min, self.rho_max, self.iterations))

    def _kernel_fft(self, kernel, target_h, target_w):
        device = kernel.device
        kh, kw = kernel.shape
        
        # 타겟 사이즈만큼 제로 패딩
        padded = torch.zeros((target_h, target_w), device=device)
        padded[:kh, :kw] = kernel 
        
        # 커널 중심을 (0,0)으로 이동
        padded = torch.roll(padded, shifts=(-(kh // 2), -(kw // 2)), dims=(0, 1))
        
        fft_h = torch.fft.fft2(padded)
        return fft_h.unsqueeze(0).unsqueeze(0)

    def _fft_deblur(self, y, H_fft, z, rho):
        # y, z가 이미 Reflect Padding 된 상태
        Y = torch.fft.fft2(y) 
        Z = torch.fft.fft2(z)
        
        x_hat = (torch.conj(H_fft) * Y + rho * Z) / (torch.abs(H_fft) ** 2 + rho)
        
        x_large = torch.fft.ifft2(x_hat).real
        
        # Loop 도는 동안은 패딩된 크기 유지
        return x_large 

    def solve(self, y, blur_kernel, sigma_map):
        rhos = self._get_rho_schedule()
        B, C, H, W = y.shape
        kh, kw = blur_kernel.shape
        device = y.device
        
        # 커널 크기만큼 패딩
        pad_h, pad_w = kh, kw
        
        # y와 sigma_map을 모두 Reflect Padding, kernel의 2배 크기만큼
        y_pad = F.pad(y, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        sigma_pad = F.pad(sigma_map, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        
        # 패딩된 후의 새로운 타겟 사이즈
        target_h, target_w = y_pad.shape[2], y_pad.shape[3]
        
        # 커널 FFT
        H_fft = self._kernel_fft(blur_kernel, target_h, target_w)
                
        z = y_pad.clone() # z도 패딩된 크기로 시작
        
        for k in range(self.iterations):
            current_rho = torch.tensor(rhos[k], device=device, dtype=torch.float32)
            
            # 1. Data fidelity step (x-step)
            x = self._fft_deblur(y_pad, H_fft, z, current_rho)
            
            # 2. Prior step (z-step)
            with torch.no_grad():
                x_cat = torch.cat([x, sigma_pad], dim=1)
                z = self.nafnet(x_cat)
                z = torch.clamp(z, 0, 1)
        
        # 최종 결과 반환 직전에 Crop (원래 크기로 복구)
        return z[:, :, pad_h:-pad_h, pad_w:-pad_w]