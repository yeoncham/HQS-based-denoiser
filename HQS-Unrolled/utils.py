import torch
import torch.fft
import math

def get_psnr(pred, target, max_pixel=1.0):
    mse = torch.mean((pred - target)**2)
    if mse == 0: 
        return 100
    return 20 * math.log10(max_pixel / math.sqrt(mse))
    
def fourier_data_step(y, h, z, rho):
    _, _, H_img, W_img = y.shape
    _, _, kh, kw = h.shape

    if kh != H_img or kw != W_img:
        pad_h = torch.zeros((y.shape[0], 1, H_img, W_img), device=y.device)
        pad_h[:, :, :kh, :kw] = h
        h = pad_h 
        
        h = torch.roll(h, shifts=(-(kh//2), -(kw//2)), dims=(-2, -1))

    Y = torch.fft.fft2(y)
    H = torch.fft.fft2(h) 
    Z = torch.fft.fft2(z)
    
    x_new = (torch.conj(H) * Y + rho * Z) / (torch.abs(H)**2 + rho)
    
    return torch.fft.ifft2(x_new).real