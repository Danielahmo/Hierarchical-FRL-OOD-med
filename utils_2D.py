import torch.nn.functional as F
import torch
from torch.autograd import Variable
import cv2
import torchvision.transforms as transforms
import numpy as np

mse = torch.nn.MSELoss()


def gaussian_blur(x, k, stride=1, padding=0):
    res = []
    x = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0)
    for xx in x.split(1, 1):
        res.append(F.conv2d(xx, k, stride=stride, padding=0))
    return torch.cat(res, 1)


def get_gaussian_kernel(size=3):
    kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
    return kernel


def find_pixel_high_freq(im, gauss_kernel, find=False, index=None, is_gray=False):
    padding = (gauss_kernel.shape[-1] - 1) // 2
    if is_gray:
        im_gray = im[:, 0, ...]
    else:
        im_gray = im[:, 0, ...] * 0.299 + im[:, 1, ...] * 0.587 + im[:, 2, ...] * 0.114
    im_gray = im_gray.unsqueeze_(dim=1).repeat(1, 3, 1, 1)
    low_gray = gaussian_blur(im_gray, gauss_kernel, padding=padding)
    high_gray = im_gray - low_gray


    return low_gray[:, 0:1, :, :], high_gray[:, 0:1, :, :]


def process_x(x, args, is_gray = False):

    gauss_kernel = get_gaussian_kernel(args.gauss_size).cuda()
    return (find_pixel_high_freq(x, gauss_kernel, is_gray=is_gray)[1] + 1) / 2


def process_target(x, args=None):
    return Variable(x.data.view(-1) * 255).long()


def produce_concat_x(x, opt):
    x_H_org = process_x(x, opt)
    x = torch.cat((x, x_H_org), dim=1)
    return x
    
def process_2gaus(x, args, is_gray=False):
    gauss_kernel = get_gaussian_kernel(args.gauss_size).cuda()
    low_freq, high_freq = find_pixel_high_freq(x, gauss_kernel, is_gray=is_gray)

    # Optional: scale both to [0, 1] for visualization
    low_freq = (low_freq + 1) / 2
    high_freq = (high_freq + 1) / 2

    return low_freq, high_freq


def produce_concat_low_high(x, args):
    low_freq, high_freq = process_2gaus(x, args)
    return torch.cat((x, low_freq, high_freq), dim=1)


def hi_lo_fft(img_color):
    to_gray = transforms.Grayscale()
    img = to_gray(img_color).cpu()

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Create low-pass and high-pass masks
    rows, cols = img.shape
    crow, ccol = rows//2 , cols//2

    # Radius of the low-pass filter
    r = 5
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), r, 1, thickness=-1)

    # Low frequencies
    low_pass = fshift * mask
    low_freq = np.abs(np.fft.ifft2(np.fft.ifftshift(low_pass)))

    # High frequencies
    high_pass = fshift * (1 - mask)
    high_freq = np.abs(np.fft.ifft2(np.fft.ifftshift(high_pass)))

    cat= torch.cat((img_color, low_freq, high_freq), dim=1)

    return cat