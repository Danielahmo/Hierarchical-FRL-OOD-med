
"""
Train DCGAN-VAE with frequency regularization.

Key idea:
- Augment input with low/high frequency components
- Encourage model to use full frequency spectrum
- Evaluate robustness to frequency-based OOD

Author: Danielahmo

This code is based on: https://github.com/mu-cai/FRL

Date: 2026

Cite paper: Hierarchical frequency regularization for out-of-distribution detection in generative models for medical images, ISBI 2026

"""


import argparse
import random
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F


from torch import distributed as dist
import DCGAN_VAE_freq as DVAE

from utils_2D import process_x, process_target, process_2gaus
import torch.utils.data as data
from tqdm import tqdm
import os

import pandas as pd
from PIL import Image


def KL_div(mu, logvar, reduction="avg"):

    """
    Compute KL divergence between N(mu, sigma) and N(0,1).

    Args:
        mu (Tensor): Mean of latent distribution [B, Z]
        logvar (Tensor): Log-variance of latent distribution [B, Z]
        reduction (str): "sum" or "avg" (per-sample KL)

    Returns:
        Tensor: KL divergence (scalar if sum, else per-sample vector)
    """
    mu = mu.view(mu.size(0), mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    if reduction == "sum":
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        return KL

class MidogDataset(torch.utils.data.Dataset):

    """
    Custom dataset for MIDOG images.

    Expected input file format:
        A txt file with lines: <relative_path> <label>
    """

    def __init__(self, list_images, data_path, transforms=None):
        self.data_path = data_path
        self.list_images = pd.read_csv(list_images, sep=' ', header=None, names=['path', 'label'])
        self.transforms = transforms

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        im = Image.open(os.path.join(self.data_path,self.list_images['path'][idx])).convert('RGB')
        label = self.list_images['label'][idx]

        if self.transforms:
            im = self.transforms(im)

        return im, label

def main():
    parser = argparse.ArgumentParser()
   
    # -----------------------
    # Data parameters
    # -----------------------
    parser.add_argument("--workers", type=int, default=4, help="Data loading workers")
    parser.add_argument("--batchSize", type=int, default=32, help="Batch size")
    parser.add_argument("--imageSize", type=int, default=64, help="Input image size")
    parser.add_argument("--list_txt", type=str, help="Path to dataset list file")

    # -----------------------
    # Model parameters
    # -----------------------
    parser.add_argument("--nc", type=int, default=3, help="Input channels (RGB)")
    parser.add_argument("--nz", type=int, default=100, help="Latent dimension")
    parser.add_argument("--ngf", type=int, default=32, help="Feature maps")
    parser.add_argument("--gauss_size", type=int, default=5)

    # -----------------------
    # Training parameters
    # -----------------------
    parser.add_argument("--num_epoch", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=1, help="Beta-VAE weight")
    

    # -----------------------
    # System / logging
    # -----------------------
    parser.add_argument("--experiment", help="Output directory", type=str)
    parser.add_argument("--data_path", help="data directory", type=str)
    parser.add_argument("--cuda_num", type=str, default="1")
    parser.add_argument("--seed_val", type=int, default=-1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--model_save_num", type=int, default=100)
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument('--train_txt', type=str, help='Path to file list') 

    opt = parser.parse_args()

    cuda_num = opt.cuda_num
    if len(opt.cuda_num) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_num
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if opt.seed_val != -1:
        opt.manualSeed = opt.seed_val  # random.randint(1, 10000) # fix seed
    else:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    cudnn.benchmark = True

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((opt.imageSize, opt.imageSize)),
            transforms.ToTensor(),
        ]
    )
    train_txt = opt.train_txt
    dataset = MidogDataset( train_txt, opt.data_path, transforms=transform)


    opt.nc = opt.nc+ 2 #Add 2 channels for the number of frequency regularizatio channels (high and low frequency)

    if len(opt.cuda_num) == 1:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers),
        )

    if not os.path.exists(opt.experiment):
        os.mkdir(opt.experiment)

    print(f'Please see the path "{opt.experiment}" for the saved model !')

    #############################################################

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = int(opt.nc)
    print(f"Channel {nc}, ngf {ngf}, nz {nz}")
    beta = opt.beta

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = DVAE.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, 256)
    netG.apply(weights_init)

    netE = DVAE.Encoder(opt.imageSize, nz, nc, ngf, ngpu)
    netE.apply(weights_init)

    #muulti GP set-up
    if len(opt.cuda_num) > 1:
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        def synchronize():
            if not dist.is_available():
                return
            if not dist.is_initialized():
                return
            world_size = dist.get_world_size()
            if world_size == 1:
                return
            dist.barrier()

        synchronize()
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        netE = nn.parallel.DistributedDataParallel(
            netE,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )
        netG = nn.parallel.DistributedDataParallel(
            netG,
            device_ids=[0],
            output_device=0,
            broadcast_buffers=False,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers),
            sampler=data.distributed.DistributedSampler(dataset, shuffle=True),
            drop_last=True,
        )

    # setup optimizer

    optimizer1 = optim.Adam(netE.parameters(), lr=opt.lr, weight_decay=0)
    optimizer2 = optim.Adam(netG.parameters(), lr=opt.lr, weight_decay=0)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=30, gamma=0.5)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=30, gamma=0.5)

    netE.to(device)
    netG.to(device)

    start_epoch = 0


    print("start_epoch :", start_epoch)

    netE.train()
    netG.train()

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    rec_l = []
    kl = []
    history = []
    start = datetime.today()

    for epoch in tqdm(range(start_epoch, opt.num_epoch)):

        """
        Training loop:
        1. Encode input into latent space (z, mu, logvar)
        2. Decode to reconstruct input
        3. Compute:
            - Reconstruction loss (cross-entropy over pixels)
            - KL divergence (latent regularization)
        4. Optimize encoder + generator jointly
        """

        mean_loss = 0.0
        i = 0
        for x, _ in dataloader:
            i += 1
            x = x.to(device)
            # Decompose image into low- and high-frequency components
            x_L_org, x_H_org = process_2gaus(x, opt)

            # Concatenate original + frequency components along channel dimension
            # Final input shape: [B, 3 + 2, H, W]
            x = torch.cat((x, x_L_org, x_H_org), dim=1)

            b = x.size(0)
            target = process_target(x)
            [z, mu, logvar] = netE(x)
            # Reconstruct image from latent vector
            recon = netG(z)

            recon = recon.contiguous()
            recon = recon.view(-1, 256)
            recl = loss_fn(recon, target)
            recl = torch.sum(recl) / b

            kld = KL_div(mu, logvar)
            loss = recl + opt.beta * kld.mean()

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward(retain_graph=True)

            optimizer1.step()
            optimizer2.step()
            rec_l.append(recl.detach().item())
            kl.append(kld.mean().detach().item())
            mean_loss = (mean_loss * i + loss.detach().item()) / (i + 1)

            # Save model periodically for reproducibility and later evaluation
            if not i % 100:
                print(f"epoch:{epoch} recon:{np.mean(rec_l):.6f} kl:{np.mean(kl):.6f} ")
                if opt.print_text:
                    txt = open(
                        opt.experiment + f"/ngf_{ngf}_nz_{nz}_beta_{beta}_epoch.txt",
                        "a",
                    )
                    txt.writelines(
                        f"epoch:{epoch} recon:{np.mean(rec_l):.6f} kl:{np.mean(kl):.6f}"
                    )
                    txt.writelines("\n")
                    txt.close()

        history.append(mean_loss)
        scheduler1.step()
        scheduler2.step()
        now = datetime.today()

        if epoch % opt.model_save_num == opt.model_save_num - 1 or epoch==(opt.num_epoch-1):
            save_dict = {
                "epoch": epoch,
                "state_dict_E": netE.state_dict(),
                "state_dict_G": netG.state_dict(),
            }
            save_name_pth = (
                os.path.join(opt.experiment
                , f"net_ngf_{ngf}_nz_{nz}_beta_{beta}_epoch_{epoch+1}.pth")
            )

            torch.save(save_dict, save_name_pth)


if __name__ == "__main__":
    main()