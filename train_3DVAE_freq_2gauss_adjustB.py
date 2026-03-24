"""
Train 3D VAE with frequency regularization.

Author: Danielahmo

Key contributions:
- Extension to 3D volumetric data (medical scans)
- Frequency-based input augmentation (low/high frequency channels)
- KL annealing strategy (gradual beta increase)
- Mixed precision + gradient accumulation for memory efficiency

This code is based on: https://github.com/mu-cai/FRL

Date: 2026

Cite paper: Hierarchical frequency regularization for out-of-distribution detection in generative models for medical images, ISBI 2026

"""

import argparse
import random
import numpy as np
from datetime import datetime

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch


import VAE3D as DVAE
from utils import process_freq
import torch.utils.data as data
from tqdm import tqdm

import pandas as pd
from PIL import Image
import torchio as tio
import os


def KL_div(mu, logvar, reduction="avg"):

    """
    KL divergence between N(mu, sigma) and standard normal N(0,1).

    Encourages latent space to follow a unit Gaussian prior.

    Args:
        mu: Mean of latent distribution [B, Z]
        logvar: Log-variance [B, Z]
        reduction: "sum" or "avg" (per-sample)

    Returns:
        KL divergence (scalar or per-sample tensor)
    """
    mu = mu.view(mu.size(0), mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    if reduction == "sum":
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        return KL

class Dataloader_3DScan(tio.SubjectsDataset):
    """
    Dataset for 3D medical volumes stored as .npy files.

    Expected:
        list_images: text file with relative paths (one per line)
        data_path: base directory

    Each sample:
        - Loaded as numpy array
        - Expanded to [1, D, H, W] (single-channel 3D volume)
        - Wrapped into TorchIO Subject for transformations
    """
    def __init__(self, list_images, data_path, transforms=None):
        self.data_path = data_path
        self.list_images = pd.read_csv(list_images, header=None)[0]
        self.transforms = transforms

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):

        im = tio.Subject(image = tio.ScalarImage(tensor=torch.from_numpy(np.expand_dims(np.load(os.path.join(self.data_path , self.list_images[idx][:-2])), 0))))

 
        if self.transforms:
            im = self.transforms(im)
    

        return im['image']

def main():
    parser = argparse.ArgumentParser()
   
    parser.add_argument( "--workers", type=int, help="number of data loading workers", default=4)
    parser.add_argument("--batchSize", type=int, default=16, help="input batch size")
    parser.add_argument("--imageSize",type=int,default=64, help="the height / width of the input image to network",)
    parser.add_argument("--nc", type=int, default=1, help="input image channels")
    parser.add_argument("--nz", type=int, default=100, help="size of the latent z vector") 
    parser.add_argument("--ngf", type=int, default=32, help="hidden channel sieze")
    parser.add_argument("--num_epoch", type=int, default=100, help="number of epochs to train for")  
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument('--repeat', type=int, default=20)

    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam. default=0.9")
    parser.add_argument("--beta", type=float, default=0, help="beta for beta-vae")
    parser.add_argument("--gauss_size", type=int, default=5)
    parser.add_argument("--cuda_num", type=str, default="1", help="")

    parser.add_argument('--experiment', help='Where to store samples and models')
    parser.add_argument('--train_txt', type=str, help='Path to file list') 
    parser.add_argument('--data_path', type=str)
    parser.add_argument("--print_text", default=True, help="")
    parser.add_argument("--model_save_num", type=int, default=100)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed_val", type=int, default=-1)

    opt = parser.parse_args()
    ngpu = torch.cuda.device_count()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    if opt.seed_val != -1:
        opt.manualSeed = opt.seed_val  # random.randint(1, 10000) # fix seed
    else:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)


    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    is_dp = ngpu > 1

    print('Number of GPUS available: ', ngpu)

    transform = tio.Compose([
    tio.Resize(opt.imageSize),
    tio.ZNormalization()])

    train_txt = opt.train_txt
    dataset = Dataloader_3DScan(train_txt, data_path=opt.data_path, transforms=transform)
    print(len(dataset))

    dataloader = tio.SubjectsLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers),
        pin_memory=True, 
        persistent_workers=True

    )

    if not os.path.exists(opt.experiment):

        os.mkdir(opt.experiment)

    print(f'Please see the path "{opt.experiment}" for the saved model !')

    nz = int(opt.nz)
    ngf = int(opt.ngf)

    # Increase input channels:
    # original (1) + low-frequency + high-frequency = 3 channels
    nc = int(opt.nc) + 2
    print(f"Channel {nc}, ngf {ngf}, nz {nz}")
    kl_beta = opt.beta

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    netG = DVAE.DCGAN_G(opt.imageSize, nz, nc, ngf)
    netG.apply(weights_init)

    netE = DVAE.Encoder(opt.imageSize, nz, nc, ngf)
    netE.apply(weights_init)

    # Wrap the models with DataParallel
    if is_dp:
        netG = nn.DataParallel(netG)
        netE = nn.DataParallel(netE)
        
    netE.to(device)
    netG.to(device)

    # setup optimizer
    optimizer1 = optim.Adam(netE.parameters(), lr=opt.lr, weight_decay=0)
    optimizer2 = optim.Adam(netG.parameters(), lr=opt.lr, weight_decay=0)
    scheduler1 = optim.lr_scheduler.StepLR(optimizer1, step_size=30, gamma=0.5)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=30, gamma=0.5)


    start_epoch = 0


    print("start_epoch :", start_epoch)

    netE.train()
    netG.train()

    
    rec_l = []
    kl = []
    history = []
    start = datetime.today()
    # Mixed precision for faster training and lower memory usage
    scaler = torch.cuda.amp.GradScaler()

    accum_steps = 4

    for epoch in tqdm(range(start_epoch, opt.num_epoch)):

        # Gradually increase KL weight (beta) during training
        # Helps stabilize training and avoid posterior collapse
        if epoch%10==0:
            if kl_beta<=1:
                kl_beta = min(round(kl_beta + 0.1, 1), 1)

        mean_loss = 0.0

        for i, x in enumerate(dataloader):
      
            x = x['data'].to(device)
            # Extract low- and high-frequency components from 3D volume
            x_L_org, x_H_org = process_freq(x, opt.gauss_size)

            # Concatenate along channel dimension
            # Final shape: [B, 3, D, H, W]
            x = torch.cat((x, x_L_org, x_H_org), dim=1)

            b = x.size(0)
            
            with torch.cuda.amp.autocast():
            
                [z, mu, logvar] = netE(x)
                recon = netG(z)

                recl = F.mse_loss(recon.view_as(x), x, reduction="mean")

                kld = KL_div(mu, logvar)
                loss = recl + kl_beta * kld.mean()

            # Scale loss for mixed precision and normalize by accumulation steps
            scaler.scale(loss / accum_steps).backward()

            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer1)
                scaler.step(optimizer2)
                scaler.update()
                optimizer1.zero_grad()
                optimizer2.zero_grad()

                
            rec_l.append(recl.detach().item())
            kl.append(kld.mean().detach().item())
            mean_loss = (mean_loss * i + loss.detach().item()) / (i + 1)

            if not i % 100:
                if opt.print_text:
                    txt = open(
                       os.path.join(opt.experiment , f"loss_ngf_{ngf}_nz_{nz}_epoch.txt"),
                        "a",
                    )
                    txt.writelines(
                        f"epoch:{epoch} recon:{np.mean(rec_l):.6f} kl:{np.mean(kl):.6f} kl_beta:{kl_beta:.6f} "
                    )
                    txt.writelines("\n")
                    txt.close()

        history.append(mean_loss)
        scheduler1.step()
        scheduler2.step()
        now = datetime.today()

        if (epoch % opt.model_save_num == opt.model_save_num - 1) or (epoch==(opt.num_epoch-1)):
            if is_dp: 
                save_dict = {
                    "epoch": epoch,
                    "state_dict_E": netE.module.state_dict(),
                    "state_dict_G": netG.module.state_dict(),
                }
            else:
                save_dict = {
                    "epoch": epoch,
                    "state_dict_E": netE.state_dict(),
                    "state_dict_G": netG.state_dict(),
                }

            save_name_pth = (
                opt.experiment
                + f"/net_ngf_{ngf}_nz_{nz}_epoch_{epoch+1}.pth"
            )
            torch.save(save_dict, save_name_pth)
            print('Weights saved at: ', save_name_pth)

if __name__ == "__main__":
    main()
    