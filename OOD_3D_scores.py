"""
3D OOD Detection using VAE + Frequency Regularization.

Description:
- Computes sample-level NLL using importance sampling
- Evaluates robustness under realistic medical artifacts:
    - Elastic deformation
    - Ghosting
    - Noise
    - Swap artifacts

Key idea:
- Augment input with frequency decomposition (low/high)
- Evaluate how likelihood behaves under distribution shifts
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import os, sys
from pathlib import Path
from torch.utils.data import Dataset
import nibabel as nib


from tqdm import tqdm
import VAE3D as DVAE


import pandas as pd
from PIL import Image
import torchio as tio
from torchio import RandomElasticDeformation, RandomMotion, RandomGhosting, RandomSpike, RandomBlur, RandomNoise, RandomSwap
from utils import process_freq

def process_all_score(score):
    for i in range(len(score)):
        score[i] = process_only_nan(score[i])
    return score


def compute_NLL(weights):
    with torch.no_grad():
        NLL_loss = -(
            torch.log(torch.mean(torch.exp(weights - weights.max()))) + weights.max()
        )
    return NLL_loss


def store_NLL(x, recon, mu, logvar, z, repeat):
    with torch.no_grad():
        sigma = torch.exp(0.5 * logvar)
        b = x.size(0)
        log_p_z = -torch.sum(z**2 / 2 + np.log(2 * np.pi) / 2, 1)
        z_eps = (z - mu) / sigma
        z_eps = z_eps.view(repeat, -1)
        log_q_z_x = -torch.sum(z_eps**2 / 2 + np.log(2 * np.pi) / 2 + logvar / 2, 1)

        mse = F.mse_loss(recon.view_as(x), x, reduction="none")

        log_p_x_z_org = -torch.sum(mse.view(b, -1), 1)
        weights_org = log_p_x_z_org + log_p_z - log_q_z_x

    return weights_org


def process_only_nan(NLL_loss):
    if np.isnan(NLL_loss):
        NLL_loss = 1e30
    return NLL_loss


def KL_div(mu, logvar, reduction="avg"):
    mu = mu.view(mu.size(0), mu.size(1))
    logvar = logvar.view(logvar.size(0), logvar.size(1))
    if reduction == "sum":
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:
        KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
        return KL

def normalize_sample_abdomen(score):
    return np.clip((score - A_SAMPLE) / (B_SAMPLE - A_SAMPLE), 0, 1)

def normalize_pixel_abdomen(pixel_map):
    return np.clip((pixel_map - A_PIXEL) / (B_PIXEL - A_PIXEL), 0, 1)

class Dataloader_3DScan(tio.SubjectsDataset):
    """
    3D medical dataset using TorchIO.

    Each sample:
        - Loaded as NIfTI or compatible format
        - Stored as TorchIO Subject
        - Supports optional artifact simulation

    Args:
        trans_basic: preprocessing (resize, normalization)
        trans_noise: artifact simulation (OOD generation)
    """
    def __init__(self, list_images, data_path='abdomen_hugging/', trans_basic=None, trans_noise=None):
        self.data_path = data_path
        self.list_images = pd.read_csv(list_images, header=None)[0][:10]
        self.trans_basic = trans_basic
        self.trans_noise = trans_noise

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        im = tio.Subject(image=tio.ScalarImage(os.path.join(self.data_path,self.list_images[idx])),
                        name_image= self.list_images[idx],
                        shape_info = None)
        im['shape'] = im['image'].shape

        if self.trans_noise:
            im = self.trans_noise(im)
 
        if self.trans_basic:
            im = self.trans_basic(im)


        return im



def main():
    parser = argparse.ArgumentParser()
   
    parser.add_argument( "--workers", type=int, help="number of data loading workers", default=4)
    parser.add_argument("--batchSize", type=int, default=1, help="input batch size")
    parser.add_argument("--imageSize",type=int,default=64, help="the height / width of the input image to network",)
    parser.add_argument("--nc", type=int, default=1, help="input image channels")
    parser.add_argument("--nz", type=int, default=100, help="size of the latent z vector") 
    parser.add_argument("--ngf", type=int, default=32, help="hidden channel sieze")

    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam. default=0.9")
    parser.add_argument("--beta", type=float, default=1, help="beta for beta-vae")

    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--gauss_size", type=int, default=5)

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--test_path', type=str, default= '/brain/val_patients.txt')
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--state_dict', type=str)

    parser.add_argument("--trade_off_ratio", type=float, default=1)
    parser.add_argument("--test_num", type=int, default=5000)
    parser.add_argument("--seed_val", type=int, default=2021)
    parser.add_argument('--repeat', type=int, default=10)


    opt = parser.parse_args()
    ngpu = torch.cuda.device_count()

    random.seed(opt.seed_val)
    np.random.seed(opt.seed_val)
    torch.manual_seed(opt.seed_val)
    cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    is_dp = ngpu > 1

    #############################################################

    trans_basic = tio.Compose([tio.Resize(opt.imageSize),tio.ZNormalization()])
    trans_elastic = tio.Compose([RandomElasticDeformation(num_control_points=(11),  locked_borders=0,)])
    trans_ghosting = tio.Compose([RandomGhosting()])
    trans_noise = tio.Compose([RandomNoise()])
    trans_swap = tio.Compose([ RandomSwap()])

    print('testing for data: ', opt.data_path)

    test_txt = opt.test_path

    dataset_in = Dataloader_3DScan(test_txt, data_path = opt.data_path, trans_basic=trans_basic, trans_noise=None)

    dataset_elastic = Dataloader_3DScan(test_txt, data_path = opt.data_path, trans_basic=trans_basic, trans_noise=trans_elastic)
    dataset_ghosting = Dataloader_3DScan(test_txt, data_path = opt.data_path, trans_basic=trans_basic, trans_noise=trans_ghosting)
    dataset_noise = Dataloader_3DScan(test_txt, data_path = opt.data_path, trans_basic=trans_basic, trans_noise=trans_noise)
    dataset_swap = Dataloader_3DScan(test_txt, data_path = opt.data_path, trans_basic=trans_basic, trans_noise=trans_swap)
    

    # DataLoader without DistributedSampler
    dataloader_in = tio.SubjectsLoader(dataset_in, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers), pin_memory=True, persistent_workers=True)
    dataloader_elastic = tio.SubjectsLoader(dataset_elastic, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers), pin_memory=True, persistent_workers=True)
    dataloader_ghosting = tio.SubjectsLoader(dataset_ghosting, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers), pin_memory=True, persistent_workers=True)
    dataloader_noise = tio.SubjectsLoader(dataset_noise, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers), pin_memory=True, persistent_workers=True)
    dataloader_swap = tio.SubjectsLoader(dataset_swap, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers), pin_memory=True, persistent_workers=True)


    #############################################################

    nz = int(opt.nz)
    ngf = int(opt.ngf)
    nc = int(opt.nc) + 2

    beta = opt.beta

    # Load weights from trained model

    netG = DVAE.DCGAN_G(opt.imageSize, nz, nc, ngf)
    netE = DVAE.Encoder(opt.imageSize, nz, nc, ngf)
    checkpoint = torch.load(opt.state_dict, map_location=device)
    state_G = checkpoint["state_dict_G"]
    state_E = checkpoint["state_dict_E"]
    netG.load_state_dict(state_G)
    netE.load_state_dict(state_E)
    netG.to(device)
    netG.eval()
    netE.to(device)
    netE.eval()

    # set the dataloaders

    indist = []
    dataloaders = ['in', 'elastic', 'ghosting', 'noise', 'swap']
    save_path = opt.save_path
    sample = []
    pixel = []
    labels = []
    
    
    for index, ood_dataloader in enumerate([dataloader_in, dataloader_elastic, dataloader_ghosting, dataloader_noise, dataloader_swap]):
        print('Testing for ', dataloaders[index])

        """
        Evaluate model under different distribution shifts:
        - in-distribution
        - elastic deformation
        - ghosting
        - noise
        - swap artifacts
        """

        NLL = []
        ood_ = []
        save_pixel = 0
        names=[]
        df = pd.DataFrame(columns=['Image', 'NLL'])

        for i, subject in enumerate(tqdm(ood_dataloader)):
            

            x = subject['image']['data']
            name_image = subject['name_image']
            names.append(name_image[0])
            
            c, h, w, d = subject['shape'][0]

            subject = []

            x = x.to(device)
            x_L_org, x_H_org = process_freq(x, opt.gauss_size)


            x = torch.cat((x.to(device), x_L_org.to(device), x_H_org.to(device)), dim=1)



            weights_agg = []

            #transform_upsample = tio.Compose([tio.Resize((h, w, d))])
            transform_upsample = tio.Compose([tio.Resize((64))])
            
            x_sample = x.expand(opt.repeat, -1, -1, -1, -1).contiguous()

            with torch.no_grad():
                x = x.to(device)
                x_sample = x_sample
              
                b = x_sample.size(0)
                [z, mu, logvar] = netE(x_sample)
                recon = netG(z)
                mu = mu.view(mu.size(0), mu.size(1))
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0), z.size(1))

                # SAMPLE LEVEL
                weights = store_NLL(x_sample, recon, mu, logvar, z, opt.repeat)
                weights_agg.append(weights)
                weights_agg = torch.stack(weights_agg).view(-1)
                NLL_loss = compute_NLL(weights_agg).detach().cpu().numpy()

                NLL.append(NLL_loss)

                ## PIXEL LEVEL
                
                #z_pix, _, _ = netE(x)
                #recon_pix = netG(z_pix)

                #diff = torch.pow(x-recon_pix, 2).mean(dim=1)
                #diff = transform_upsample(diff.detach().cpu().numpy())
                #diff = np.squeeze(diff)
                #pixel.append(diff)

                #if len(affine.shape)==2:
                #    final_nimg = nib.Nifti1Image(diff, affine=affine)
                #elif len(affine.shape)==3:
                #    final_nimg = nib.Nifti1Image(diff, affine=affine[0, :, :])
                #else:
                #    print('Affine not correct shape: ', str(affine.shape),' in image ', name_image)
                #final_nimg = nib.Nifti1Image(diff)

                #if save_pixel < 3:
                #    nib.save(final_nimg, save_path+'pixel/'+dataloaders[index]+'_'+ name_image)
                    #np.save(save_path+'pixel/'+dataloaders[index]+'_'+ name_image[:-7] +'.npy', diff)
                #    save_pixel = save_pixel + 1
                

        
        df['Image']= names
        df['NLL']=NLL
        df.to_csv(os.path.join(save_path,dataloaders[index]+ '_scores.csv'))  




if __name__ == "__main__":
    main()