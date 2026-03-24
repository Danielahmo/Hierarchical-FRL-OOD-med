"""
Evaluation script for OOD detection using VAE + Frequency Regularization.

Author: Daniela

Key ideas:
- Compute Negative log likelihood (NLL) using importance sampling
- Compute likelihood with compression-based complexity
- Evaluate across:
    - ID / near-OOD / far-OOD datasets
    - Noise corruption (SNR)
    - Appearance shifts (gamma, motion)


"""

import argparse, random
import cv2
import numpy as np

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
import torch.utils.data
import DCGAN_VAE_freq as DVAE


import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils_2D import process_target, process_2gaus
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os

import torchio as tio
from tqdm import tqdm

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

        target = process_target(x)
        recon = recon.contiguous()
        recon = recon.view(-1, 256)

        cross_entropy = F.cross_entropy(recon, target, reduction="none")
        log_p_x_z_org = -torch.sum(cross_entropy.view(b, -1), 1)

        weights_org = log_p_x_z_org + log_p_z - log_q_z_x
    return weights_org


def process_only_nan(NLL_loss):
    if np.isnan(NLL_loss):
        NLL_loss = 1e30
    return NLL_loss

class MidogDataset(torch.utils.data.Dataset):
    """
    MIDOG dataset loader.

    Supports:
    - Single txt file
    - Multiple txt files (merged)

    Each line:
        <image_path> <label>

    Returns:
        (image, ood_flag)
    """
    def __init__(self, list_images, ood, data_path, txt_path, transforms=None):
        self.data_path = data_path
        self.txt_path = txt_path
        if type(list_images)==str: 
            self.list_images = pd.read_csv(os.path.join(self.txt_path,list_images), sep=' ', header=None, names=['path', 'label'])
        else: 
            df = pd.DataFrame(columns = ['path', 'label'])
            for l in list_images: 
                df = df._append(pd.read_csv(os.path.join(self.txt_path,l), sep=' ', header=None, names=['path', 'label']), ignore_index=True)
            self.list_images = df  
        self.transforms = transforms
        self.ood = ood

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        im = Image.open(os.path.join(self.data_path,self.list_images['path'].iloc[idx])).convert('RGB')
        label = self.list_images['label'][idx]
        name_image = self.list_images['path'].iloc[idx]

        if self.transforms:
            im = self.transforms(im)

        return (name_image, im, self.ood)

def add_gaussian_noise_snr_safe(img, snr_db):
    img = torch.Tensor.numpy(img.cpu())
    signal_power = np.mean(img**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), img.shape)
    noisy_img = img + noise
    clip = np.clip(noisy_img, 0, 1)  # Returns normalized image
    return torch.from_numpy(clip).type(torch.cuda.FloatTensor)

class RandomGamma:
    def __init__(self, gamma_range, gain=1.0):
        self.gamma_range = gamma_range
        self.gain = gain

    def __call__(self, img):
        gamma = random.uniform(*self.gamma_range)
        return transforms.functional.adjust_gamma(img, gamma, gain=self.gain)

class MotionArtifactTransform:
    def __init__(self, degrees=10, translation=10, p=1.0):
        self.motion_transform = tio.RandomMotion(
            degrees=degrees,
            translation=translation,
            p=p  # Probability of applying the transform
        )

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Add a dummy z-dimension (depth)
        if img.ndim == 3:
            img = img.unsqueeze(-1)  # (C, H, W) -> (C, H, W, 1)

        subject = tio.Subject(
            image=tio.ScalarImage(tensor=img)
        )
        transformed = self.motion_transform(subject)
        img_transformed = transformed.image.tensor

        # Remove dummy z-dim to get back to (C, H, W)
        img_transformed = img_transformed.squeeze(-1)
        return img_transformed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade_off_ratio", type=float, default=1)
    parser.add_argument("--gauss_size", type=int, default=5)
    parser.add_argument("--test_num", type=int, default=5000)
    parser.add_argument("--seed_val", type=int, default=2021)

    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument("--ngpu", type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--imageSize",type=int,default=64, help="the height / width of the input image to network",)
    parser.add_argument("--state_dict",type=str, help="path to saved checkpoints and weights",)
    parser.add_argument('--repeat', type=int, default=20)
    parser.add_argument("--batchSize", type=int, default=1, help="input batch size")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--txt_path", type=str, help="path for the txt list files for midog")
    

    args = parser.parse_args()
    random.seed(args.seed_val)
    np.random.seed(args.seed_val)
    torch.manual_seed(args.seed_val)
    cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_path = args.save_path
    nc = int(args.nc) + 2  # frequency channel
    gauss_size = args.gauss_size

    auroc_list = []

    ngpu = int(args.ngpu)
    nz = args.nz
    ngf = int(args.ngf)

    netG = DVAE.DCGAN_G(args.imageSize, nz, nc, ngf, ngpu)
    netE = DVAE.Encoder(args.imageSize, nz, nc, ngf, ngpu)
    checkpoint = torch.load(args.state_dict, map_location=device)
    state_G = checkpoint["state_dict_G"]
    state_E = checkpoint["state_dict_E"]
    netG.load_state_dict(state_G)
    netE.load_state_dict(state_E)
    netG.to(device)
    netG.eval()
    netE.to(device)
    netE.eval()

    trans = transforms.Compose(
        [
            transforms.Resize((args.imageSize, args.imageSize)),
            transforms.ToTensor(),
        ]
    )

    id_txt = 'test_midog.txt'
    id_data = MidogDataset( ood=0, list_images=id_txt, data_path=args.data_path, txt_path=args.txt_path, transforms=trans)
    id_dataloader = DataLoader(id_data, batch_size=args.batchSize)

    csid_txt = ['test_midog_1b.txt', 'test_midog_1c.txt']
    csid_data = MidogDataset( ood=1, list_images=csid_txt,data_path=args.data_path, txt_path=args.txt_path, transforms=trans)
    csid_dataloader = DataLoader(csid_data, batch_size=args.batchSize)

    nearood_txt = ['test_midog_2.txt', 'test_midog_3.txt', 'test_midog_4.txt', 'test_midog_5.txt', 'test_midog_6a.txt', 'test_midog_6b.txt', 'test_midog_7.txt']
    nearood_data = MidogDataset( ood=1, list_images=nearood_txt,data_path=args.data_path, txt_path=args.txt_path, transforms=trans)
    nearood_dataloader = DataLoader(nearood_data, batch_size=args.batchSize)

    farood_txt = ['test_midog_ccagt.txt', 'test_midog_fnac2019.txt']
    farood_data = MidogDataset( ood=1, list_images=farood_txt, data_path=args.data_path, txt_path=args.txt_path, transforms=trans)
    farood_dataloader = DataLoader(farood_data, batch_size=args.batchSize)

    list_data_type = ['in', 'csid', 'nearood', 'farood']

    ######################################################################
    

    for ood_index, dataloader in enumerate([id_dataloader, csid_dataloader, nearood_dataloader]):#, farood_dataloader]):
        """
        Evaluate model on:
        - ID data
        - Close OOD
        - Near OOD
        - Far OOD

        Compute:
        - NLL score
        - FRL score (NLL - compression penalty)
        """
        difference = []
        NLL = []
        ood_ = []
        names = []
        df = pd.DataFrame(columns=['Image', 'NLL', 'FRL'])
        print(f'Computing for {list_data_type[ood_index]}')

        for i, x in enumerate(tqdm(dataloader)):
            try:
                name, x, _ = x
            except:
                print('Exception passed')
                pass

            x_L_org, x_H_org = process_2gaus(x.to(device), args)

            x = torch.cat((x.to(device), x_L_org.to(device), x_H_org.to(device)), dim=1)

            x = x.expand(args.repeat, -1, -1, -1).contiguous()
            weights_agg = []
            with torch.no_grad():
                x = x.to(device)
                b = x.size(0)
                [z, mu, logvar] = netE(x)
                recon = netG(z)
                mu = mu.view(mu.size(0), mu.size(1))
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0), z.size(1))
                weights = store_NLL(x, recon, mu, logvar, z, args.repeat)
                weights_agg.append(weights)
                weights_agg = torch.stack(weights_agg).view(-1)
                NLL_loss = compute_NLL(weights_agg).detach().cpu().numpy()

                img = x[0, : args.nc, :, :].permute(1, 2, 0)
                img = img.detach().cpu().numpy()
                img *= 255
                img = img.astype(np.uint8)
                img_encoded = cv2.imencode(
                    ".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
                )
                L = len(img_encoded[1]) * 8
                difference.append(NLL_loss - args.trade_off_ratio * L)
                NLL.append(NLL_loss)
                names.append(name[0])
   

        difference = process_all_score(difference)
        df['Image'] = names
        df['NLL'] = NLL
        df['FRL'] = difference
        df.to_csv(os.path.join(save_path, list_data_type[ood_index]+'_scores.csv'))


        

    ######################################################################

    """
    Evaluate robustness to distribution shifts:
    - Noise (SNR levels)
    - Intensity changes (gamma)
    - Motion artifacts
    """
    random.seed(args.seed_val)
    np.random.seed(args.seed_val)
    torch.manual_seed(args.seed_val)

    for snr_db in {10,20,30}:
        print('Testing for ', str(snr_db), 'snr')


        difference = []
        NLL = []
        ood_ = []
        names = []
        df = pd.DataFrame(columns=['Image', 'NLL', 'FRL'])

        for i, x in enumerate(id_dataloader):
            try:
                name, x, _ = x
                
            except:
                print('Exception passed')
                pass

            x_L_org, x_H_org = process_2gaus(x.to(device), args)

            x = torch.cat((x.to(device), x_L_org.to(device), x_H_org.to(device)), dim=1)
            x = add_gaussian_noise_snr_safe(x, snr_db)
            x = x.to(device)
            if snr_db > 0:
                x = x.expand(args.repeat, -1, -1, -1).contiguous()
            weights_agg = []
            with torch.no_grad():
                x = x.to(device)
        
                b = x.size(0)
                [z, mu, logvar] = netE(x)
                recon = netG(z)
                mu = mu.view(mu.size(0), mu.size(1))
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0), z.size(1))
                weights = store_NLL(x, recon, mu, logvar, z, args.repeat)
                weights_agg.append(weights)
                weights_agg = torch.stack(weights_agg).view(-1)
                NLL_loss = compute_NLL(weights_agg).detach().cpu().numpy()

                img = x[0, : args.nc, :, :].permute(1, 2, 0)
                img = img.detach().cpu().numpy()
                img *= 255
                img = img.astype(np.uint8)
                img_encoded = cv2.imencode(
                    ".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
                )
                L = len(img_encoded[1]) * 8
                difference.append(NLL_loss - args.trade_off_ratio * L)
                NLL.append(NLL_loss)
                names.append(name[0])
   

        difference = process_all_score(difference)
        df['Image'] = names
        df['NLL'] = NLL
        df['FRL'] = difference
        df.to_csv(os.path.join(save_path, 'snr_'+str(snr_db)+'_scores.csv'))
   

            
    trans_gamma = transforms.Compose(
        [
            transforms.Resize((args.imageSize, args.imageSize)),
            transforms.ToTensor(),
            RandomGamma(gamma_range=(0.5, 1.5)),

        ]
    )
    trans_motion = transforms.Compose(
        [
            transforms.Resize((args.imageSize, args.imageSize)),
            transforms.ToTensor(),
            MotionArtifactTransform(),

        ]
    )
    random.seed(args.seed_val)
    np.random.seed(args.seed_val)
    torch.manual_seed(args.seed_val)


    id_data_gamma = MidogDataset( ood=0, list_images=id_txt, data_path=args.data_path, txt_path=args.txt_path, transforms=trans_gamma)
    id_dataloader_gamma = DataLoader(id_data_gamma, batch_size=args.batchSize)

    id_data_motion = MidogDataset( ood=0, list_images=id_txt, data_path=args.data_path, txt_path=args.txt_path, transforms=trans_motion)
    id_dataloader_motion= DataLoader(id_data_motion, batch_size=args.batchSize)
    

    ######################################################################

    dataloaders = ['gamma', 'motion']
    
    for index, ood_dataloader in enumerate([id_dataloader_gamma, id_dataloader_motion]):
        print('Testing for ', dataloaders[index])


        difference = []
        NLL = []
        ood_ = []
        names = []
        df = pd.DataFrame(columns=['Image', 'NLL', 'FRL'])

        for i, x in enumerate(ood_dataloader):
            try:
                name, x, _ = x
                
            except:
                print('Exception passed')
                pass
        

            x = torch.cat((x.to(device), x_L_org.to(device), x_H_org.to(device)), dim=1)
            
            x = x.expand(args.repeat, -1, -1, -1).contiguous()
            weights_agg = []
            with torch.no_grad():
                x = x.to(device)
              
                b = x.size(0)
                [z, mu, logvar] = netE(x)
                recon = netG(z)
                mu = mu.view(mu.size(0), mu.size(1))
                logvar = logvar.view(logvar.size(0), logvar.size(1))
                z = z.view(z.size(0), z.size(1))
                weights = store_NLL(x, recon, mu, logvar, z, args.repeat)
                weights_agg.append(weights)
                weights_agg = torch.stack(weights_agg).view(-1)
                NLL_loss = compute_NLL(weights_agg).detach().cpu().numpy()

                img = x[0, : args.nc, :, :].permute(1, 2, 0)
                img = img.detach().cpu().numpy()
                img *= 255
                img = img.astype(np.uint8)
                img_encoded = cv2.imencode(
                    ".png", img, [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
                )
                L = len(img_encoded[1]) * 8
                difference.append(NLL_loss - args.trade_off_ratio * L)
                NLL.append(NLL_loss)
                names.append(name[0])
   

        difference = process_all_score(difference)
        df['Image'] = names
        df['NLL'] = NLL
        df['FRL'] = difference
        df.to_csv(os.path.join(save_path, dataloaders[index]+'_scores.csv'))
   



            
    