import numpy as np
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import time
import models.dcgan3D as WGAN_Models
import models.VAE_models as VAE_Models
import models.CVGAN_conv3d as VGAN
import json
import pkbar
import redis
import time
import multiprocessing as mp
from multiprocessing import Pool,RLock
import os 
import pickle
import json


def get_parser():
    parser = argparse.ArgumentParser(
        description='Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--nbsize', action='store',
                        type=int, default=1,
                        help='Batch size for generation')

    parser.add_argument('--nevents', action='store',
                        type=int, default=1000,
                        help='Desired number of showers')
    
    parser.add_argument('--maxE', action='store',
                        type=int, default=50,
                        help='Maximum energy of the shower')
    
    parser.add_argument('--minE', action='store',
                        type=int, default=50,
                        help='Maximum energy of the shower')

    parser.add_argument('--model', action='store',
                        type=str, default="wgan",
                        help='type of model')


    return parser



def wGAN(model, number, E_max, E_min, batchsize, fixed_noise, input_energy, device):


    pbar = pkbar.Pbar(name='Generating {} photon showers with energies [{},{}]'.format(number, E_max,E_min), target=number)
    
    fake_list=[]
    energy_list = []
    
    for i in np.arange(0, number, batchsize):
        with torch.no_grad():
            fixed_noise.uniform_(-1,1)
            input_energy.uniform_(E_min,E_max)
            fake = model(fixed_noise, input_energy)
            fake = fake.data.cpu().numpy()
            fake_list.append(fake)
            energy_list.append(input_energy.data.cpu().numpy())
            pbar.update(i- 1 + batchsize)

    energy_full = np.vstack(energy_list)
    fake_full = np.vstack(fake_list)
    fake_full = fake_full.reshape(len(fake_full), 30, 30, 30)

    return fake_full, energy_full

def BibAE(model, model_PostProcess, number, E_max, E_min, batchsize, latent_dim,  device='cpu', thresh=0.0):
 
    z = torch.cuda.FloatTensor(batchsize, latent_dim)
    E = torch.cuda.FloatTensor(batchsize, 1)

    fake_list=[]
    energy_list = []

    for i in np.arange(0, number, batchsize):
        with torch.no_grad():
            z.normal_()
            E.uniform_(E_min*100, E_max*100)
           
            data = model(x=z, E_true=E, z=z, mode='decode')
            dataPP = model_PostProcess.forward(data, E)

            dataPP = dataPP.data.cpu().numpy()
            fake_list.append(dataPP)
            energy_list.append(E.data.cpu().numpy())
    
    fake_full = np.vstack(fake_list)
    energy_full = np.vstack(energy_list)
    fake_full = fake_full.reshape(len(fake_full), 30, 30, 30)

    return fake_full, energy_full



def shower_photons(nevents, model, bsize, emax, emin): 
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    
    if model == 'wgan':
        LATENT_DIM = 100
        ngf = 32
        model_WGAN = WGAN_Models.DCGAN_G(ngf,LATENT_DIM).to(device)
        model_WGAN = nn.DataParallel(model_WGAN)
        weightsGAN = 'weights/wgan.pth'
        model_WGAN.load_state_dict(torch.load(weightsGAN, map_location=torch.device(device)))

        if cuda:
            noise = torch.cuda.FloatTensor(bsize, LATENT_DIM, 1,1,1)
            energy = torch.cuda.FloatTensor(bsize, 1,1,1,1)
        else:
            noise = torch.FloatTensor(bsize, LATENT_DIM, 1,1,1)
            energy = torch.FloatTensor(bsize, 1,1,1,1) 
    
        
        showers, energy = wGAN(model_WGAN, nevents, emax, emin, bsize, noise, energy, device)
        energy = energy.flatten()
    
    elif model == 'bib-ae':
        args = {
        'E_cond' : True,
        'latent' : 512
        }

        model = VAE_Models.BiBAE_F_3D_LayerNorm_SmallLatent(args, device=device, z_rand=512-24,
                                                        z_enc=24).to(device) 

        model = nn.DataParallel(model)
        checkpoint = torch.load('weights/bib-ae-PP.pth')

        model.load_state_dict(checkpoint['model_state_dict'])

        model_P = VAE_Models.PostProcess_Size1Conv_EcondV2(bias=True, out_funct='none').to(device)
        model_P = nn.DataParallel(model_P)
        
        model_P.load_state_dict(checkpoint['model_P_state_dict'])
        
        showers, energy = BibAE(model, model_P, nevents, emax, emin, bsize, 512, device)
        energy = energy.flatten()
    
    return showers, energy  

def write_to_cache(showers, model_name, N):
    
    ## get the dictionary
    f = open('cell-map.pickle', 'rb')
    cmap = pickle.load(f)  
    
    pbar_cache = pkbar.Pbar(name='Writing to cache', target=N)

    elist = []

    for event in range(len(showers)):      ## events
        elist.append([])
        for layer in range(30):              ## loop over layers
            nx, nz = np.nonzero(showers[event][layer])   ## get non-zero energy cells  
            for j in range(0,len(nx)):
                try:
                    cell_energy = showers[event][layer][nx[j]][nz[j]]
                    tmp = cmap[(layer, nx[j], nz[j])]
                    elist[event].append([tmp[0], tmp[1], tmp[2], cell_energy, tmp[3], tmp[4]])
                except KeyError:
                    # Key is not present
                    pass
                    
                                
        
        pbar_cache.update(event)

    
    data = np.array(elist)
    np.savez('cache_'+model_name, x=data)
    

         



if __name__ == "__main__":

    parser = get_parser()
    parse_args = parser.parse_args() 
    
    bsize = parse_args.nbsize
    N = parse_args.nevents
    emax = parse_args.maxE
    emin = parse_args.minE
    model_name = parse_args.model

    showers, energy = shower_photons(N, model_name, bsize, emax, emin)
    write_to_cache(showers, model_name, N)
    
