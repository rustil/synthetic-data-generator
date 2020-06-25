import numpy as np
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import time
import models.dcgan3D as WGAN_Models
import json
import pkbar
import redis
import time
import multiprocessing as mp
from multiprocessing import Pool,RLock
import os 
import pickle
import json

def getFakeImagesGAN(model, number, E_max, E_min, batchsize, fixed_noise, input_energy, device):


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

def shower_photons(nevents, bsize, emax, emin): 
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    LATENT_DIM = 100
    ngf = 32
    model_WGAN = WGAN_Models.DCGAN_G(ngf,LATENT_DIM).to(device)
    model_WGAN = nn.DataParallel(model_WGAN)
    weightsGAN = 'weights/netG_itrs_21999.pth'
    model_WGAN.load_state_dict(torch.load(weightsGAN, map_location=torch.device(device)))


    if cuda:
        noise = torch.cuda.FloatTensor(bsize, LATENT_DIM, 1,1,1)
        energy = torch.cuda.FloatTensor(bsize, 1,1,1,1)
    else:
        noise = torch.FloatTensor(bsize, LATENT_DIM, 1,1,1)
        energy = torch.FloatTensor(bsize, 1,1,1,1) 
   
    
    showers, energy = getFakeImagesGAN(model_WGAN, nevents, emax, emin, bsize, noise, energy, device)
    energy = energy.flatten()

    return showers, energy  

def write_to_cache(showers, N):
    
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
    np.savez('cache_wgan', x=data)
    

         



if __name__ == "__main__":
    nevents = 5
    showers, energy = shower_photons(nevents, 1, 50,50)
    write_to_cache(showers, nevents)
    
