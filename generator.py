import numpy as np
import argparse
import torch
import array as arr
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import models.dcgan3D as WGAN_Models
import models.VAE_models as VAE_Models
import models.GAN as VGAN


import streamlit as st
import pkbar
import time
import os 
import pickle
import random

import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('GTKAgg')


'''
# Synthetic Shower Generation 

Shower simulation is CPU intensive. Generative machine learning models based on deep neural networks are speeding up this task by several orders of magnitude

## Generative Models:
*  Generative Adversarial Network (GAN)
*  Wasserstein Generative Adversarial Network (WGAN) 
*  Bounded Information Bottleneck Variational Autoendoder (BiB-AE)

'''



from pyLCIO import EVENT, UTIL, IOIMPL, IMPL

def get_parser():
    parser = argparse.ArgumentParser(
        description='Generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--nbsize', action='store',
                        type=int, default=10,
                        help='Batch size for generation')

    parser.add_argument('--nevents', action='store',
                        type=int, default=1000,
                        help='Desired number of showers')
    
    parser.add_argument('--model', action='store',
                        type=str, default="wgan",
                        help='type of model (bib-ae , wgan or gan)')

    parser.add_argument('--output', action='store',
                        type=str, help='Name of the output file')



    return parser

def getTotE(data, xbins, ybins, layers=30):
    data = np.reshape(data,[-1, layers*xbins*ybins])
    etot_arr = np.sum(data, axis=(1))
    return etot_arr

def make_plots(fake_data):
    
    status_text = st.empty()
    status_text.text("Making some plots...")

    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    etot_fake = getTotE(fake_data, xbins=30, ybins=30)

    n, bins, patches = axs.hist(etot_fake.flatten(), bins=25, range=[0, 1500],
               weights=np.ones_like(etot_fake)/(float(len(etot_fake))),
               label = "Simulated hits", color= 'blue',
               histtype='stepfilled')
    

    st.pyplot(fig)




def wGAN(model, number, E_max, E_min, batchsize, fixed_noise, input_energy, device):


    #pbar = pkbar.Pbar(name='Generating {} photon showers with energies [{},{}]'.format(number, E_max,E_min), target=number)
    
    fake_list=[]
    energy_list = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Generating {} photon showers. Current models is WGAN".format(number))

    for i in np.arange(batchsize, number+1, batchsize):
        with torch.no_grad():
            fixed_noise.uniform_(-1,1)
            input_energy.uniform_(E_min,E_max)
            fake = model(fixed_noise, input_energy)
            fake = fake.data.cpu().numpy()
            
            fake_list.append(fake)
            energy_list.append(input_energy.data.cpu().numpy())

            
            progress_bar.progress(int(i* 100 / number))            
            

    energy_full = np.vstack(energy_list)
    fake_full = np.vstack(fake_list)
    fake_full = fake_full.reshape(len(fake_full), 30, 30, 30)

    
    progress_bar.empty()
    

    return fake_full, energy_full


def vGAN(model, number, E_max, E_min, batchsize, fixed_noise, input_energy, device):


    #pbar = pkbar.Pbar(name='Generating {} photon showers with energies [{},{}]'.format(number, E_max,E_min), target=number)
    
    fake_list=[]
    energy_list = []
    
    model.eval()

    for i in np.arange(0, number, batchsize):
        with torch.no_grad():
            fixed_noise.uniform_(-1,1)
            input_energy.uniform_(E_min,E_max)
            fake = model(fixed_noise, input_energy)
            fake = fake.data.cpu().numpy()
            fake_list.append(fake)
            energy_list.append(input_energy.data.cpu().numpy())
            #pbar.update(i- 1 + batchsize)

    energy_full = np.vstack(energy_list)
    fake_full = np.vstack(fake_list)
    fake_full = fake_full.reshape(len(fake_full), 30, 30, 30)

    return fake_full, energy_full

def BibAE(model, model_PostProcess, number, E_max, E_min, batchsize, latent_dim,  device='cpu', thresh=0.0):
 
    if device == 'cuda': 
        z = torch.cuda.FloatTensor(batchsize, latent_dim)
        E = torch.cuda.FloatTensor(batchsize, 1)
    else :
        z = torch.FloatTensor(batchsize, latent_dim)
        E = torch.FloatTensor(batchsize, 1)
        

    fake_list=[]
    energy_list = []

    for i in np.arange(0, number, batchsize):
        with torch.no_grad():
            z.normal_()
            E.uniform_(E_min, E_max)
           
            data = model(x=z, E_true=E, z=z, mode='decode')
            dataPP = F.relu(model_PostProcess.forward(data, E))

            dataPP = dataPP.data.cpu().numpy()
            #data = data.data.cpu().numpy()
            
            ## hard cut for noisy images
            #dataPP[dataPP < 0.001] = 0.00
            
            fake_list.append(data)
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
    
    elif model == 'vgan':
        netG = VGAN.Generator(1).to(device)
        #netG = nn.DataParallel(netG)
        w = 'weights/vgan.pth'
        checkpoint = torch.load(w, map_location=torch.device(device))
        netG.load_state_dict(checkpoint['Generator'])

        LATENT_DIM = 100
        if cuda:
            noise = torch.cuda.FloatTensor(bsize, LATENT_DIM, 1,1,1)
            energy = torch.cuda.FloatTensor(bsize, 1,1,1,1)
        else:
            noise = torch.FloatTensor(bsize, LATENT_DIM, 1,1,1)
            energy = torch.FloatTensor(bsize, 1,1,1,1) 
    
        
        showers, energy = vGAN(netG, nevents, emax, emin, bsize, noise, energy, device)
        energy = energy.flatten()



    elif model == 'bib-ae':
        args = {
        'E_cond' : True,
        'latent' : 512
        }

        model = VAE_Models.BiBAE_F_3D_LayerNorm_SmallLatent(args, device=device, z_rand=512-24,
                                                        z_enc=24).to(device) 

        model = nn.DataParallel(model)
        checkpoint = torch.load('weights/bib-ae-PP.pth', map_location=torch.device(device))

        model.load_state_dict(checkpoint['model_state_dict'])

        model_P = VAE_Models.PostProcess_Size1Conv_EcondV2(bias=True, out_funct='none').to(device)
        model_P = nn.DataParallel(model_P)
        
        model_P.load_state_dict(checkpoint['model_P_state_dict'])
        
        showers, energy = BibAE(model, model_P, nevents, emax, emin, bsize, 512, device)
        energy = energy.flatten()
    
    return showers, energy  


def write_to_lcio(showers, energy, model_name, outfile, N):
    

    status_text = st.empty()
    
    status_text.text("Writing to LCIO files...")
    progress_bar = st.progress(0)

    ## get the dictionary
    f = open('cell-map.pickle', 'rb')
    cmap = pickle.load(f)  
    
    #pbar_cache = pkbar.Pbar(name='Writing to lcio files', target=N)

    wrt = IOIMPL.LCFactory.getInstance().createLCWriter( )

    wrt.open( outfile , EVENT.LCIO.WRITE_NEW ) 

    random.seed()



    #========== MC particle properties ===================
    genstat  = 1
    charge = 0.
    mass = 0.00 
    decayLen = 1.e32 
    pdg = 22


    # write a RunHeader
    run = IMPL.LCRunHeaderImpl() 
    run.setRunNumber( 0 ) 
    run.parameters().setValue("Generator", model_name)
    run.parameters().setValue("PDG", pdg )
    wrt.writeRunHeader( run ) 

    for j in range( 0, N ):

        ### MC particle Collections
        colmc = IMPL.LCCollectionVec( EVENT.LCIO.MCPARTICLE ) 

        ## we are shooting 90 deg. ECAL 
        px = 0.00 
        py = energy[j] 
        pz = 0.00 

        vx = 0.00
        vy = 50.00
        vz = 0.000

        epx = 0.00
        epy = 1800
        epz = 0.00

        momentum = arr.array('f',[ px, py, pz ] )  
        vertex = arr.array('d',[vx,vy,vz])
        endpoint = arr.array('d', [epx,epy,epz])


        mcp = IMPL.MCParticleImpl() 
        mcp.setGeneratorStatus( genstat ) 
        mcp.setMass( mass )
        mcp.setPDG( pdg ) 
        mcp.setMomentum( momentum )
        mcp.setCharge( charge )
        mcp.setVertex(vertex)
        mcp.setEndpoint(endpoint)

        colmc.addElement( mcp )
        
        evt = IMPL.LCEventImpl() 
        evt.setEventNumber( j ) 
        evt.addCollection( colmc , "MCParticle" )


        ### Calorimeter Collections
        col = IMPL.LCCollectionVec( EVENT.LCIO.SIMCALORIMETERHIT ) 
        flag =  IMPL.LCFlagImpl(0) 
        flag.setBit( EVENT.LCIO.CHBIT_LONG )
        flag.setBit( EVENT.LCIO.CHBIT_ID1 )

        col.setFlag( flag.getFlag() )

        col.parameters().setValue(EVENT.LCIO.CellIDEncoding, 'system:0:5,module:5:3,stave:8:4,tower:12:4,layer:16:6,wafer:22:6,slice:28:4,cellX:32:-16,cellY:48:-16')
        evt.addCollection( col , "EcalBarrelCollection" )

        

        for layer in range(30):              ## loop over layers
            nx, nz = np.nonzero(showers[j][layer])   ## get non-zero energy cells  
            for k in range(0,len(nx)):
                try:
                    cell_energy = showers[j][layer][nx[k]][nz[k]] / 1000.0
                    tmp = cmap[(layer, nx[k], nz[k])]

                    sch = IMPL.SimCalorimeterHitImpl()

                    position = arr.array('f', [tmp[0],tmp[1],tmp[2]])
    
                    sch.setPosition(position)
                    sch.setEnergy(cell_energy)
                    sch.setCellID0(int(tmp[3]))
                    sch.setCellID1(int(tmp[4]))
                    col.addElement( sch )

                except KeyError:
                    # Key is not present
                    pass
                    
                                
        progress_bar.progress(int(j*100/N))
        
        wrt.writeEvent( evt ) 

    progress_bar.empty()
    st.info('LCIO file was created: {}'.format(os.getcwd() + '/'+outfile))
    status_text_rec = st.empty()
    status_text_rec.text("We are ready to run reconstruction via iLCsoft")
    wrt.close() 


if __name__ == "__main__":

    parser = get_parser()
    parse_args = parser.parse_args() 
    
    bsize = parse_args.nbsize
    N = parse_args.nevents
    
    model_name = parse_args.model
    output_lcio = parse_args.output

    

    wgan_check = st.checkbox('Generate photons with WGAN model')
    values = st.slider( 'Select a range of photon energies', 0.0, 100.0, (25.0, 75.0))

    emin=values[0]
    emax=values[1]

    if wgan_check:
        showers, energy = shower_photons(N, model_name, bsize, emax, emin)
        write_to_lcio(showers, energy, model_name, output_lcio, N)
        make_plots(showers)
    
