import numpy as np
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import time
import models.dcgan3D as WGAN_Models
import models.VAE_models as VAE_Models
# import models.CVGAN_conv3d as VGAN
import nvgpu
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
                        type=int, default=15,
                        help='Maximum energy of the shower')

    parser.add_argument('--minE', action='store',
                        type=int, default=15,
                        help='Maximum energy of the shower')

    parser.add_argument('--model', action='store',
                        type=str, default="WGAN",
                        help='type of model')

    parser.add_argument('--nexp', action='store',
                        type=int, default=30,
                        help='number of experiments (generation)')

    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Using GPU')


    return parser



def getFakeImagesGAN(model, number, E_max, E_min, batchsize, fixed_noise, input_energy): # there's no use of device here -> removed

    fake_list=[]


    for i in np.arange(0, number, batchsize):
        with torch.no_grad():
            fixed_noise.uniform_(-1,1)
            input_energy.uniform_(E_min,E_max)

            # start_event = torch.cuda.Event(enable_timing=True) # that's not doing anything
            # end_event = torch.cuda.Event(enable_timing=True)
            #
            # start_event.record()


            fake = model(fixed_noise, input_energy)



            ## memory stuff
            #print (nvgpu.gpu_info())


            # end_event.record()
            # torch.cuda.synchronize()
            # elapsed_time_ms = start_event.elapsed_time(end_event)



            fake = fake.data.cpu().numpy()
            fake_list.append(fake)

    fake_full = np.vstack(fake_list)
    fake_full = fake_full.reshape(len(fake_full), 30, 30, 30)

    return fake_full




def getFakeImagesVAE_ENR_PostProcess(model, model_PostProcess, number, E_max, E_min, batchsize, latent_dim,  device, thresh=0.0):

    z = torch.empty([batchsize, latent_dim], device=device)
    E = torch.empty([batchsize, 1], device=device)

    fake_list=[]
    for i in np.arange(0, number, batchsize):
        with torch.no_grad():
            z.normal_()
            E.uniform_(E_min*100, E_max*100)

            data = model(x=z, E_true=E, z=z, mode='decode')
            dataPP = model_PostProcess.forward(data, E)

            dataPP = dataPP.data.cpu().numpy()
            fake_list.append(dataPP)

    fake_full = np.vstack(fake_list)
    fake_full = fake_full.reshape(len(fake_full), 30, 30, 30)

    return fake_full


def main():

    LATENT_DIM = 100
    nc = 30
    ngf = 32

    parser = get_parser()
    parse_args = parser.parse_args()

    bsize = parse_args.nbsize
    N = parse_args.nevents
    emax = parse_args.maxE
    emin = parse_args.minE
    model_name = parse_args.model
    nexp = parse_args.nexp
    gpu = parse_args.cuda and torch.cuda.is_available()
    assert gpu or not parse_args.cuda, "Cuda is chosen, but not available."
    device = torch.device("cuda" if gpu else "cpu")
    print("Device used: {}".format(gpu, device))
    if not gpu:
        torch.set_num_threads(1)



    if model_name == "WGAN":
        ### MODEL WGAN ###
        #model_WGAN = WGAN_Models.DCGAN_G(ngf,LATENT_DIM).to(device)
        model_WGAN = WGAN_Models.DCGAN_G_nonSeq(ngf,LATENT_DIM).to(device)

        # model_WGAN = nn.DataParallel(model_WGAN) # This always moves the model to GPU


        #weightsGAN = 'WGAN_model/netG_itrs_10999.pth'
        #model_WGAN.load_state_dict(torch.load(weightsGAN, map_location=torch.device(device)))
        #################


        tlist = []
        for exp in range(1,nexp+1):

            noise = torch.empty([bsize, LATENT_DIM, 1,1,1], device=device)
            energy = torch.empty([bsize, 1,1,1,1], device=device)

            if gpu:
                # noise = torch.cuda.FloatTensor(bsize, LATENT_DIM, 1,1,1)
                # energy = torch.cuda.FloatTensor(bsize, 1,1,1,1)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()


                a = getFakeImagesGAN(model_WGAN, N, emax, emin, bsize, noise, energy)

                #print (nvgpu.gpu_info()[0]['mem_used'])

                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_ms = start_event.elapsed_time(end_event)
                tlist.append(elapsed_time_ms/N)
                #print("WGAN. Batch size is {1}. ms/shower: {0}. Memory GPU total: {2}, cached : {3}, alllocated:  {4}"
                #                .format(elapsed_time_ms/N, bsize, t, c, alo))


                print("WGAN. Batch size is {1}. ms/shower: {0}. Memory usage (Mb): {2}".format(elapsed_time_ms/N, bsize, nvgpu.gpu_info()[0]['mem_used']))
                # time.sleep(1.0)

            else:   ## CPU

                start = time.perf_counter()
                a = getFakeImagesGAN(model_WGAN, N, emax, emin, bsize, noise, energy)
                end = time.perf_counter()
                diff = (end-start) * 1000
                tlist.append(diff/N)
                print("WGAN. Batch size is {1}. ms/shower: {0}".format(diff/N, bsize))


        tlist.pop(0)
        time_all = np.asarray(tlist)
        #print (time_all)
        print("WGAN, batchsize : {0}. Experiment conducted {1}. ms/shower: {2} +- {3}".format(bsize, nexp, time_all.mean(),time_all.std() ))


    elif model_name == "Bib-AE":
        ### MODEL BIB-AE ###
        args = {
            'E_cond' : True,
            'latent' : 512
        }

        model = VAE_Models.BiBAE_F_3D_LayerNorm_SmallLatent_Fast(args, ngf=8, device=device, z_rand=512-24,
                                                        z_enc=24).to(device)

        # model = nn.DataParallel(model)


        model_P = VAE_Models.PostProcess_EScale_EcondV2(bias=True, out_funct='none').to(device)
        # model_P = nn.DataParallel(model_P)

        tlist = []
        for exp in range(1,nexp+1):

            if gpu:

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                a = getFakeImagesVAE_ENR_PostProcess(model, model_P, N, emax, emin, bsize, 512, device)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_ms = start_event.elapsed_time(end_event)
                tlist.append(elapsed_time_ms/N)
                print("Bib-AE. Batch size is {1}. ms/shower: {0}".format(elapsed_time_ms/N, bsize))
                # time.sleep(3.0)

            else:   ## CPU
                start = time.perf_counter()
                getFakeImagesVAE_ENR_PostProcess(model, model_P, N, emax, emin, bsize, 512, device)
                end = time.perf_counter()
                diff = (end-start) * 1000
                tlist.append(diff/N)
                print("Bib-AE. Batch size is {1}. ms/shower: {0}".format(diff/N, bsize))

        tlist.pop(0)
        time_all = np.asarray(tlist)
        print("Bib-AE, batchsize : {0}. Experiment conducted {1}. ms/shower: {2} +- {3}".format(bsize, nexp, time_all.mean(),time_all.std() ))

if __name__ == "__main__":
    main()
