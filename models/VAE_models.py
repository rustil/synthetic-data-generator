import numpy as np
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch import autograd


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(4500, 900)
        self.fc2 = nn.Linear(900, 400)
        self.fc31 = nn.Linear(400, 40)
        self.fc32 = nn.Linear(400, 40)
        self.fc4 = nn.Linear(40, 400)
        self.fc5 = nn.Linear(400, 900)
        self.fc6 = nn.Linear(900, 4500)


    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return F.relu(self.fc6(h5))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 4500))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    
    
    
class VAE_3Dconv_DIRENR(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x5 images
    """
    def __init__(self, args, nc=1, ngf=8, z=100):
        super(VAE_3Dconv_DIRENR, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z = z
        self.args = args

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(3,4,4), stride=(1,2,2),
                               padding=(1,2,2), bias=False, padding_mode='zeros')
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(1,3,3), stride=(1,1,1),
                               padding=(0,1,1), bias=False, padding_mode='zeros')
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(3,4,4), stride=(1,2,2),
                               padding=(1,2,2), bias=False, padding_mode='zeros')
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(1,3,3), stride=(1,1,1),
                               padding=(0,1,1), bias=False, padding_mode='zeros')

     
        self.fc1 = nn.Linear(9*9*5*ngf*8+1, ngf*100, bias=True)
        self.fc2 = nn.Linear(ngf*100, 400, bias=True)
        
        self.fc31 = nn.Linear(400, z, bias=True)
        self.fc32 = nn.Linear(400, z, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z+1, 400, bias=True)
        self.cond2 = torch.nn.Linear(400, 10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,3,3), padding=(0,1,1), bias=False)
        self.deconv2 = torch.nn.ConvTranspose3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*8, ngf, kernel_size=(3,4,4), stride=(3,4,4), padding=(0,0,0), bias=False)
        self.conv1 = torch.nn.Conv3d(ngf, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.conv2 = torch.nn.Conv3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.conv3 = torch.nn.Conv3d(ngf*8, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.conv4 = torch.nn.Conv3d(ngf*4, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        
    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.enconv1(x.view(-1,1,5,30,30)), 0.2, inplace=True)
        x = F.leaky_relu(self.enconv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.enconv3(x), 0.2, inplace=True)
        x = F.leaky_relu(self.enconv4(x), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.2, inplace=True)
        return self.fc31(x), self.fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,1,10,10)        

        ## apply series of deconv2d and batch-norm

        x = F.leaky_relu(self.deconv1(x, output_size=[x.size(0), 1, 3, 30, 30]), 0.2, inplace=True) #
        x = F.leaky_relu(self.deconv2(x, output_size=[x.size(0), 1, 7, 60, 60]), 0.2, inplace=True) #
        x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.conv0(x), 0.2, inplace=True)

        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)

        return x

   
    def forward(self, x, E_true):
        mu, logvar = self.encode(x, E_true)
        if not self.args["E_cond"]:
            E_true = torch.randn_like(E_true)
        z = self.reparameterize(mu, logvar)
        return self.decode(torch.cat((z,E_true), 1)), mu, logvar, E_true 
    
    

class VAE_3Dconv_DIRENR(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x5 images
    """
    def __init__(self, args, nc=1, ngf=8, z=100):
        super(VAE_3Dconv_DIRENR, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z = z
        self.args = args

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(3,4,4), stride=(1,2,2),
                               padding=(1,2,2), bias=False, padding_mode='zeros')
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(1,3,3), stride=(1,1,1),
                               padding=(0,1,1), bias=False, padding_mode='zeros')
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(3,4,4), stride=(1,2,2),
                               padding=(1,2,2), bias=False, padding_mode='zeros')
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(1,3,3), stride=(1,1,1),
                               padding=(0,1,1), bias=False, padding_mode='zeros')

     
        self.fc1 = nn.Linear(9*9*5*ngf*8+1, ngf*100, bias=True)
        self.fc2 = nn.Linear(ngf*100, 400, bias=True)
        
        self.fc31 = nn.Linear(400, z, bias=True)
        self.fc32 = nn.Linear(400, z, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z+1, 400, bias=True)
        self.cond2 = torch.nn.Linear(400, 10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,3,3), padding=(0,1,1), bias=False)
        self.deconv2 = torch.nn.ConvTranspose3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*8, ngf, kernel_size=(3,4,4), stride=(3,4,4), padding=(0,0,0), bias=False)
        self.conv1 = torch.nn.Conv3d(ngf, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.conv2 = torch.nn.Conv3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.conv3 = torch.nn.Conv3d(ngf*8, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.conv4 = torch.nn.Conv3d(ngf*4, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        
    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.enconv1(x.view(-1,1,5,30,30)), 0.2, inplace=True)
        x = F.leaky_relu(self.enconv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.enconv3(x), 0.2, inplace=True)
        x = F.leaky_relu(self.enconv4(x), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.2, inplace=True)
        return self.fc31(x), self.fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,1,10,10)        

        ## apply series of deconv2d and batch-norm

        x = F.leaky_relu(self.deconv1(x, output_size=[x.size(0), 1, 3, 30, 30]), 0.2, inplace=True) #
        x = F.leaky_relu(self.deconv2(x, output_size=[x.size(0), 1, 7, 60, 60]), 0.2, inplace=True) #
        x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.conv0(x), 0.2, inplace=True)

        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)

        return x

   
    def forward(self, x, E_true):
        mu, logvar = self.encode(x, E_true)
        if not self.args["E_cond"]:
            E_true = torch.randn_like(E_true)
        z = self.reparameterize(mu, logvar)
        return self.decode(torch.cat((z,E_true), 1)), mu, logvar, E_true 
    

class VAE_3Dconv_DIRENR_v2(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x5 images
    faster version
    """
    def __init__(self, args, nc=1, ngf=8, z=100):
        super(VAE_3Dconv_DIRENR_v2, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z = z
        self.args = args

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(3,4,4), stride=(1,2,2),
                               padding=(1,2,2), bias=False, padding_mode='zeros')
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(1,3,3), stride=(1,1,1),
                               padding=(0,1,1), bias=False, padding_mode='zeros')
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(3,4,4), stride=(1,2,2),
                               padding=(1,2,2), bias=False, padding_mode='zeros')
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(1,3,3), stride=(1,1,1),
                               padding=(0,1,1), bias=False, padding_mode='zeros')

     
        self.fc1 = nn.Linear(9*9*5*ngf*8+1, ngf*100, bias=True)
        self.fc2 = nn.Linear(ngf*100, 400, bias=True)
        
        self.fc31 = nn.Linear(400, z, bias=True)
        self.fc32 = nn.Linear(400, z, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z+1, 400, bias=True)
        self.cond2 = torch.nn.Linear(400, 10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,3,3), padding=(0,1,1), bias=False)
        self.deconv2 = torch.nn.ConvTranspose3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*4, ngf, kernel_size=(3,2,2), stride=(1,2,2), padding=(0,0,0), bias=False)
        self.conv1 = torch.nn.Conv3d(ngf, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.conv2 = torch.nn.Conv3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.conv3 = torch.nn.Conv3d(ngf*8, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.conv4 = torch.nn.Conv3d(ngf*4, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        
    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.enconv1(x.view(-1,1,5,30,30)), 0.2, inplace=True)
        x = F.leaky_relu(self.enconv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.enconv3(x), 0.2, inplace=True)
        x = F.leaky_relu(self.enconv4(x), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.2, inplace=True)
        return self.fc31(x), self.fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,1,10,10)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.deconv1(x, output_size=[x.size(0), 1, 3, 30, 30]), 0.2, inplace=True) #
        x = F.leaky_relu(self.deconv2(x, output_size=[x.size(0), 1, 7, 60, 60]), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.conv0(x), 0.2, inplace=True)

        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)

        return x

   
    def forward(self, x, E_true):
        mu, logvar = self.encode(x, E_true)
        if not self.args["E_cond"]:
            E_true = torch.randn_like(E_true)
        z = self.reparameterize(mu, logvar)
        return self.decode(torch.cat((z,E_true), 1)), mu, logvar, E_true 
    
    

    
class VAE_F_3Dconv_DIRENR(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, nc=1, ngf=8, z=200):
        super(VAE_F_3Dconv_DIRENR, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        self.z = z
        self.args = args

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.enconv4 = nn.Conv2d(in_channels=ngf*4*5, out_channels=ngf*8*5, kernel_size=(3,3), stride=(1,1),
                               padding=(1,1), bias=False, padding_mode='zeros')

     
        self.fc1 = nn.Linear(5*5*5*ngf*8+1, ngf*200, bias=True)
        self.fc2 = nn.Linear(ngf*200, 800, bias=True)
        
        self.fc31 = nn.Linear(800, z, bias=True)
        self.fc32 = nn.Linear(800, z, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z+1, 800, bias=True)
        self.cond2 = nn.Linear(800, ngf*200, bias=True)
        self.cond3 = torch.nn.Linear(ngf*200, 10*10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(3,3,3), padding=(1,1,1), bias=False)
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), bias=False)
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        
    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.enconv1(x.view(-1,1,30,30,30)), 0.2, inplace=True)
        x = F.leaky_relu(self.enconv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.enconv3(x), 0.2, inplace=True)
        x = F.leaky_relu(self.enconv4(x.view(-1,self.ngf*4*5, 5, 5)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)), E_true), 1)
                       
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.2, inplace=True)
        return self.fc31(x), self.fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,10,10,10)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.deconv1(x, output_size=[x.size(0), 1, 30, 30, 30]), 0.2, inplace=True) #
        x = F.leaky_relu(self.deconv2(x, output_size=[x.size(0), 1, 60, 60, 60]), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x

   
    #def forward(self, x, E_true):
    #    print(x.size())
    #    mu, logvar = self.encode(x, E_true)
    #    if not self.args["E_cond"]:
    #        E_true = torch.randn_like(E_true)
    #    z = self.reparameterize(mu, logvar)
    #    return self.decode(torch.cat((z,E_true), 1)), mu, logvar, E_true 
    def forward(self, x, E_true, mu, logvar, mode):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            return mu, logvar
        elif mode == 'decode':
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), E_true 
    
    

    
class VAE_F_3Dconv_DIRENR_VarLatent(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, nc=1, ngf=8):
        super(VAE_F_3Dconv_DIRENR_VarLatent, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z = args["latent"]
        self.args = args

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.enconv4 = nn.Conv2d(in_channels=ngf*4*5, out_channels=ngf*8*5, kernel_size=(3,3), stride=(1,1),
                               padding=(1,1), bias=False, padding_mode='zeros')

     
        self.fc1 = nn.Linear(5*5*5*ngf*8+1, ngf*500, bias=True)
        self.fc2 = nn.Linear(ngf*500, int(self.z*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z*1.5), self.z, bias=True)
        self.fc32 = nn.Linear(int(self.z*1.5), self.z, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z+1, int(self.z*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z*1.5), ngf*500, bias=True)
        self.cond3 = torch.nn.Linear(ngf*500, 10*10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(3,3,3), padding=(1,1,1), bias=False)
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), bias=False)
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        
    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.enconv1(x.view(-1,1,30,30,30)), 0.2, inplace=True)
        x = F.leaky_relu(self.enconv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.enconv3(x), 0.2, inplace=True)
        x = F.leaky_relu(self.enconv4(x.view(-1,self.ngf*4*5, 5, 5)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)), E_true), 1)
                       
        x = F.leaky_relu(self.fc1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.fc2(x), 0.2, inplace=True)
        return self.fc31(x), self.fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,10,10,10)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.deconv1(x, output_size=[x.size(0), 1, 30, 30, 30]), 0.2, inplace=True) #
        x = F.leaky_relu(self.deconv2(x, output_size=[x.size(0), 1, 60, 60, 60]), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.conv0(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv2(x), 0.2, inplace=True)
        x = F.leaky_relu(self.conv3(x), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x

   
    def forward(self, x, E_true, mu, logvar, mode):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            return mu, logvar
        elif mode == 'decode':
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), E_true 
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(1800, 800)
        self.fc2 = nn.Linear(800, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, 20)
        self.fc5 = nn.Linear(20, 1)
  
    def forward(self, input):
        d1 = F.relu(self.fc1(input))
        d2 = F.relu(self.fc2(d1))
        d3 = F.relu(self.fc3(d2))
        d4 = F.relu(self.fc4(d3))
        output = (self.fc5(d4))
        return output
    
    
class Discriminator_Conv(nn.Module):
    def __init__(self):
        super(Discriminator_Conv, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=(5,5), stride=1, padding=2, bias=True, padding_mode='zeros'),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=(5,5), stride=2, padding=2, bias=True, padding_mode='zeros'),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=(5,5), stride=2, padding=2, bias=True, padding_mode='zeros'),
            nn.LeakyReLU(),
        )

        self.main = main
        self.linear1 = nn.Linear(64*8*8, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 64*8*8)
        output = F.relu(self.linear1(output))
        output = self.linear2(output)
        return output
    
    
    
    
    
    
    
    
    
    
class Discriminator_3DConv_DIRENR(nn.Module):
    def __init__(self):
        super(Discriminator_3DConv_DIRENR, self).__init__()
        conv_structa = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1,5,5), stride=(1,1,1), padding=(0,2,2), bias=False, padding_mode='zeros'),
            nn.LeakyReLU(),
            nn.Conv3d(32, 64, kernel_size=(3,5,5), stride=(1,2,2), padding=(0,2,2), bias=False, padding_mode='zeros'),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, kernel_size=(3,5,5), stride=(1,2,2), padding=(0,2,2), bias=False, padding_mode='zeros'),
            nn.LeakyReLU(),
        )
        
        conv_structb= nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1,5,5), stride=(1,1,1), padding=(0,2,2), bias=False, padding_mode='zeros'),
            nn.LeakyReLU(),
            nn.Conv3d(32, 64, kernel_size=(3,5,5), stride=(1,2,2), padding=(0,2,2), bias=False, padding_mode='zeros'),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, kernel_size=(3,5,5), stride=(1,2,2), padding=(0,2,2), bias=False, padding_mode='zeros'),
            nn.LeakyReLU(),
        )

        self.conv_structa = conv_structa
        self.conv_structb = conv_structb
        self.linear1a = nn.Linear(64*8*8, 200)
        self.linear1b = nn.Linear(64*8*8, 200)
        self.linear1e = nn.Linear(1, 10)
        self.linear2 = nn.Linear(410, 100)
        self.linear3 = nn.Linear(100, 1)

    def forward(self, img, E_true):
        x1 = self.conv_structa(img[:,0:1,:,:])
        x2 = self.conv_structb(img[:,1:2,:,:])

        x1 = F.leaky_relu(self.linear1a(x1.view(-1, 64*8*8)), 0.2, inplace=True)
        x2 = F.leaky_relu(self.linear1b(x2.view(-1, 64*8*8)), 0.2, inplace=True)
        
        x3 = F.leaky_relu(self.linear1e(E_true), 0.2, inplace=True)

        x1 = torch.cat((x1, x2, x3), 1)
        x1 = F.leaky_relu(self.linear2(x1))
        x1 = F.leaky_relu(self.linear3(x1))

        return x1
    
class Discriminator_3DConv_DIRENR_v2(nn.Module):
    """
    features a mix of 2 and 3d convoutions to speed up trianing
    """
    def __init__(self, n_chn=32):
        super(Discriminator_3DConv_DIRENR_v2, self).__init__()
        self.n_chn =n_chn
        self.c1a = nn.Conv3d(1, n_chn, kernel_size=(3,5,5), stride=(1,1,1), padding=(0,2,2), bias=False, padding_mode='zeros')
        self.c1b = nn.Conv2d(n_chn*3, n_chn*2, kernel_size=(5,5), stride=(2,2), padding=(2,2), bias=False, padding_mode='zeros')
        self.c1c = nn.Conv2d(n_chn*2, n_chn*2, kernel_size=(5,5), stride=(2,2), padding=(2,2), bias=False, padding_mode='zeros')
       
        self.c2a = nn.Conv3d(1, n_chn, kernel_size=(3,5,5), stride=(1,1,1), padding=(0,2,2), bias=False, padding_mode='zeros')
        self.c2b = nn.Conv2d(n_chn*3, n_chn*2, kernel_size=(5,5), stride=(2,2), padding=(2,2), bias=False, padding_mode='zeros')
        self.c2c = nn.Conv2d(n_chn*2, n_chn*2, kernel_size=(5,5), stride=(2,2), padding=(2,2), bias=False, padding_mode='zeros')

        self.linear1a = nn.Linear(64*8*8, 200)
        self.linear1b = nn.Linear(64*8*8, 200)
        self.linear1e = nn.Linear(1, 10)
        self.linear2 = nn.Linear(410, 100)
        self.linear3 = nn.Linear(100, 1)

    def forward(self, img, E_true):
        x1 =  F.leaky_relu(self.c1a(img[:,0:1,:,:,:]), inplace=True)
        x1 =  F.leaky_relu(self.c1b(x1.view(-1,self.n_chn*3, 30, 30)), inplace=True)
        x1 =  F.leaky_relu(self.c1c(x1), inplace=True)

        x2 =  F.leaky_relu(self.c2a(img[:,1:2,:,:,:]), inplace=True)
        x2 =  F.leaky_relu(self.c2b(x2.view(-1,self.n_chn*3, 30, 30)), inplace=True)
        x2 =  F.leaky_relu(self.c2c(x2), inplace=True)
             
        x1 = F.leaky_relu(self.linear1a(x1.view(-1, 64*8*8)), 0.2, inplace=True)
        x2 = F.leaky_relu(self.linear1b(x2.view(-1, 64*8*8)), 0.2, inplace=True)
        
        x3 = F.leaky_relu(self.linear1e(E_true), 0.2, inplace=True)

        x1 = torch.cat((x1, x2, x3), 1)
        x1 = F.leaky_relu(self.linear2(x1))
        x1 = F.leaky_relu(self.linear3(x1))

        return x1

    
    
    
    
class Discriminator_F_3DConv_DIRENR(nn.Module):
    """
    features a mix of 2 and 3d convoutions to speed up trianing
    """
    def __init__(self, n_chn=32):
        super(Discriminator_F_3DConv_DIRENR, self).__init__()
        self.n_chn =n_chn
        self.c1a = nn.Conv3d(1, n_chn, kernel_size=(5,5,5), stride=(1,1,1), padding=(2,2,2), bias=False, padding_mode='zeros')
        self.c1b = nn.Conv3d(n_chn, n_chn, kernel_size=(5,5,5), stride=(3,3,3), padding=(2,2,2), bias=False, padding_mode='zeros')
        self.c1c = nn.Conv2d(n_chn*10, n_chn*10, kernel_size=(5,5), stride=(1,1), padding=(2,2), bias=False, padding_mode='zeros')
        self.c1d = nn.Conv2d(n_chn*10, n_chn*5, kernel_size=(5,5), stride=(2,2), padding=(2,2), bias=False, padding_mode='zeros')
       
        self.c2a = nn.Conv3d(1, n_chn, kernel_size=(5,5,5), stride=(1,1,1), padding=(2,2,2), bias=False, padding_mode='zeros')
        self.c2b = nn.Conv3d(n_chn, n_chn, kernel_size=(5,5,5), stride=(3,3,3), padding=(2,2,2), bias=False, padding_mode='zeros')
        self.c2c = nn.Conv2d(n_chn*10, n_chn*10, kernel_size=(5,5), stride=(1,1), padding=(2,2), bias=False, padding_mode='zeros')
        self.c2d = nn.Conv2d(n_chn*10, n_chn*5, kernel_size=(5,5), stride=(2,2), padding=(2,2), bias=False, padding_mode='zeros')

        self.linear1a = nn.Linear(n_chn*5*5*5, 200)
        self.linear1b = nn.Linear(n_chn*5*5*5, 200)
        self.linear1e = nn.Linear(1, 10)
        self.linear2 = nn.Linear(410, 100)
        self.linear3 = nn.Linear(100, 1)

    def forward(self, img, E_true):
        x1 =  F.leaky_relu(self.c1a(img[:,0:1,:,:,:]), inplace=True)
        x1 =  F.leaky_relu(self.c1b(x1), inplace=True)
        x1 =  F.leaky_relu(self.c1c(x1.view(-1,self.n_chn*10, 10, 10)), inplace=True)
        x1 =  F.leaky_relu(self.c1d(x1), inplace=True)

        x2 =  F.leaky_relu(self.c2a(img[:,1:2,:,:,:]), inplace=True)
        x2 =  F.leaky_relu(self.c2b(x2), inplace=True)
        x2 =  F.leaky_relu(self.c2c(x2.view(-1,self.n_chn*10, 10, 10)), inplace=True)
        x2 =  F.leaky_relu(self.c2d(x2), inplace=True)
             
        x1 = F.leaky_relu(self.linear1a(x1.view(-1, self.n_chn*5*5*5)), 0.2, inplace=True)
        x2 = F.leaky_relu(self.linear1b(x2.view(-1, self.n_chn*5*5*5)), 0.2, inplace=True)
        
        x3 = F.leaky_relu(self.linear1e(E_true), 0.2, inplace=True)

        x1 = torch.cat((x1, x2, x3), 1)
        x1 = F.leaky_relu(self.linear2(x1))
        x1 = F.leaky_relu(self.linear3(x1))

        return x1
    
    
class Discriminator_F_3DConv_DIRENR_v2(nn.Module):
    """
    features a mix of 2 and 3d convoutions to speed up trianing
    """
    def __init__(self, n_chn=32):
        super(Discriminator_F_3DConv_DIRENR_v2, self).__init__()
        self.n_chn =n_chn
        self.c1a = nn.Conv3d(1, n_chn, kernel_size=(5,5,5), stride=(1,1,1), padding=(2,2,2), bias=False, padding_mode='zeros')
        self.c1b = nn.Conv3d(n_chn, n_chn*2, kernel_size=(5,5,5), stride=(3,3,3), padding=(2,2,2), bias=False, padding_mode='zeros')
        self.c1c = nn.Conv2d(n_chn*20, n_chn*10, kernel_size=(5,5), stride=(1,1), padding=(2,2), bias=False, padding_mode='zeros')
        self.c1d = nn.Conv2d(n_chn*10, n_chn*5, kernel_size=(5,5), stride=(2,2), padding=(2,2), bias=False, padding_mode='zeros')
       
        self.c2a = nn.Conv3d(1, n_chn, kernel_size=(5,5,5), stride=(1,1,1), padding=(2,2,2), bias=False, padding_mode='zeros')
        self.c2b = nn.Conv3d(n_chn, n_chn*2, kernel_size=(5,5,5), stride=(3,3,3), padding=(2,2,2), bias=False, padding_mode='zeros')
        self.c2c = nn.Conv2d(n_chn*20, n_chn*10, kernel_size=(5,5), stride=(1,1), padding=(2,2), bias=False, padding_mode='zeros')
        self.c2d = nn.Conv2d(n_chn*10, n_chn*5, kernel_size=(5,5), stride=(2,2), padding=(2,2), bias=False, padding_mode='zeros')

        self.linear1a = nn.Linear(n_chn*5*5*5, n_chn*25)
        self.linear1b = nn.Linear(n_chn*5*5*5, n_chn*25)
        self.linear1e = nn.Linear(1, n_chn)
        self.linear2 = nn.Linear(n_chn*51, n_chn*25)
        self.linear3 = nn.Linear(n_chn*25, n_chn*10)
        self.linear4 = nn.Linear(n_chn*10, 1)

    def forward(self, img, E_true):
        x1 =  F.leaky_relu(self.c1a(img[:,0:1,:,:,:]), inplace=True)
        x1 =  F.leaky_relu(self.c1b(x1), inplace=True)
        x1 =  F.leaky_relu(self.c1c(x1.view(-1,self.n_chn*20, 10, 10)), inplace=True)
        x1 =  F.leaky_relu(self.c1d(x1), inplace=True)

        x2 =  F.leaky_relu(self.c2a(img[:,1:2,:,:,:]), inplace=True)
        x2 =  F.leaky_relu(self.c2b(x2), inplace=True)
        x2 =  F.leaky_relu(self.c2c(x2.view(-1,self.n_chn*20, 10, 10)), inplace=True)
        x2 =  F.leaky_relu(self.c2d(x2), inplace=True)
             
        x1 = F.leaky_relu(self.linear1a(x1.view(-1, self.n_chn*5*5*5)), 0.2, inplace=True)
        x2 = F.leaky_relu(self.linear1b(x2.view(-1, self.n_chn*5*5*5)), 0.2, inplace=True)
        
        x3 = F.leaky_relu(self.linear1e(E_true), 0.2, inplace=True)

        x1 = torch.cat((x1, x2, x3), 1)
        x1 = F.leaky_relu(self.linear2(x1), inplace=True)
        x1 = F.leaky_relu(self.linear3(x1), inplace=True)
        x1 = self.linear4(x1)
        return x1
    
    
class Discriminator_F_Conv_DIRENR_Diff(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128):
        super(Discriminator_F_Conv_DIRENR_Diff, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
       
        self.conv1 = torch.nn.Conv3d(1, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1 = torch.nn.LayerNorm([14,14,14])
        self.conv2 = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn2 = torch.nn.LayerNorm([6,6,6])
        self.conv3 = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=1, padding=0, bias=False)
 
        #self.fc = torch.nn.Linear(5 * 4 * 4, 1)
        self.fc1a = torch.nn.Linear(30*30*30, int(ndf/2))
    
        self.fc1b = torch.nn.Linear(ndf * 4 * 4 * 4, int(ndf/2))
        self.fc2 = torch.nn.Linear(int(ndf/2)*2+1, ndf*2)
        self.fc3 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc4 = torch.nn.Linear(ndf*2, 1)


    def forward(self, img, E_true):
        imga = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        imgb = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)
        
        #img = c1a - c1b
        img = imgb - imga
        
        #energy = F.leaky_relu(self.cond1(E_true), 0.2)
        #energy = self.cond2(energy)
        
        #energy = energy.view(-1, 1, self.isize, self.isize)
        
        #x = torch.cat((img, energy), 1)
        x = F.leaky_relu(self.bn1(self.conv1(imgb)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(-1, self.ndf * 4 * 4 * 4)
        x = F.leaky_relu(self.fc1b(x), 0.2)

        x = torch.cat((x, F.leaky_relu(self.fc1a(img.view(-1, 30*30*30))) , E_true), 1)

        #x = F.leaky_relu(self.fc1a(img.view(-1, 900)))
        #x = F.leaky_relu(self.fc(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        output_wgan = self.fc4(x)
        
        output_wgan = output_wgan.view(-1) ### flattens

        return output_wgan
    
    
class Discriminator_F_Conv_DIRENR_Diff_v3(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128):
        super(Discriminator_F_Conv_DIRENR_Diff_v3, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc

        
        self.conv1b = torch.nn.Conv3d(1, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1b = torch.nn.LayerNorm([14,14,14])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn2b = torch.nn.LayerNorm([6,6,6])
        self.conv3b = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=1, padding=0, bias=False)


        self.conv1c = torch.nn.Conv3d(1, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1c = torch.nn.LayerNorm([14,14,14])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn2c = torch.nn.LayerNorm([6,6,6])
        self.conv3c = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=1, padding=0, bias=False)

 
        #self.fc = torch.nn.Linear(5 * 4 * 4, 1)
        self.fc1a = torch.nn.Linear(30*30*30, int(ndf/2)) 
        self.fc1b = torch.nn.Linear(ndf * 4 * 4 * 4, int(ndf/2))
        self.fc1c = torch.nn.Linear(ndf * 4 * 4 * 4, int(ndf/2))
        self.fc1e = torch.nn.Linear(1, int(ndf/2))
        self.fc2 = torch.nn.Linear(int(ndf/2)*4, ndf*2)
        self.fc3 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc4 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc5 = torch.nn.Linear(ndf*2, 1)


    def forward(self, img, E_true):
        imga = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        imgb = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)

        img = imgb - imga
        
        xb = F.leaky_relu(self.bn1b(self.conv1b(imgb)), 0.2)
        xb = F.leaky_relu(self.bn2b(self.conv2b(xb)), 0.2)
        xb = F.leaky_relu(self.conv3b(xb), 0.2)
        xb = xb.view(-1, self.ndf * 4 * 4 * 4)
        xb = F.leaky_relu(self.fc1b(xb), 0.2)        
        
        xc = F.leaky_relu(self.bn1c(self.conv1c(torch.log(imgb+1.0))), 0.2)
        xc = F.leaky_relu(self.bn2c(self.conv2c(xc)), 0.2)
        xc = F.leaky_relu(self.conv3c(xc), 0.2)
        xc = xc.view(-1, self.ndf * 4 * 4 * 4)
        xc = F.leaky_relu(self.fc1c(xc), 0.2)
        
        xb = torch.cat((xb, xc, F.leaky_relu(self.fc1a(img.view(-1, 30*30*30))) , F.leaky_relu(self.fc1e(E_true), 0.2)), 1)

        xb = F.leaky_relu(self.fc2(xb), 0.2)
        xb = F.leaky_relu(self.fc3(xb), 0.2)
        xb = F.leaky_relu(self.fc4(xb), 0.2)
        xb = self.fc5(xb)

        return xb.view(-1) ### flattens
    
class Discriminator_F_Conv_DIRENR_Diff_v3LinOut(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128):
        super(Discriminator_F_Conv_DIRENR_Diff_v3LinOut, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc

        
        self.conv1b = torch.nn.Conv3d(1, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1b = torch.nn.LayerNorm([14,14,14])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn2b = torch.nn.LayerNorm([6,6,6])
        self.conv3b = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=1, padding=0, bias=False)


        self.conv1c = torch.nn.Conv3d(1, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1c = torch.nn.LayerNorm([14,14,14])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn2c = torch.nn.LayerNorm([6,6,6])
        self.conv3c = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=1, padding=0, bias=False)

 
        #self.fc = torch.nn.Linear(5 * 4 * 4, 1)
        self.fc1a = torch.nn.Linear(30*30*30, int(ndf/2)) 
        self.fc1b = torch.nn.Linear(ndf * 4 * 4 * 4, int(ndf/2))
        self.fc1c = torch.nn.Linear(ndf * 4 * 4 * 4, int(ndf/2))
        self.fc1e = torch.nn.Linear(1, int(ndf/2))
        self.fc2 = torch.nn.Linear(int(ndf/2)*4, ndf*2)
        self.fc3 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc4 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc5 = torch.nn.Linear(ndf*2, 1)


    def forward(self, img, E_true):
        imga = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        imgb = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)

        img = imgb - imga
        
        xb = F.leaky_relu(self.bn1b(self.conv1b(imgb)), 0.2)
        xb = F.leaky_relu(self.bn2b(self.conv2b(xb)), 0.2)
        xb = F.leaky_relu(self.conv3b(xb), 0.2)
        xb = xb.view(-1, self.ndf * 4 * 4 * 4)
        xb = F.leaky_relu(self.fc1b(xb), 0.2)        
        
        imgb_relu = F.relu(imgb)
        
        xc = F.leaky_relu(self.bn1c(self.conv1c(torch.log(imgb_relu+1.0))), 0.2)
        xc = F.leaky_relu(self.bn2c(self.conv2c(xc)), 0.2)
        xc = F.leaky_relu(self.conv3c(xc), 0.2)
        xc = xc.view(-1, self.ndf * 4 * 4 * 4)
        xc = F.leaky_relu(self.fc1c(xc), 0.2)
        
        xb = torch.cat((xb, xc, F.leaky_relu(self.fc1a(img.view(-1, 30*30*30))) , F.leaky_relu(self.fc1e(E_true), 0.2)), 1)

        xb = F.leaky_relu(self.fc2(xb), 0.2)
        xb = F.leaky_relu(self.fc3(xb), 0.2)
        xb = F.leaky_relu(self.fc4(xb), 0.2)
        xb = self.fc5(xb)

        return xb.view(-1) ### flattens
    
    
    
    
class Latent_Critic(nn.Module):
    def __init__(self, ):
        super(Latent_Critic, self).__init__()
        self.linear1 = nn.Linear(1, 50)
        self.linear2 = nn.Linear(50, 100)        
        self.linear3 = nn.Linear(100, 50)
        self.linear4 = nn.Linear(50, 1)

    def forward(self, x):      
        x = F.leaky_relu(self.linear1(x.view(-1,1)), inplace=True)
        x = F.leaky_relu(self.linear2(x), inplace=True)
        x = F.leaky_relu(self.linear3(x), inplace=True)
        return self.linear4(x)
    

    
class Latent_Critic_Broad(nn.Module):
    def __init__(self, args):
        super(Latent_Critic_Broad, self).__init__()
        self.z = args["latent"]

        self.linear1 = nn.Linear(self.z, int(self.z*1.5))
        self.linear2 = nn.Linear(int(self.z*1.5), int(self.z*2))        
        self.linear3 = nn.Linear(int(self.z*2), 200)
        self.linear4 = nn.Linear(200, 100)
        self.linear5 = nn.Linear(100, 1)

    def forward(self, x):      
        x = F.leaky_relu(self.linear1(x), inplace=True)
        x = F.leaky_relu(self.linear2(x), inplace=True)
        x = F.leaky_relu(self.linear3(x), inplace=True)
        x = F.leaky_relu(self.linear4(x), inplace=True)
        return self.linear5(x)
    
    
class BiBAE_F_3D_Norm(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, nc=1, ngf=8):
        super(BiBAE_F_3D_Norm, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z = args["latent"]
        self.args = args

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.BatchNorm3d(ngf)
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.BatchNorm3d(ngf*2)
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.BatchNorm3d(ngf*4)
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.BatchNorm3d(ngf*8)

     
        self.fc1 = nn.Linear(5*5*5*ngf*8+1, ngf*500, bias=True)
        self.fc2 = nn.Linear(ngf*500, int(self.z*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z*1.5), self.z, bias=True)
        self.fc32 = nn.Linear(int(self.z*1.5), self.z, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z+1, int(self.z*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z*1.5), ngf*500, bias=True)
        self.cond3 = torch.nn.Linear(ngf*500, 10*10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(3,3,3), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.BatchNorm3d(ngf)
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.BatchNorm3d(ngf*2)

        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), bias=False)
        self.bnco0 = torch.nn.BatchNorm3d(ngf)
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.BatchNorm3d(ngf*2)
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.BatchNorm3d(ngf*4)
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.BatchNorm3d(ngf*2)
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        self.dr03 = nn.Dropout(p=0.3, inplace=False)
        self.dr05 = nn.Dropout(p=0.5, inplace=False)

    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,30,30,30))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu(self.dr03(self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.dr03(self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return self.fc31(x), self.fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu(self.dr03(self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu(self.dr03(self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.dr03(self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,10,10,10)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 30, 30, 30])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 60, 60, 60])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z

        
        
class BiBAE_F_3D_Norm_v2(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, nc=1, ngf=8):
        super(BiBAE_F_3D_Norm_v2, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z = args["latent"]
        self.args = args

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.BatchNorm3d(ngf)
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.BatchNorm3d(ngf*2)
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.BatchNorm3d(ngf*4)
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.BatchNorm3d(ngf*8)

     
        #self.fc1 = nn.Linear(5*5*5*ngf*8+1, ngf*500, bias=True)
        #self.fc2 = nn.Linear(ngf*500, int(self.z*1.5), bias=True)
        self.fc1 = nn.Linear(5*5*5*ngf*8+1, ngf*100, bias=True)
        self.fc2 = nn.Linear(ngf*100, int(self.z*1.5), bias=True)

        
        self.fc31 = nn.Linear(int(self.z*1.5), self.z, bias=True)
        self.fc32 = nn.Linear(int(self.z*1.5), self.z, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z+1, int(self.z*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z*1.5), ngf*500, bias=True)
        self.cond3 = torch.nn.Linear(ngf*500, 10*10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(3,3,3), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.BatchNorm3d(ngf)
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.BatchNorm3d(ngf*2)

        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), bias=False)
        self.bnco0 = torch.nn.BatchNorm3d(ngf)
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.BatchNorm3d(ngf*2)
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.BatchNorm3d(ngf*4)
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.BatchNorm3d(ngf*2)
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        #self.dr03 = nn.Dropout(p=0.3, inplace=False)
        #self.dr05 = nn.Dropout(p=0.5, inplace=False)

    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,30,30,30))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return self.fc31(x), self.fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,10,10,10)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 30, 30, 30])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 60, 60, 60])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z

        
        

class BiBAE_F_3D_LayerNorm(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, nc=1, ngf=8):
        super(BiBAE_F_3D_LayerNorm, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z = args["latent"]
        self.args = args

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([16,16,16])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([9,9,9])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([5,5,5])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*8+1, ngf*500, bias=True)
        self.fc2 = nn.Linear(ngf*500, int(self.z*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z*1.5), self.z, bias=True)
        self.fc32 = nn.Linear(int(self.z*1.5), self.z, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z+1, int(self.z*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z*1.5), ngf*500, bias=True)
        self.cond3 = torch.nn.Linear(ngf*500, 10*10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(3,3,3), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([30,30,30])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([60,60,60])

        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), bias=False)
        self.bnco0 = torch.nn.LayerNorm([30,30,30])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([30,30,30])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([30,30,30])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([30,30,30])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        #self.dr03 = nn.Dropout(p=0.3, inplace=False)
        #self.dr05 = nn.Dropout(p=0.5, inplace=False)

    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,30,30,30))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return self.fc31(x), self.fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print(std)
        #print(mu)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,10,10,10)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 30, 30, 30])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 60, 60, 60])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        
        
        
        
class Discriminator_F_Conv_DIRENR_Diff_v4(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128):
        #Differenc critic with linear and log scale in convolution and difference section
        #Pooling befor and after difference
        
        
        super(Discriminator_F_Conv_DIRENR_Diff_v4, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc

        
        self.conv1b = torch.nn.Conv3d(1, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1b = torch.nn.LayerNorm([14,14,14])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn2b = torch.nn.LayerNorm([6,6,6])
        self.conv3b = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=1, padding=0, bias=False)


        self.conv1c = torch.nn.Conv3d(1, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1c = torch.nn.LayerNorm([14,14,14])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn2c = torch.nn.LayerNorm([6,6,6])
        self.conv3c = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=1, padding=0, bias=False)

        self.maxpool = torch.nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
 
        #self.fc = torch.nn.Linear(5 * 4 * 4, 1)
        self.fc1p = torch.nn.Linear(15*15*15, int(ndf/4)) 
        self.fc1pd = torch.nn.Linear(15*15*15, int(ndf/4)) 
        self.fc1plog = torch.nn.Linear(15*15*15, int(ndf/4)) 
        self.fc1pdlog = torch.nn.Linear(15*15*15, int(ndf/4)) 
        self.fc1b = torch.nn.Linear(ndf * 4 * 4 * 4, int(ndf/2))
        self.fc1c = torch.nn.Linear(ndf * 4 * 4 * 4, int(ndf/2))
        self.fc1e = torch.nn.Linear(1, int(ndf/2))
        self.fc2 = torch.nn.Linear(int(ndf/2)*3+int(ndf/4)*4, ndf*2)
        #self.fc2 = torch.nn.Linear(int(ndf/2)*3, ndf*2)
        self.fc3 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc4 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc5 = torch.nn.Linear(ndf*2, 1)


    def forward(self, img, E_true):
        imga = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        imgb = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)

        img = imgb - imga
        
        xb = F.leaky_relu(self.bn1b(self.conv1b(imgb)), 0.2)
        xb = F.leaky_relu(self.bn2b(self.conv2b(xb)), 0.2)
        xb = F.leaky_relu(self.conv3b(xb), 0.2)
        xb = xb.view(-1, self.ndf * 4 * 4 * 4)
        xb = F.leaky_relu(self.fc1b(xb), 0.2)        
        
        xc = F.leaky_relu(self.bn1c(self.conv1c(torch.log(imgb*50+1.0))), 0.2)
        xc = F.leaky_relu(self.bn2c(self.conv2c(xc)), 0.2)
        xc = F.leaky_relu(self.conv3c(xc), 0.2)
        xc = xc.view(-1, self.ndf * 4 * 4 * 4)
        xc = F.leaky_relu(self.fc1c(xc), 0.2)
        
        img_pool = self.maxpool(img)
        img_pool_diff = self.maxpool(imga)-self.maxpool(imgb)

        
        xb = torch.cat((xb, 
                        xc, 
                        F.leaky_relu(self.fc1p(img_pool.view(-1, 15*15*15))),
                        F.leaky_relu(self.fc1pd(img_pool_diff.view(-1, 15*15*15))),
                        F.leaky_relu(self.fc1plog(torch.log(1.0+50*torch.abs(img_pool.view(-1, 15*15*15))))),
                        F.leaky_relu(self.fc1pdlog(torch.log(1.0+50*torch.abs(img_pool_diff.view(-1, 15*15*15))))),
                        F.leaky_relu(self.fc1e(E_true), 0.2)), 1)

        xb = F.leaky_relu(self.fc2(xb), 0.2)
        xb = F.leaky_relu(self.fc3(xb), 0.2)
        xb = F.leaky_relu(self.fc4(xb), 0.2)
        xb = self.fc5(xb)

        return xb.view(-1) ### flattens
    
    
class Discriminator_F_Conv_DIRENR_Diff_v5(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128):
        super(Discriminator_F_Conv_DIRENR_Diff_v5, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc

        
        self.conv1b = torch.nn.Conv3d(1, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1b = torch.nn.LayerNorm([14,14,14])
        self.conv2b = torch.nn.Conv3d(ndf, ndf*2, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn2b = torch.nn.LayerNorm([6,6,6])
        self.conv3b = torch.nn.Conv3d(ndf*2, ndf*2, kernel_size=3, stride=1, padding=0, bias=False)
 
        #self.fc = torch.nn.Linear(5 * 4 * 4, 1)
        self.fc1a = torch.nn.Linear(30*30*30, int(ndf/2)) 
        self.fc1b = torch.nn.Linear(ndf *2 * 4 * 4 * 4, 2*int(ndf/2))
        #self.fc1c = torch.nn.Linear(ndf * 4 * 4 * 4, int(ndf/2))
        self.fc1e = torch.nn.Linear(1, int(ndf/2))
        self.fc2 = torch.nn.Linear(int(ndf/2)*4, ndf*2)
        self.fc3 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc4 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc5 = torch.nn.Linear(ndf*2, 1)


    def forward(self, img, E_true):
        imga = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        imgb = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)

        img = imgb - imga
        
        xb = F.leaky_relu(self.bn1b(self.conv1b(imgb)), 0.2)
        xb = F.leaky_relu(self.bn2b(self.conv2b(xb)), 0.2)
        xb = F.leaky_relu(self.conv3b(xb), 0.2)
        xb = xb.view(-1, self.ndf* 2 * 4 * 4 * 4)
        xb = F.leaky_relu(self.fc1b(xb), 0.2)        
        
        
        xb = torch.cat((xb, F.leaky_relu(self.fc1a(img.view(-1, 30*30*30))) , F.leaky_relu(self.fc1e(E_true), 0.2)), 1)

        xb = F.leaky_relu(self.fc2(xb), 0.2)
        xb = F.leaky_relu(self.fc3(xb), 0.2)
        xb = F.leaky_relu(self.fc4(xb), 0.2)
        xb = self.fc5(xb)

        return xb.view(-1) ### flattens
    
    
    
    
    
    
    
class BiBAE_3D_LayerNorm_core(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, nc=1, ngf=8):
        super(BiBAE_3D_LayerNorm_core, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z = args["latent"]
        self.args = args

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(5,5,5), stride=(2,1,1),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([15,6,6])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(5,5,5), stride=(2,1,1),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([8,6,6])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(5,5,5), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([4,3,3])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([4,3,3])

     
        self.fc1 = nn.Linear(4*3*3*ngf*8+1, ngf*500, bias=True)
        self.fc2 = nn.Linear(ngf*500, int(self.z*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z*1.5), self.z, bias=True)
        self.fc32 = nn.Linear(int(self.z*1.5), self.z, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z+1, int(self.z*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z*1.5), ngf*500, bias=True)
        self.cond3 = torch.nn.Linear(ngf*500, 10*2*2*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(3,3,3), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([30,6,6])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([60,12,12])

        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), bias=False)
        self.bnco0 = torch.nn.LayerNorm([30,6,6])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([30,6,6])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([30,6,6])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([30,6,6])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        #self.dr03 = nn.Dropout(p=0.3, inplace=False)
        #self.dr05 = nn.Dropout(p=0.5, inplace=False)

    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,30,6,6))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return self.fc31(x), self.fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,10,2,2)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 30, 6, 6])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 60, 12, 12])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        
        
        
        
        
class Discriminator_Conv_DIRENR_Diff_v3_core(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128):
        super(Discriminator_Conv_DIRENR_Diff_v3_core, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc

        
        self.conv1b = torch.nn.Conv3d(1, ndf, kernel_size=(3,3,3), stride=(2,1,1), padding=(0,1,1), bias=False)
        self.bn1b = torch.nn.LayerNorm([14,6,6])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=(3,3,3), stride=(2,1,1), padding=(0,1,1), bias=False)
        self.bn2b = torch.nn.LayerNorm([6,6,6])
        self.conv3b = torch.nn.Conv3d(ndf, ndf, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,0,0), bias=False)


        self.conv1c = torch.nn.Conv3d(1, ndf, kernel_size=(3,3,3), stride=(2,1,1), padding=(0,1,1), bias=False)
        self.bn1c = torch.nn.LayerNorm([14,6,6])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=(3,3,3), stride=(2,1,1), padding=(0,1,1), bias=False)
        self.bn2c = torch.nn.LayerNorm([6,6,6])
        self.conv3c = torch.nn.Conv3d(ndf, ndf, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,0,0), bias=False)

 
        #self.fc = torch.nn.Linear(5 * 4 * 4, 1)
        self.fc1a = torch.nn.Linear(30*6*6, int(ndf/2)) 
        self.fc1b = torch.nn.Linear(ndf * 4 * 4 * 4, int(ndf/2))
        self.fc1c = torch.nn.Linear(ndf * 4 * 4 * 4, int(ndf/2))
        self.fc1e = torch.nn.Linear(1, int(ndf/2))
        self.fc2 = torch.nn.Linear(int(ndf/2)*4, ndf*2)
        self.fc3 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc4 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc5 = torch.nn.Linear(ndf*2, 1)


    def forward(self, img, E_true):
        imga = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        imgb = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)

        img = imgb - imga
        
        xb = F.leaky_relu(self.bn1b(self.conv1b(imgb)), 0.2)
        xb = F.leaky_relu(self.bn2b(self.conv2b(xb)), 0.2)
        xb = F.leaky_relu(self.conv3b(xb), 0.2)
        xb = xb.view(-1, self.ndf * 4 * 4 * 4)
        xb = F.leaky_relu(self.fc1b(xb), 0.2)        
        
        xc = F.leaky_relu(self.bn1c(self.conv1c(torch.log(imgb+1.0))), 0.2)
        xc = F.leaky_relu(self.bn2c(self.conv2c(xc)), 0.2)
        xc = F.leaky_relu(self.conv3c(xc), 0.2)
        xc = xc.view(-1, self.ndf * 4 * 4 * 4)
        xc = F.leaky_relu(self.fc1c(xc), 0.2)
        
        xb = torch.cat((xb, xc, F.leaky_relu(self.fc1a(img.view(-1, 30*6*6))) , F.leaky_relu(self.fc1e(E_true), 0.2)), 1)

        xb = F.leaky_relu(self.fc2(xb), 0.2)
        xb = F.leaky_relu(self.fc3(xb), 0.2)
        xb = F.leaky_relu(self.fc4(xb), 0.2)
        xb = self.fc5(xb)

        return xb.view(-1) ### flattens
    
    
    
    
    
class BiBAE_F_3D_1ConvSkip(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, nc=1, ngf=8):
        super(BiBAE_F_3D_1ConvSkip, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z = args["latent"]
        self.args = args

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([16,16,16])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([9,9,9])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([5,5,5])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*8+1, ngf*500, bias=True)
        self.fc2 = nn.Linear(ngf*500, int(self.z*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z*1.5), self.z, bias=True)
        self.fc32 = nn.Linear(int(self.z*1.5), self.z, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z+1, int(self.z*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z*1.5), ngf*500, bias=True)
        self.cond3 = torch.nn.Linear(ngf*500, 10*10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(3,3,3), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([30,30,30])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([60,60,60])

        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), bias=False)
        self.bnco0 = torch.nn.LayerNorm([30,30,30])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([30,30,30])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([30,30,30])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([30,30,30])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.conv15 = torch.nn.Conv3d(1, ngf*4, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=True)
        self.conv16 = torch.nn.Conv3d(ngf*4, ngf*4, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=True)
        self.conv17 = torch.nn.Conv3d(ngf*4, ngf*4, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=True)
        self.conv18 = torch.nn.Conv3d(ngf*4, 1, kernel_size=(1,1,1), stride=(1,1,1), padding=(0,0,0), bias=True)
    
        #self.dr03 = nn.Dropout(p=0.3, inplace=False)
        #self.dr05 = nn.Dropout(p=0.5, inplace=False)

    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,30,30,30))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return self.fc31(x), self.fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print(std)
        #print(mu)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,10,10,10)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 30, 30, 30])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 60, 60, 60])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.conv4(x), inplace=True)
        identity = x 
        x = F.leaky_relu(self.conv15(x), inplace=True)
        x = F.leaky_relu(self.conv16(x), inplace=True)
        x = F.leaky_relu(self.conv17(x), inplace=True)
        x = F.leaky_relu(self.conv18(x), inplace=True)

        return x + identity
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z

        

    
    

class BasicResNetBlock3d(nn.Module):
    def __init__(self, in_featuremaps, in_dim, ngf, bias=False, rel = 'LeakyReLU'):
        super(BasicResNetBlock3d, self).__init__()
        
        self.bnco1 = torch.nn.LayerNorm([in_dim,in_dim,in_dim])
        self.conv1 = torch.nn.Conv3d(in_featuremaps, ngf, kernel_size=3, stride=1, padding=1, bias=bias)

        self.bnco2 = torch.nn.LayerNorm([in_dim,in_dim])
        self.conv2 = torch.nn.Conv3d(ngf, in_featuremaps, kernel_size=3, stride=1, padding=1, bias=bias)

        if rel=='LeakyReLU':
            self.relu = nn.LeakyReLU()
        elif rel=='ReLU': 
            self.relu = nn.ReLU()
       
    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.bnco1(out)
        out = self.conv1(out)
        
        out = self.relu(out)
        out = self.bnco2(out)
        out = self.conv2(out)

        out += identity
        #out = self.relu(out)

        return out
    
    
class InceptionResNetBlockTranspose3d(nn.Module):
    def __init__(self, in_featuremaps, in_dim, ngf, bias=False, rel = 'LeakyReLU'):
        super(InceptionResNetBlockTranspose3d, self).__init__()
        
        self.in_dim = in_dim
        self.ngf = ngf
        self.in_featuremaps = in_featuremaps
        
        self.bnco1a = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        self.conv1a = torch.nn.Conv3d(in_featuremaps, ngf*2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bnco1b = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        self.conv1b = torch.nn.Conv3d(ngf*2, in_featuremaps, kernel_size=3, stride=1, padding=1, bias=bias)

        self.bnco2a = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        self.conv2a = torch.nn.Conv3d(in_featuremaps, ngf*2, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2b = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        self.conv2b = torch.nn.Conv3d(ngf*2, in_featuremaps, kernel_size=1, stride=1, padding=0, bias=bias)

        #self.bnco3a = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        #self.deconv3a = torch.nn.ConvTranspose3d(in_featuremaps, ngf, kernel_size=3, stride=2, padding=1, bias=bias)
        #self.bnco3b = torch.nn.LayerNorm([in_dim*2, in_dim*2, in_dim*2])
        #self.conv3b = torch.nn.Conv3d(ngf, in_featuremaps, kernel_size=3, stride=2, padding=1, bias=bias)
 
        self.bnco4a = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        self.conv4a = torch.nn.Conv3d(in_featuremaps, ngf*4, kernel_size=3, stride=2, padding=1, bias=bias)
        self.bnco4b = torch.nn.LayerNorm([int(in_dim/2), int(in_dim/2), int(in_dim/2)])
        self.deconv4b = torch.nn.ConvTranspose3d(ngf*4, in_featuremaps, kernel_size=3, stride=2, padding=1, bias=bias)

        if rel=='LeakyReLU':
            self.relu = nn.LeakyReLU()
        elif rel=='ReLU': 
            self.relu = nn.ReLU()
       
    def forward(self, x):
        identity = x

        out1 = self.relu(x)
        out1 = self.bnco1a(out1)
        out1 = self.conv1a(out1)        
        out1 = self.relu(out1)
        out1 = self.bnco1b(out1)
        out1 = self.conv1b(out1)

        out2 = self.relu(x)
        out2 = self.bnco2a(out2)
        out2 = self.conv2a(out2)        
        out2 = self.relu(out2)
        out2 = self.bnco2b(out2)
        out2 = self.conv2b(out2)

        #out3 = self.relu(x)
        #out3 = self.bnco3a(out3)
        #out3 = self.deconv3a(out3, output_size=[out3.size(0), self.ngf, self.in_dim*2, self.in_dim*2])        
        #out3 = self.relu(out3)
        #out3 = self.bnco3b(out3)
        #out3 = self.conv3b(out3)
        
        out4 = self.relu(x)
        out4 = self.bnco4a(out4)
        out4 = self.conv4a(out4)
        out4 = self.relu(out4)
        out4 = self.bnco4b(out4)
        out4 = self.deconv4b(out4, output_size=[out4.size(0), self.in_featuremaps, self.in_dim, self.in_dim, self.in_dim])        


        #return identity + out1 + out2 + out3 + out4
        return identity + out1 + out2 + out4
    
    
    
class InceptionResNetBlock3d(nn.Module):
    def __init__(self, in_featuremaps, in_dim, ngf, bias=False, rel = 'LeakyReLU'):
        super(InceptionResNetBlock3d, self).__init__()
        
        self.in_dim = in_dim
        self.ngf = ngf
        self.in_featuremaps = in_featuremaps
        
        self.bnco1a = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        self.conv1a = torch.nn.Conv3d(in_featuremaps, ngf*2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bnco1b = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        self.conv1b = torch.nn.Conv3d(ngf*2, in_featuremaps, kernel_size=3, stride=1, padding=1, bias=bias)

        self.bnco2a = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        self.conv2a = torch.nn.Conv3d(in_featuremaps, ngf*2, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2b = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        self.conv2b = torch.nn.Conv3d(ngf*2, in_featuremaps, kernel_size=1, stride=1, padding=0, bias=bias)

        if rel=='LeakyReLU':
            self.relu = nn.LeakyReLU()
        elif rel=='ReLU': 
            self.relu = nn.ReLU()
            
    def forward(self, x):
        identity = x

        out1 = self.relu(x)
        out1 = self.bnco1a(out1)
        out1 = self.conv1a(out1)        
        out1 = self.relu(out1)
        out1 = self.bnco1b(out1)
        out1 = self.conv1b(out1)

        out2 = self.relu(x)
        out2 = self.bnco2a(out2)
        out2 = self.conv2a(out2)        
        out2 = self.relu(out2)
        out2 = self.bnco2b(out2)
        out2 = self.conv2b(out2)

        return identity + out1 + out2

    
    
class InceptionResNetBlockTranspose2d(nn.Module):
    def __init__(self, in_featuremaps, in_dim, ngf, bias=False, rel = 'LeakyReLU'):
        super(InceptionResNetBlockTranspose2d, self).__init__()
        
        self.in_dim = in_dim
        self.ngf = ngf
        self.in_featuremaps = in_featuremaps
        
        self.bnco1a = torch.nn.LayerNorm([in_dim, in_dim])
        self.conv1a = torch.nn.Conv2d(in_featuremaps, ngf*2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bnco1b = torch.nn.LayerNorm([in_dim, in_dim])
        self.conv1b = torch.nn.Conv2d(ngf*2, in_featuremaps, kernel_size=3, stride=1, padding=1, bias=bias)

        self.bnco2a = torch.nn.LayerNorm([in_dim, in_dim])
        self.conv2a = torch.nn.Conv2d(in_featuremaps, ngf*2, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2b = torch.nn.LayerNorm([in_dim, in_dim])
        self.conv2b = torch.nn.Conv2d(ngf*2, in_featuremaps, kernel_size=1, stride=1, padding=0, bias=bias)
 
        self.bnco4a = torch.nn.LayerNorm([in_dim, in_dim])
        self.conv4a = torch.nn.Conv2d(in_featuremaps, ngf*4, kernel_size=3, stride=2, padding=1, bias=bias)
        self.bnco4b = torch.nn.LayerNorm([int(in_dim/2), int(in_dim/2)])
        self.deconv4b = torch.nn.ConvTranspose2d(ngf*4, in_featuremaps, kernel_size=3, stride=2, padding=1, bias=bias)

        if rel=='LeakyReLU':
            self.relu = nn.LeakyReLU()
        elif rel=='ReLU': 
            self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out1 = self.relu(x)
        out1 = self.bnco1a(out1)
        out1 = self.conv1a(out1)        
        out1 = self.relu(out1)
        out1 = self.bnco1b(out1)
        out1 = self.conv1b(out1)

        out2 = self.relu(x)
        out2 = self.bnco2a(out2)
        out2 = self.conv2a(out2)        
        out2 = self.relu(out2)
        out2 = self.bnco2b(out2)
        out2 = self.conv2b(out2)
        
        out4 = self.relu(x)
        out4 = self.bnco4a(out4)
        out4 = self.conv4a(out4)
        out4 = self.relu(out4)
        out4 = self.bnco4b(out4)
        out4 = self.deconv4b(out4, output_size=[out4.size(0), self.in_featuremaps, self.in_dim, self.in_dim])        

        return identity + out1 + out2 + out4
    
    
    
class InceptionResNetBlock2d(nn.Module):
    def __init__(self, in_featuremaps, in_dim, ngf, bias=False, rel = 'LeakyReLU'):
        super(InceptionResNetBlock2d, self).__init__()
        
        self.in_dim = in_dim
        self.ngf = ngf
        self.in_featuremaps = in_featuremaps
        
        self.bnco1a = torch.nn.LayerNorm([in_dim, in_dim])
        self.conv1a = torch.nn.Conv2d(in_featuremaps, ngf*2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bnco1b = torch.nn.LayerNorm([in_dim, in_dim])
        self.conv1b = torch.nn.Conv2d(ngf*2, in_featuremaps, kernel_size=3, stride=1, padding=1, bias=bias)

        self.bnco2a = torch.nn.LayerNorm([in_dim, in_dim])
        self.conv2a = torch.nn.Conv2d(in_featuremaps, ngf*2, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2b = torch.nn.LayerNorm([in_dim, in_dim])
        self.conv2b = torch.nn.Conv2d(ngf*2, in_featuremaps, kernel_size=1, stride=1, padding=0, bias=bias)

        if rel=='LeakyReLU':
            self.relu = nn.LeakyReLU()
        elif rel=='ReLU': 
            self.relu = nn.ReLU()
       
    def forward(self, x):
        identity = x

        out1 = self.relu(x)
        out1 = self.bnco1a(out1)
        out1 = self.conv1a(out1)        
        out1 = self.relu(out1)
        out1 = self.bnco1b(out1)
        out1 = self.conv1b(out1)

        out2 = self.relu(x)
        out2 = self.bnco2a(out2)
        out2 = self.conv2a(out2)        
        out2 = self.relu(out2)
        out2 = self.bnco2b(out2)
        out2 = self.conv2b(out2)

        return identity + out1 + out2
    
    
class InceptionResNetBlockDeep3d(nn.Module):
    def __init__(self, in_featuremaps, in_dim, kernelDepth = 8, bias=False, rel = 'LeakyReLU'):
        super(InceptionResNetBlockDeep3d, self).__init__()
        self.in_dim = in_dim
        self.in_featuremaps = in_featuremaps
        self.kernelDepth = kernelDepth
        #input b, in_featuremaps, in_dim, in_dim, in_dim
        
        self.bnco1a = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        
        #b, in_featuremaps*in_dim, in_dim, in_dim
        self.conv1a = torch.nn.Conv2d(in_featuremaps*in_dim, in_featuremaps*in_dim*kernelDepth, 
                                      kernel_size=3, stride=1, padding=1, bias=bias)
        #b, K*in_featuremaps*in_dim, in_dim, in_dim
        self.bnco1b = torch.nn.LayerNorm([in_dim, in_dim])
        
        #b, K*in_featuremaps, in_dim, in_dim, in_dim
        self.conv1b = torch.nn.Conv3d(in_featuremaps, in_featuremaps, kernel_size=(kernelDepth,1,1), 
                                      stride=(kernelDepth,1,1), padding=(0,0,0), bias=bias)
        #b, in_featuremaps, in_dim, in_dim, in_dim
        if rel=='LeakyReLU':
            self.relu = nn.LeakyReLU()
        elif rel=='ReLU': 
            self.relu = nn.ReLU()
       
    def forward(self, x):
        identity = x

        out1 = self.relu(x)
        out1 = self.bnco1a(out1)
        out1 = self.conv1a(out1.view(-1, self.in_featuremaps*self.in_dim, self.in_dim, self.in_dim))        
        out1 = self.relu(out1)
        out1 = self.bnco1b(out1)
        out1 = self.conv1b(out1.view(-1, self.in_featuremaps, self.kernelDepth*self.in_dim, self.in_dim, self.in_dim))   

        return identity + out1

    
class CriticInceptionResNetBlock3d(nn.Module):
    def __init__(self, in_featuremaps, in_dim, ngf, bias=False, rel = 'LeakyReLU'):
        super(CriticInceptionResNetBlock3d, self).__init__()
        
        self.in_dim = in_dim
        self.ngf = ngf
        self.in_featuremaps = in_featuremaps
        
        self.bnco1a = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        self.conv1a = torch.nn.Conv3d(in_featuremaps, ngf, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bnco1b = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        self.conv1b = torch.nn.Conv3d(ngf, in_featuremaps, kernel_size=3, stride=1, padding=1, bias=bias)

        self.bnco2a = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        self.conv2a = torch.nn.Conv3d(in_featuremaps, ngf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2b = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        self.conv2b = torch.nn.Conv3d(ngf, in_featuremaps, kernel_size=1, stride=1, padding=0, bias=bias)

        self.bnco3a = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        self.conv3a = torch.nn.Conv3d(in_featuremaps, ngf, kernel_size=5, stride=1, padding=2, bias=bias)
        self.bnco3b = torch.nn.LayerNorm([in_dim, in_dim, in_dim])
        self.conv3b = torch.nn.Conv3d(ngf, in_featuremaps, kernel_size=5, stride=1, padding=2, bias=bias)
        
        if rel=='LeakyReLU':
            self.relu = nn.LeakyReLU()
        elif rel=='ReLU': 
            self.relu = nn.ReLU()
       
    def forward(self, x):
        identity = x

        out1 = self.relu(x)
        out1 = self.bnco1a(out1)
        out1 = self.conv1a(out1)        
        out1 = self.relu(out1)
        out1 = self.bnco1b(out1)
        out1 = self.conv1b(out1)

        out2 = self.relu(x)
        out2 = self.bnco2a(out2)
        out2 = self.conv2a(out2)        
        out2 = self.relu(out2)
        out2 = self.bnco2b(out2)
        out2 = self.conv2b(out2)

        out3 = self.relu(x)
        out3 = self.bnco3a(out3)
        out3 = self.conv3a(out3)        
        out3 = self.relu(out3)
        out3 = self.bnco3b(out3)
        out3 = self.conv3b(out3)
        
        return identity + out1 + out2 + out3

        
class BiBAE_F_3D_InceptRes(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, nc=1, ngf=8):
        super(BiBAE_F_3D_InceptRes, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z = args["latent"]
        self.args = args

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([16,16,16])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([9,9,9])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([5,5,5])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*8+1, ngf*500, bias=True)
        self.fc2 = nn.Linear(ngf*500, int(self.z*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z*1.5), self.z, bias=True)
        self.fc32 = nn.Linear(int(self.z*1.5), self.z, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z+1, int(self.z*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z*1.5), ngf*500, bias=True)
        self.cond3 = torch.nn.Linear(ngf*500, 10*10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(3,3,3), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([30,30,30])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([60,60,60])

        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf, ngf, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), bias=False)    
    
        self.block1 = InceptionResNetBlock3d(in_featuremaps=ngf, in_dim=30, ngf=ngf*2, bias=True)
        self.block2 = InceptionResNetBlockTranspose3d(in_featuremaps=ngf, in_dim=30, ngf=ngf, bias=True)
        self.block3 = InceptionResNetBlock3d(in_featuremaps=ngf, in_dim=30, ngf=ngf*2, bias=True)
        self.block4 = InceptionResNetBlockTranspose3d(in_featuremaps=ngf, in_dim=30, ngf=ngf, bias=True)
        self.block5 = InceptionResNetBlock3d(in_featuremaps=ngf, in_dim=30, ngf=ngf*2, bias=True)
        self.block6 = InceptionResNetBlockTranspose3d(in_featuremaps=ngf, in_dim=30, ngf=ngf, bias=True)

        self.conv4 = torch.nn.Conv3d(ngf, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)

        
        #self.dr03 = nn.Dropout(p=0.3, inplace=False)
        #self.dr05 = nn.Dropout(p=0.5, inplace=False)

    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,30,30,30))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return self.fc31(x), self.fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print(std)
        #print(mu)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,10,10,10)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 30, 30, 30])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 60, 60, 60])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu((self.conv0(x)), 0.2, inplace=True)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        
        
        
        
class Discriminator_F_Res_Diff(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=64):
        super(Discriminator_F_Res_Diff, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc

        #self.block1b = BasicResNetBlock3d(in_featuremaps=1, in_dim=30, ngf=ndf)        
        self.conv1b = torch.nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=0, bias=False)
        self.bn1b = torch.nn.LayerNorm([14,14,14])

        self.block2b = BasicResNetBlock3d(in_featuremaps=ndf, in_dim=14, ngf=ndf)
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=0, bias=False)
        self.bn2b = torch.nn.LayerNorm([6,6,6])

        self.block3b = BasicResNetBlock3d(in_featuremaps=ndf, in_dim=6, ngf=ndf*2)        
        self.conv3b = torch.nn.Conv3d(ndf, ndf*2, kernel_size=3, stride=1, padding=0, bias=False)


 
        #self.fc = torch.nn.Linear(5 * 4 * 4, 1)
        self.fc1a = torch.nn.Linear(30*30*30, ndf) 
        self.fc1b = torch.nn.Linear(ndf * 4 * 4 * 4 * 2, ndf)
        self.fc1e = torch.nn.Linear(1, ndf)
        self.fc2 = torch.nn.Linear(ndf*3, ndf*8)
        self.fc3 = torch.nn.Linear(ndf*8, ndf*4)
        self.fc4 = torch.nn.Linear(ndf*4, ndf*2)
        self.fc5 = torch.nn.Linear(ndf*2, 1)


    def forward(self, img, E_true):
        imga = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        imgb = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)

        img = imgb - imga
        
        #xb = self.block1b(imgb)
        xb = F.leaky_relu(self.bn1b(self.conv1b(imgb)), 0.2)
        xb = self.block2b(xb)
        xb = F.leaky_relu(self.bn2b(self.conv2b(xb)), 0.2)
        xb = self.block3b(xb)
        xb = F.leaky_relu(self.conv3b(xb), 0.2)
        xb = xb.view(-1, self.ndf * 4 * 4 * 4 * 2)
        xb = F.leaky_relu(self.fc1b(xb), 0.2)        
                
        xb = torch.cat((xb, F.leaky_relu(self.fc1a(img.view(-1, 30*30*30))) , F.leaky_relu(self.fc1e(E_true), 0.2)), 1)

        xb = F.leaky_relu(self.fc2(xb), 0.2)
        xb = F.leaky_relu(self.fc3(xb), 0.2)
        xb = F.leaky_relu(self.fc4(xb), 0.2)
        xb = self.fc5(xb)

        return xb.view(-1) ### flattens
    
    
    
class BiBAE_F_3D_2D_InceptRes(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, nc=1, ngf=8):
        super(BiBAE_F_3D_2D_InceptRes, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z = args["latent"]
        self.args = args

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([16,16,16])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([9,9,9])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([5,5,5])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*8+1, ngf*100, bias=True)
        self.fc2 = nn.Linear(ngf*100, int(self.z*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z*1.5), self.z, bias=True)
        self.fc32 = nn.Linear(int(self.z*1.5), self.z, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z+1, int(self.z*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z*1.5), ngf*100, bias=True)
        self.cond3 = torch.nn.Linear(ngf*100, 10*10*ngf, bias=True)
        
        fmp = 2
        #b,nfg,10,10
        self.conv2d_1 = torch.nn.Conv2d(ngf, 10*fmp, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        #b,fmp*10,10,10
        self.block_2 = InceptionResNetBlockTranspose2d(in_featuremaps=10*fmp, in_dim=10, ngf=10*fmp*2, bias=True, rel='ReLU')
        #b,fmp*10,10,10
        self.conv2d_3 = torch.nn.Conv2d(10*fmp, 20*fmp, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        #b,fmp*20,10,10
        self.deco2d_4 = torch.nn.ConvTranspose2d(20*fmp, 20, kernel_size=(2,2), stride=(2,2), padding=(0,0), bias=False)
        #b,20,20,20
        self.block_5 = InceptionResNetBlockTranspose2d(in_featuremaps=20, in_dim=20, ngf=20*fmp*2, bias=True, rel='ReLU')
        #b,20,20,20
        #b,1,20,20,20
        self.block_6 = InceptionResNetBlock3d(in_featuremaps=1, in_dim=20, ngf=ngf, bias=True)
        #b,1,20,20,20
        self.block_7 = InceptionResNetBlockDeep3d(in_featuremaps=1, in_dim=20, kernelDepth = 8, bias=True, rel = 'ReLU')
        #b,1,20,20,20
        #b,20,20,20
        self.conv2d_8 = torch.nn.Conv2d(20, 40*fmp, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
        #b,fmp*40,20,20
        self.deco2d_9 = torch.nn.ConvTranspose2d(40*fmp, 40, kernel_size=(2,2), stride=(2,2), padding=(0,0), bias=False)
        #b,40,40,40
        self.block_10 = InceptionResNetBlockTranspose2d(in_featuremaps=40, in_dim=40, ngf=20*fmp, bias=True, rel='ReLU')
        #b,40,40,40
        #b,1,40,40,40
        self.block_11 = InceptionResNetBlock3d(in_featuremaps=1, in_dim=40, ngf=ngf, bias=True, rel='ReLU')
        #b,1,40,40,40
        self.block_12 = InceptionResNetBlockDeep3d(in_featuremaps=1, in_dim=40, kernelDepth = 4, bias=True, rel = 'ReLU')
        #b,1,40,40,40
        self.conv3d_13 = torch.nn.Conv3d(1, ngf, kernel_size=(5,5,5), stride=(1,1,1), padding=(0,0,0), bias=False)
        #b,nfg,36,36,36
        self.block_14 = InceptionResNetBlock3d(in_featuremaps=ngf, in_dim=36, ngf=ngf*2, bias=True, rel='ReLU')
        #b,nfg,36,36,36
        self.conv3d_15 = torch.nn.Conv3d(ngf, ngf, kernel_size=(5,5,5), stride=(1,1,1), padding=(0,0,0), bias=False)
        #b,nfg,32,32,32
        self.block_16 = InceptionResNetBlock3d(in_featuremaps=ngf, in_dim=32, ngf=ngf*2, bias=True, rel='ReLU')
        #b,nfg,32,32,32
        self.conv3d_17 = torch.nn.Conv3d(ngf, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,0,0), bias=False)
        #b,nfg,30,30,30


    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,30,30,30))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return self.fc31(x), self.fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print(std)
        #print(mu)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        ## change size for deconv2d network. Image is 10x10

        x = x.view(-1,self.ngf,10,10)        


        x = F.leaky_relu(self.conv2d_1(x), inplace=True)
        x = self.block_2(x)
        x = F.leaky_relu(self.conv2d_3(x), inplace=True)
        x = F.leaky_relu(self.deco2d_4(x, output_size=[x.size(0), 1, 20, 20]), inplace=True)
        x = self.block_5(x)
        x = self.block_6(x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3)))
        x = self.block_7(x)
        x = F.leaky_relu(self.conv2d_8(x.view(x.size(0), x.size(2), x.size(3), x.size(4))), inplace=True)
        x = F.leaky_relu(self.deco2d_9(x, output_size=[x.size(0), 1, 40, 40]), inplace=True)
        x = self.block_10(x)
        x = self.block_11(x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3)))
        x = self.block_12(x)
        x = F.leaky_relu(self.conv3d_13(x), inplace=True)
        x = self.block_14(x)        
        x = F.leaky_relu(self.conv3d_15(x), inplace=True)
        x = self.block_16(x)       
        x = F.relu(self.conv3d_17(x), inplace=True)
        
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        

        
class Discriminator_F_ResV2_Diff(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=64):
        super(Discriminator_F_ResV2_Diff, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc

        #self.block1b = BasicResNetBlock3d(in_featuremaps=1, in_dim=30, ngf=ndf)        
        self.conv1b = torch.nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1b = torch.nn.LayerNorm([15,15,15])

        self.block2b = BasicResNetBlock3d(in_featuremaps=ndf, in_dim=15, ngf=ndf)
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2b = torch.nn.LayerNorm([8,8,8])

        self.block3b = BasicResNetBlock3d(in_featuremaps=ndf, in_dim=8, ngf=ndf*2)        
        self.conv3b = torch.nn.Conv3d(ndf, ndf*2, kernel_size=3, stride=1, padding=1, bias=False)


 
        #self.fc = torch.nn.Linear(5 * 4 * 4, 1)
        self.fc1a = torch.nn.Linear(30*30*30, ndf) 
        self.fc1b = torch.nn.Linear(ndf * 8 * 8 * 8 * 2, ndf)
        self.fc1e = torch.nn.Linear(1, ndf)
        self.fc2 = torch.nn.Linear(ndf*3, ndf*8)
        self.fc3 = torch.nn.Linear(ndf*8, ndf*4)
        self.fc4 = torch.nn.Linear(ndf*4, ndf*2)
        self.fc5 = torch.nn.Linear(ndf*2, 1)


    def forward(self, img, E_true):
        imga = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        imgb = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)

        img = imgb - imga
        
        #xb = self.block1b(imgb)
        xb = F.leaky_relu(self.bn1b(self.conv1b(imgb)), 0.2)
        xb = self.block2b(xb)
        xb = F.leaky_relu(self.bn2b(self.conv2b(xb)), 0.2)
        xb = self.block3b(xb)
        xb = F.leaky_relu(self.conv3b(xb), 0.2)
        xb = xb.view(-1, self.ndf * 8 * 8 * 8 * 2)
        xb = F.leaky_relu(self.fc1b(xb), 0.2)        
                
        xb = torch.cat((xb, F.leaky_relu(self.fc1a(img.view(-1, 30*30*30))) , F.leaky_relu(self.fc1e(E_true), 0.2)), 1)

        xb = F.leaky_relu(self.fc2(xb), 0.2)
        xb = F.leaky_relu(self.fc3(xb), 0.2)
        xb = F.leaky_relu(self.fc4(xb), 0.2)
        xb = self.fc5(xb)

        return xb.view(-1) ### flattens
    
    
    
class BiBAE_F_3D_InceptRes_Flip(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, nc=1, ngf=8):
        super(BiBAE_F_3D_InceptRes_Flip, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z = args["latent"]
        self.args = args

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([16,16,16])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([9,9,9])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([5,5,5])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*8+1, ngf*500, bias=True)
        self.fc2 = nn.Linear(ngf*500, int(self.z*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z*1.5), self.z, bias=True)
        self.fc32 = nn.Linear(int(self.z*1.5), self.z, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z+1, int(self.z*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z*1.5), ngf*500, bias=True)
        self.cond3 = torch.nn.Linear(ngf*500, 10*10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(3,3,3), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([30,30,30])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([60,60,60])

        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf, ngf, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), bias=False)    
    
        self.block1 = InceptionResNetBlock3d(in_featuremaps=ngf, in_dim=30, ngf=ngf*2, bias=True)
        self.block2 = InceptionResNetBlockTranspose3d(in_featuremaps=ngf, in_dim=30, ngf=ngf, bias=True)
        self.block3 = InceptionResNetBlock3d(in_featuremaps=ngf, in_dim=30, ngf=ngf*2, bias=True)
        self.block4 = InceptionResNetBlockTranspose3d(in_featuremaps=ngf, in_dim=30, ngf=ngf, bias=True)
        self.block5 = InceptionResNetBlock3d(in_featuremaps=ngf, in_dim=30, ngf=ngf*2, bias=True)
        self.block6 = InceptionResNetBlockTranspose3d(in_featuremaps=ngf, in_dim=30, ngf=ngf, bias=True)

        self.conv4 = torch.nn.Conv3d(ngf, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)

        
        #self.dr03 = nn.Dropout(p=0.3, inplace=False)
        #self.dr05 = nn.Dropout(p=0.5, inplace=False)

    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,30,30,30))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return self.fc31(x), self.fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print(std)
        #print(mu)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,10,10,10)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 30, 30, 30])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 60, 60, 60])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu((self.conv0(x)), 0.2, inplace=True)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = F.relu(self.conv4(x), inplace=True)
        
        
        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, [3])
    
        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, [4])

        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        
        
        
class BiBAE_F_3D_InceptRes_Flip_SmallLatent(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_InceptRes_Flip_SmallLatent, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([16,16,16])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([9,9,9])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([5,5,5])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*8+1, ngf*500, bias=True)
        self.fc2 = nn.Linear(ngf*500, int(self.z_full*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full*1.5), ngf*500, bias=True)
        self.cond3 = torch.nn.Linear(ngf*500, 10*10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(3,3,3), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([30,30,30])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([60,60,60])

        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf, ngf, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), bias=False)    
    
        self.block1 = InceptionResNetBlock3d(in_featuremaps=ngf, in_dim=30, ngf=ngf*2, bias=True)
        self.block2 = InceptionResNetBlockTranspose3d(in_featuremaps=ngf, in_dim=30, ngf=ngf, bias=True)
        self.block3 = InceptionResNetBlock3d(in_featuremaps=ngf, in_dim=30, ngf=ngf*2, bias=True)
        self.block4 = InceptionResNetBlockTranspose3d(in_featuremaps=ngf, in_dim=30, ngf=ngf, bias=True)
        self.block5 = InceptionResNetBlock3d(in_featuremaps=ngf, in_dim=30, ngf=ngf*2, bias=True)
        self.block6 = InceptionResNetBlockTranspose3d(in_featuremaps=ngf, in_dim=30, ngf=ngf, bias=True)

        self.conv4 = torch.nn.Conv3d(ngf, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)

        
        #self.dr03 = nn.Dropout(p=0.3, inplace=False)
        #self.dr05 = nn.Dropout(p=0.5, inplace=False)

    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,30,30,30))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print(std)
        #print(mu)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,10,10,10)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 30, 30, 30])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 60, 60, 60])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu((self.conv0(x)), 0.2, inplace=True)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = F.relu(self.conv4(x), inplace=True)
        
        
        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, [3])
    
        if torch.rand(1)[0] > 0.5:
            x = torch.flip(x, [4])

        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        
        
        
class Discriminator_F_Conv_Fence_Diff_v3(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128):
        super(Discriminator_F_Conv_Fence_Diff_v3, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc

        
        self.conv1b = torch.nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=0, bias=False)
        self.bn1b = torch.nn.LayerNorm([14,14,14])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=0, bias=False)
        self.bn2b = torch.nn.LayerNorm([6,6,6])
        self.conv3b = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=1, padding=0, bias=False)


        self.conv1c = torch.nn.Conv3d(1, ndf, kernel_size=4, stride=2, padding=0, bias=False)
        self.bn1c = torch.nn.LayerNorm([14,14,14])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=4, stride=2, padding=0, bias=False)
        self.bn2c = torch.nn.LayerNorm([6,6,6])
        self.conv3c = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=1, padding=0, bias=False)

 
        #self.fc = torch.nn.Linear(5 * 4 * 4, 1)
        self.fc1a = torch.nn.Linear(30*30*30, int(ndf/2)) 
        self.fc1b = torch.nn.Linear(ndf * 4 * 4 * 4, int(ndf/2))
        self.fc1c = torch.nn.Linear(ndf * 4 * 4 * 4, int(ndf/2))
        self.fc1e = torch.nn.Linear(1, int(ndf/2))
        self.fc2 = torch.nn.Linear(int(ndf/2)*4, ndf*2)
        self.fc3 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc4 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc5 = torch.nn.Linear(ndf*2, 1)


    def forward(self, img, E_true):
        imga = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        imgb = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)

        img = imgb - imga
        
        xb = F.leaky_relu(self.bn1b(self.conv1b(imgb)), 0.2)
        xb = F.leaky_relu(self.bn2b(self.conv2b(xb)), 0.2)
        xb = F.leaky_relu(self.conv3b(xb), 0.2)
        xb = xb.view(-1, self.ndf * 4 * 4 * 4)
        xb = F.leaky_relu(self.fc1b(xb), 0.2)        
        
        xc = F.leaky_relu(self.bn1c(self.conv1c(torch.log(imgb+1.0))), 0.2)
        xc = F.leaky_relu(self.bn2c(self.conv2c(xc)), 0.2)
        xc = F.leaky_relu(self.conv3c(xc), 0.2)
        xc = xc.view(-1, self.ndf * 4 * 4 * 4)
        xc = F.leaky_relu(self.fc1c(xc), 0.2)
        
        xb = torch.cat((xb, xc, F.leaky_relu(self.fc1a(img.view(-1, 30*30*30))) , F.leaky_relu(self.fc1e(E_true), 0.2)), 1)

        xb = F.leaky_relu(self.fc2(xb), 0.2)
        xb = F.leaky_relu(self.fc3(xb), 0.2)
        xb = F.leaky_relu(self.fc4(xb), 0.2)
        xb = self.fc5(xb)

        return xb.view(-1) ### flattens

    
    
class BiBAE_F_3D_LayerNorm_SmallLatent(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_LayerNorm_SmallLatent, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([16,16,16])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([9,9,9])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([5,5,5])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*8+1, ngf*500, bias=True)
        self.fc2 = nn.Linear(ngf*500, int(self.z_full*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full*1.5), ngf*500, bias=True)
        self.cond3 = torch.nn.Linear(ngf*500, 10*10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(3,3,3), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([30,30,30])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([60,60,60])

        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), bias=False)
        self.bnco0 = torch.nn.LayerNorm([30,30,30])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([30,30,30])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([30,30,30])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([30,30,30])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        #self.dr03 = nn.Dropout(p=0.3, inplace=False)
        #self.dr05 = nn.Dropout(p=0.5, inplace=False)

    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,30,30,30))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print(std)
        #print(mu)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,10,10,10)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 30, 30, 30])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 60, 60, 60])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        

class BiBAE_F_3D_LayerNorm_SmallLatentLinOut(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_LayerNorm_SmallLatentLinOut, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([16,16,16])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([9,9,9])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([5,5,5])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*8+1, ngf*500, bias=True)
        self.fc2 = nn.Linear(ngf*500, int(self.z_full*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full*1.5), ngf*500, bias=True)
        self.cond3 = torch.nn.Linear(ngf*500, 10*10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(3,3,3), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([30,30,30])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([60,60,60])

        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), bias=False)
        self.bnco0 = torch.nn.LayerNorm([30,30,30])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([30,30,30])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([30,30,30])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([30,30,30])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        #self.dr03 = nn.Dropout(p=0.3, inplace=False)
        #self.dr05 = nn.Dropout(p=0.5, inplace=False)

    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,30,30,30))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print(std)
        #print(mu)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,10,10,10)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 30, 30, 30])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 60, 60, 60])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = self.conv4(x)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        
        
class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)*(1.0/np.sqrt(in_channels*out_channels))
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        
    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out    
    
    
    
class LocallyConnected3dSize1(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], 
                        output_size[2], 1)*(1.0/np.sqrt(in_channels*out_channels))
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        
    def forward(self, x):
        _, c, h, w, d = x.size()

        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out    
    
class PostProcess_Size1Conv(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128, bias=False, out_funct='relu'):
        super(PostProcess_Size1Conv, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.conv1 = torch.nn.Conv3d(1, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco1 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])
        
        self.conv2 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv3 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco3 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv4 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)

        self.conv5 = torch.nn.Conv3d(ndf, 1, kernel_size=1, stride=1, padding=0, bias=False)
 

    def forward(self, img, E_True=0):
        img.view(-1, 1, self.isize, self.isize, self.isize)

        img = F.leaky_relu(self.bnco1(self.conv1(img)), 0.02)
        img = F.leaky_relu(self.bnco2(self.conv2(img)), 0.02)
        img = F.leaky_relu(self.bnco3(self.conv3(img)), 0.02)
        img = F.leaky_relu(self.conv4(img), 0.02) 
        img = self.conv5(img)

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.02) 
              
        return img.view(-1, 1, self.isize, self.isize, self.isize)

    
    

    
class PostProcess_Size1Conv_Econd(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128, bias=False, out_funct='relu'):
        super(PostProcess_Size1Conv_Econd, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.fcec1 = torch.nn.Linear(1, int(ndf/4), bias=True)
        self.fcec2 = torch.nn.Linear(int(ndf/4), int(ndf/4), bias=True)
        
        self.conv1 = torch.nn.Conv3d(1, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco1 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])
        
        self.conv2 = torch.nn.Conv3d(ndf+int(ndf/4), ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv3 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco3 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv4 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco4 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv5 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)

        self.conv6 = torch.nn.Conv3d(ndf, 1, kernel_size=1, stride=1, padding=0, bias=False)
 

    def forward(self, img, E_True=0):
        img.view(-1, 1, self.isize, self.isize, self.isize)

        econd = F.leaky_relu(self.fcec1(E_True), 0.02)
        econd = F.leaky_relu(self.fcec2(econd), 0.02)
        
        econd = econd.view(-1, int(self.ndf/4), 1, 1, 1)
        econd = econd.expand(-1, -1, self.isize, self.isize, self.isize)        
        
        img = F.leaky_relu(self.bnco1(self.conv1(img)), 0.02)
        img = torch.cat((img, econd), 1)
        img = F.leaky_relu(self.bnco2(self.conv2(img)), 0.02)
        img = F.leaky_relu(self.bnco3(self.conv3(img)), 0.02)
        img = F.leaky_relu(self.bnco4(self.conv4(img)), 0.02)
        img = F.leaky_relu(self.conv5(img), 0.02) 
        img = self.conv6(img)

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.02) 
              
        return img.view(-1, 1, self.isize, self.isize, self.isize)

    
    
class PostProcess_Size1Conv_EcondV2(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128, bias=False, out_funct='relu'):
        super(PostProcess_Size1Conv_EcondV2, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.fcec1 = torch.nn.Linear(2, int(ndf/2), bias=True)
        self.fcec2 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        self.fcec3 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        
        self.conv1 = torch.nn.Conv3d(1, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco1 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])
        
        self.conv2 = torch.nn.Conv3d(ndf+int(ndf/2), ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv3 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco3 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv4 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco4 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv5 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)

        self.conv6 = torch.nn.Conv3d(ndf, 1, kernel_size=1, stride=1, padding=0, bias=False)
 

    def forward(self, img, E_True=0):
        img = img.view(-1, 1, self.isize, self.isize, self.isize)
                
        econd = torch.cat((torch.sum(img.view(-1, self.isize*self.isize*self.isize), 1).view(-1, 1), E_True), 1)

        econd = F.leaky_relu(self.fcec1(econd), 0.01)
        econd = F.leaky_relu(self.fcec2(econd), 0.01)
        econd = F.leaky_relu(self.fcec3(econd), 0.01)
        
        econd = econd.view(-1, int(self.ndf/2), 1, 1, 1)
        econd = econd.expand(-1, -1, self.isize, self.isize, self.isize)        
        
        img = F.leaky_relu(self.bnco1(self.conv1(img)), 0.01)
        img = torch.cat((img, econd), 1)
        img = F.leaky_relu(self.bnco2(self.conv2(img)), 0.01)
        img = F.leaky_relu(self.bnco3(self.conv3(img)), 0.01)
        img = F.leaky_relu(self.bnco4(self.conv4(img)), 0.01)
        img = F.leaky_relu(self.conv5(img), 0.01) 
        img = self.conv6(img)

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.01) 
              
        return img.view(-1, 1, self.isize, self.isize, self.isize)


class PostProcess_Size1Conv1D_EcondV2(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128, bias=False, out_funct='relu'):
        super(PostProcess_Size1Conv1D_EcondV2, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.fcec1 = torch.nn.Linear(2, int(ndf/2), bias=True)
        self.fcec2 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        self.fcec3 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        
        self.conv1 = torch.nn.Conv1d(1, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco1 = torch.nn.LayerNorm([self.isize * self.isize * self.isize])
        
        self.conv2 = torch.nn.Conv1d(ndf+int(ndf/2), ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2 = torch.nn.LayerNorm([self.isize * self.isize * self.isize])

        self.conv3 = torch.nn.Conv1d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco3 = torch.nn.LayerNorm([self.isize * self.isize * self.isize])

        self.conv4 = torch.nn.Conv1d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco4 = torch.nn.LayerNorm([self.isize * self.isize * self.isize])

        self.conv5 = torch.nn.Conv1d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)

        self.conv6 = torch.nn.Conv1d(ndf, 1, kernel_size=1, stride=1, padding=0, bias=False)
 

    def forward(self, img, E_True=0):
        img = img.view(-1, 1, self.isize*self.isize*self.isize)
                
        econd = torch.cat((torch.sum(img.view(-1, self.isize*self.isize*self.isize), 1).view(-1, 1), E_True), 1)

        econd = F.leaky_relu(self.fcec1(econd), 0.01)
        econd = F.leaky_relu(self.fcec2(econd), 0.01)
        econd = F.leaky_relu(self.fcec3(econd), 0.01)
        
        econd = econd.view(-1, int(self.ndf/2), 1)
        econd = econd.expand(-1, -1, self.isize * self.isize * self.isize)        
        
        img = F.leaky_relu(self.bnco1(self.conv1(img)), 0.01)
        img = torch.cat((img, econd), 1)
        img = F.leaky_relu(self.bnco2(self.conv2(img)), 0.01)
        img = F.leaky_relu(self.bnco3(self.conv3(img)), 0.01)
        img = F.leaky_relu(self.bnco4(self.conv4(img)), 0.01)
        img = F.leaky_relu(self.conv5(img), 0.01) 
        img = self.conv6(img)

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.01) 
              
        return img.view(-1, 1, self.isize, self.isize, self.isize)




    
class PostProcess_EScale_EcondV2(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=256, bias=False, out_funct='relu'):
        super(PostProcess_EScale_EcondV2, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.fcec1 = torch.nn.Linear(2, int(ndf), bias=True)
        self.fcec2 = torch.nn.Linear(int(ndf), int(ndf), bias=True)
        self.fcec3 = torch.nn.Linear(int(ndf), int(ndf), bias=True)
        self.fcec4 = torch.nn.Linear(int(ndf), 1, bias=True)
        
 

    def forward(self, img, E_True=0):
        img = img.view(-1, 1, self.isize, self.isize, self.isize)
                
        econd = torch.cat((torch.sum(img.view(-1, self.isize*self.isize*self.isize), 1).view(-1, 1), E_True), 1)

        econd = F.leaky_relu(self.fcec1(econd), 0.01)
        econd = F.leaky_relu(self.fcec2(econd), 0.01)
        econd = F.leaky_relu(self.fcec3(econd), 0.01)
        econd = (self.fcec4(econd))
        
        econd = econd.view(-1, 1, 1, 1, 1)
        

        img = img*econd

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.01) 
              
        return img.view(-1, 1, self.isize, self.isize, self.isize)
 
    
    
    
class PostProcess_Size1ConvSkip(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128, bias=False, activ_funct='relu'):
        super(PostProcess_Size1ConvSkip, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.bais = bias
        self.activ_funct = activ_funct

        self.bnco11 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])
        self.conv11 = torch.nn.Conv3d(1, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        
        self.bnco12 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])
        self.conv12 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)

        self.bnco13 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])
        self.conv13 = torch.nn.Conv3d(ndf, 1, kernel_size=1, stride=1, padding=0, bias=bias)

        
        self.bnco21 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])
        self.conv21 = torch.nn.Conv3d(1, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        
        self.bnco22 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])
        self.conv22 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)

        self.bnco23 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])
        self.conv23 = torch.nn.Conv3d(ndf, 1, kernel_size=1, stride=1, padding=0, bias=bias)

        
        self.bnco31 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])
        self.conv31 = torch.nn.Conv3d(1, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        
        self.bnco32 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])
        self.conv32 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)

        self.bnco33 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])
        self.conv33 = torch.nn.Conv3d(ndf, 1, kernel_size=1, stride=1, padding=0, bias=bias)

        
        if self.activ_funct == 'relu':
            self.relu = nn.ReLU()

        elif self.activ_funct == 'leaky_relu':
            self.relu = nn.LeakyReLU(negative_slope=0.02)


 
    def forward(self, img, E_True=0):
        img.view(-1, 1, self.isize, self.isize, self.isize)
        
        identity = img
        
        out = self.relu(img)
        out = self.bnco11(out)
        out = self.conv11(out)        
        out = self.relu(out)
        out = self.bnco12(out)
        out = self.conv12(out)        
        out = self.relu(out)
        out = self.bnco13(out)
        out = self.conv13(out)
        
        out1 = out + identity

        identity = out1
        
        out = self.relu(out1)
        out = self.bnco21(out)
        out = self.conv21(out)        
        out = self.relu(out)
        out = self.bnco22(out)
        out = self.conv22(out)        
        out = self.relu(out)
        out = self.bnco23(out)
        out = self.conv23(out)
        
        out2 = out + identity

        
        identity = out2
        
        out = self.relu(out2)
        out = self.bnco31(out)
        out = self.conv31(out)        
        out = self.relu(out)
        out = self.bnco32(out)
        out = self.conv32(out)        
        out = self.relu(out)
        out = self.bnco33(out)
        out = self.conv33(out)
        
        out3 = out + identity

        return out3.view(-1, 1, self.isize, self.isize, self.isize)
    
    
    
    
class energyRegressor(nn.Module):
    """ 
    Energy regressor of WGAN. 
    """

    def __init__(self):
        super(energyRegressor, self).__init__()
        
        ## 3d conv layers
        self.conv1 = torch.nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1 = torch.nn.LayerNorm([14,14,14])
        self.conv2 = torch.nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn2 = torch.nn.LayerNorm([6,6,6])
        self.conv3 = torch.nn.Conv3d(32, 16, kernel_size=2, stride=1, padding=0, bias=False)
 
       
        ## FC layers
        self.fc1 = torch.nn.Linear(16 * 5 * 5 * 5, 100)
        self.fc2 = torch.nn.Linear(100, 1)
        
    def forward(self, x):
        in_size = x.size(-1)
        x = x.view(-1, 1, in_size, in_size, in_size)
        
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = self.conv3(x)
      
        ## shape [5, 5, 5]
        
        ## flatten for FC
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3) * x.size(4))
        
        ## pass to FC layers
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.relu(self.fc2(x))
        return x 
    
    
    
    
class PostProcess_EScaleConv_EcondV2(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=256, bias=False, out_funct='relu'):
        super(PostProcess_EScaleConv_EcondV2, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.conv1 = torch.nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1 = torch.nn.LayerNorm([14,14,14])
        self.conv2 = torch.nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn2 = torch.nn.LayerNorm([6,6,6])
        self.conv3 = torch.nn.Conv3d(32, 16, kernel_size=2, stride=1, padding=0, bias=False)
 
       
        ## FC layers
        self.fc1 = torch.nn.Linear(16 * 5 * 5 * 5, int(ndf/2))        
        
        self.fcec1 = torch.nn.Linear(2+int(ndf/2), int(ndf), bias=True)
        self.fcec2 = torch.nn.Linear(int(ndf), int(ndf), bias=True)
        self.fcec3 = torch.nn.Linear(int(ndf), int(ndf), bias=True)
        self.fcec4 = torch.nn.Linear(int(ndf), 1, bias=True)
        
 

    def forward(self, img, E_True=0):
        img = img.view(-1, 1, self.isize, self.isize, self.isize)
        
       
        x = F.leaky_relu(self.bn1(self.conv1(img)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = self.conv3(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3) * x.size(4))
        x = F.leaky_relu(self.fc1(x), 0.2)
        
        econd = torch.cat((torch.sum(img.view(-1, self.isize*self.isize*self.isize), 1).view(-1, 1), E_True, x), 1)

        econd = F.leaky_relu(self.fcec1(econd), 0.01)
        econd = F.leaky_relu(self.fcec2(econd), 0.01)
        econd = F.leaky_relu(self.fcec3(econd), 0.01)
        econd = (self.fcec4(econd))
        
        econd = econd.view(-1, 1, 1, 1, 1)
        

        img = img*econd

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.01) 
              
        return img.view(-1, 1, self.isize, self.isize, self.isize)

    
class E_reg_Pool(nn.Module):
    """ 
    Energy regressor of WGAN. 
    """

    def __init__(self, ndf=64):
        super(E_reg_Pool, self).__init__()
        self.ndf = ndf
        ## 3d conv layers
        self.conv1 = torch.nn.AvgPool3d(kernel_size=2)
        self.bn1 = torch.nn.LayerNorm([15,15,15])
        self.conv2 = torch.nn.Conv3d(1, self.ndf, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = torch.nn.LayerNorm([8,8,8])
        self.conv3 = torch.nn.Conv3d(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, bias=False)
 
       
        ## FC layers
        self.fc1 = torch.nn.Linear(self.ndf * 6 * 6 * 6, self.ndf*2)
        self.fc2 = torch.nn.Linear(self.ndf*2, self.ndf*2)
        self.fc3 = torch.nn.Linear(self.ndf*2, self.ndf*2)
        self.fc4 = torch.nn.Linear(self.ndf*2, 1)
        
    def forward(self, x):
        in_size = x.size(-1)
        x = x.view(-1, 1, in_size, in_size, in_size)
        
        x = self.bn1(self.conv1(x))
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
      
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3) * x.size(4))
        
        x = F.leaky_relu(self.fc1(x), 0.02)
        x = F.leaky_relu(self.fc2(x), 0.02)
        x = F.leaky_relu(self.fc3(x), 0.02)
        x = self.fc4(x)
        return x 

    
    
    
class E_reg_Conv(nn.Module):
    """ 
    Energy regressor of WGAN. 
    """

    def __init__(self, ndf=64):
        super(E_reg_Conv, self).__init__()
        self.ndf = ndf
        ## 3d conv layers
        self.conv1 = torch.nn.Conv3d(1, int(self.ndf/4), kernel_size=3, stride=2, padding=1, bias=True)
        #self.bn1 = torch.nn.LayerNorm([15,15,15])
        self.conv2 = torch.nn.Conv3d(int(self.ndf/4), self.ndf, kernel_size=3, stride=2, padding=1, bias=True)
        #self.bn2 = torch.nn.LayerNorm([8,8,8])
        self.conv3 = torch.nn.Conv3d(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, bias=True)
 
       
        ## FC layers
        self.fc1 = torch.nn.Linear(self.ndf * 6 * 6 * 6, self.ndf*2)
        self.fc2 = torch.nn.Linear(self.ndf*2, self.ndf*2)
        self.fc3 = torch.nn.Linear(self.ndf*2, self.ndf*2)
        self.fc4 = torch.nn.Linear(self.ndf*2, 1)
        
    def forward(self, x):
        in_size = x.size(-1)
        x = x.view(-1, 1, in_size, in_size, in_size)
        
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = F.leaky_relu(self.conv3(x), 0.1)
      
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3) * x.size(4))
        
        x = F.leaky_relu(self.fc1(x), 0.02)
        x = F.leaky_relu(self.fc2(x), 0.02)
        x = F.leaky_relu(self.fc3(x), 0.02)
        x = self.fc4(x)
        return x 
    
    
    
    
class PP_crit_conv(nn.Module):
    def __init__(self, ndf=16):
        super(PP_crit_conv, self).__init__()
        self.ndf = ndf
        ## 3d conv layers
        self.conv1 = torch.nn.Conv3d(1, int(self.ndf/4), kernel_size=3, stride=2, padding=1, bias=True)
        self.bn1 = torch.nn.LayerNorm([15,15,15])
        self.conv2 = torch.nn.Conv3d(int(self.ndf/4), self.ndf, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn2 = torch.nn.LayerNorm([8,8,8])
        self.conv3 = torch.nn.Conv3d(self.ndf, self.ndf, kernel_size=3, stride=1, padding=0, bias=True)
 
       
        ## FC layers
        self.fc1 = torch.nn.Linear(self.ndf * 6 * 6 * 6, self.ndf)
        self.fcE = torch.nn.Linear(1, self.ndf)

        self.fc2 = torch.nn.Linear(self.ndf*2, self.ndf)
        self.fc3 = torch.nn.Linear(self.ndf, 1)
        

        
    def forward(self, x, E_true):
        in_size = x.size(-1)
        x = x.view(-1, 1, in_size, in_size, in_size)
        
        x = F.leaky_relu(self.conv1(x), 0.1)
        x = F.leaky_relu(self.conv2(x), 0.1)
        x = F.leaky_relu(self.conv3(x), 0.1)
      
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3) * x.size(4))
        
        x = F.leaky_relu(self.fc1(x), 0.02)
        E = F.leaky_relu(self.fcE(E_true), 0.02)
        x = torch.cat((x, E), 1)
        x = F.leaky_relu(self.fc2(x), 0.02)
        x = self.fc3(x)
        return x 


    
    
class PP_crit_dense(nn.Module):
    def __init__(self, ndf=32):
        super(PP_crit_dense, self).__init__()
        self.ndf = ndf
        ## 3d conv layers

        ## FC layers
        self.fc1 = torch.nn.Linear(30 * 30 * 30, self.ndf)
        self.fcE = torch.nn.Linear(1, self.ndf)

        self.fc2 = torch.nn.Linear(self.ndf*2, self.ndf)
        self.fc3 = torch.nn.Linear(self.ndf, 1)
        
    def forward(self, x, E_true):
        in_size = x.size(-1)

        x = x.view(-1, in_size * in_size * in_size)      

        x = F.leaky_relu(self.fc1(x), 0.02)
        E = F.leaky_relu(self.fcE(E_true), 0.02)
        x = torch.cat((x, E), 1)
        x = F.leaky_relu(self.fc2(x), 0.02)
        x = self.fc3(x)
        return x 

    
    
    
    
    
class Bottleneck_aux_net(nn.Module):
    def __init__(self, ndf=16, z=24):
        super(Bottleneck_aux_net, self).__init__()
        self.z = z 
        self.ndf = ndf
        
        self.struct_list = []
        for i in range(self.z):
            temp_struct = nn.Sequential(
                nn.Linear(1, self.ndf),
                nn.BatchNorm1d(num_features=self.ndf),
                nn.ELU(),
                nn.Linear(self.ndf, self.ndf),
                nn.BatchNorm1d(num_features=self.ndf),
                nn.ELU(),
                nn.Linear(self.ndf, 1)
            )
            self.struct_list.append(temp_struct)
        self.struct_list = nn.ModuleList(self.struct_list)

    def forward(self, latent):
        out_list = []
        for i in range(self.z):
            out = self.struct_list[i](latent[:,i].view(-1,1))
            out_list.append(out)
            
        x = torch.cat(out_list, 1)
        return x 

class Bottleneck_aux_Critic(nn.Module):
    def __init__(self, ndf = 25, z=24):
        super(Bottleneck_aux_Critic, self).__init__()
        self.z = z 
        self.ndf = ndf
        
        self.struct_list = []
        for i in range(self.z):
            temp_struct = nn.Sequential(
                nn.Linear(1, self.ndf),
                nn.LeakyReLU(),
                nn.Linear(self.ndf, self.ndf),
                nn.LeakyReLU(),                
                nn.Linear(self.ndf, self.ndf),
                nn.LeakyReLU(),
                nn.Linear(self.ndf, 1)
            )
            self.struct_list.append(temp_struct)
        self.struct_list = nn.ModuleList(self.struct_list)

            
    def forward(self, latent ):          
        out_list = []
        for i in range(self.z):
            out = self.struct_list[i](latent[:,i].view(-1,1))
            out_list.append(out)
            
        x = torch.cat(out_list, 1)    
        return x 
        
        
class BiBauxAE_F_3D_LayerNorm_SmallLatent(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=488, z_enc=24):
        super(BiBauxAE_F_3D_LayerNorm_SmallLatent, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([16,16,16])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([9,9,9])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([5,5,5])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*8+1, ngf*500, bias=True)
        self.fc2 = nn.Linear(ngf*500, int(self.z_full*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full*1.5), ngf*500, bias=True)
        self.cond3 = torch.nn.Linear(ngf*500, 10*10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(3,3,3), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([30,30,30])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([60,60,60])
        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), bias=False)
        self.bnco0 = torch.nn.LayerNorm([30,30,30])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([30,30,30])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([30,30,30])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([30,30,30])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,30,30,30))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1)
        return self.fc31(x), self.fc32(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print(std)
        #print(mu)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,10,10,10)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 30, 30, 30])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 60, 60, 60])), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full', aux_net=None):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            #z = self.reparameterize(mu, logvar)
            return mu, logvar
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            z = aux_net(z)
            z = torch.cat((z,torch.randn(x.size(0), self.z_rand)),1)
            
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        

        
        
        
        
        
        
        
        
        
        
class BiBAE_F_3D_LayerNorm_SmallLatent_Fast(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_LayerNorm_SmallLatent_Fast, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full*1.5), ngf*500, bias=True)
        self.cond3 = torch.nn.Linear(ngf*500, 10*10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(3,3,3), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([30,30,30])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([60,60,60])

        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), bias=False)
        self.bnco0 = torch.nn.LayerNorm([30,30,30])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([30,30,30])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([30,30,30])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([30,30,30])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    

    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        z = torch.cat((z,E_true), 1)
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,10,10,10)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 30, 30, 30])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 60, 60, 60])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 

