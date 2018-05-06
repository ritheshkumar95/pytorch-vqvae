
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder,MNIST
import os
from torchvision.utils import save_image,make_grid
from torchvision.transforms import Compose,Normalize,Resize,ToTensor
import numpy as np
import tqdm
import time


# In[2]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


# In[3]:


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, x):
        return x + self.block(x)


# In[4]:


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.embedding = nn.Embedding(K, dim)
        # self.embedding.weight.data.copy_(1./K * torch.randn(K, 256))
        self.embedding.weight.data.uniform_(-1./K, 1./K)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)

        z_e_x_transp = z_e_x.permute(0, 2, 3, 1)  # (B, H, W, C)
        emb = self.embedding.weight.transpose(0, 1)  # (C, K)
        dists = torch.pow(
            z_e_x_transp.unsqueeze(4) - emb[None, None, None],
            2
        ).sum(-2)
        latents = dists.min(-1)[1]
        return latents, z_e_x
 

    def decode(self, latents):
        shp = latents.size() + (-1, )
        z_q_x = self.embedding(latents.view(latents.size(0), -1))  # (B * H * W, C)
        z_q_x = z_q_x.view(*shp).permute(0, 3, 1, 2)  # (B, C, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde, z_q_x

    
    
    def forward(self, x):
        latents, z_e_x = self.encode(x)
        x_tilde, z_q_x = self.decode(latents)
        return x_tilde, z_e_x, z_q_x


# In[5]:


class MaskedConv2D(nn.Conv2d):
    def __init__(self, mask_type, *args,**kwargs):
        super(MaskedConv2D,self).__init__(*args,**kwargs)
        self.mask_type = mask_type
        _,_,h,w = self.weight.size()
        self.mask = torch.zeros((h,w)).to("cuda")
        self.mask[:h //2,:] = 1.
        self.mask[h//2,:w//2] = 1.
        if mask_type == "B":
            self.mask[h//2,w//2] = 1.
    def forward(self,x):
        self.weight.data = self.weight.data * self.mask
        return super(MaskedConv2D,self).forward(x)
        



class PixelCNN(nn.Module):
    def __init__(self,num_layers=15, filt_size=5,inp_channels=1,num_fm=16):
        super(PixelCNN,self).__init__()
        first_layer = MaskedConv2D("A",inp_channels,num_fm,filt_size,padding=filt_size//2,bias=False) # padding to get same conv
        layers = [first_layer]
        for i in range(num_layers-1):
            layers.append(nn.ReLU())
            layers.append(MaskedConv2D("B",num_fm,num_fm,filt_size,padding=filt_size//2,bias=False))
        # 1x 1 conv to get logits
        last_layer = nn.Conv2d(num_fm,256,1)
        layers.append(last_layer)
        self.layers = nn.Sequential(*layers)
        
    def forward(self,x):
        # no softmax cuz it gets combined in loss
        return self.layers(x)
        
    



class VMaskedConv2D(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(VMaskedConv2D,self).__init__(*args,**kwargs)
        _,_,h,w = self.weight.size()
        self.mask = torch.zeros((h,w)).to("cuda")
        self.mask[:h //2,:] = 1.
    def forward(self,x):
        self.weight.data = self.weight.data * self.mask
        return super(VMaskedConv2D,self).forward(x)


class HMaskedConv2D(nn.Conv2d):
    def __init__(self,mask_type,*args,**kwargs):
        super(HMaskedConv2D,self).__init__(*args,**kwargs)
        _,_,h,w = self.weight.size()
        self.mask = torch.zeros((h,w)).to("cuda")
        self.mask[h //2,:w//2] = 1.
        if mask_type == "B":
            self.mask[h //2,:w//2+1] = 1.
            
    def forward(self,x):
        self.weight.data = self.weight.data * self.mask
        return super(HMaskedConv2D,self).forward(x)



class GatedLayer(nn.Module):
    def __init__(self,in_channels,out_channels,filter_size,mask_type,cond_size,skip_cxn=False):
        super(GatedLayer,self).__init__()
        self.out_channels = out_channels
        self.vconv = VMaskedConv2D(in_channels,2*out_channels,filter_size,padding=filter_size//2,bias=False)
        self.hconv = HMaskedConv2D(mask_type,in_channels,2*out_channels,filter_size,padding=filter_size//2,bias=False)
        self.cond_embed = nn.Linear(cond_size, 2*out_channels)
        self.v2h_conv1x1 = nn.Conv2d(2*out_channels,2*out_channels,1)
        self.skip_conv1x1 = nn.Conv2d(out_channels,out_channels,1)
        self.skip_cxn = skip_cxn
    def forward(self,x_v,x_h,y):
        #y is the conditioning
        vc2p = self.vconv(x_v)
        vcf, vcg = torch.split(vc2p,self.out_channels,dim=1)
        y = self.cond_embed(y)
        yf,yg = torch.split(y,self.out_channels,dim=1)
        #broadcast conditioning vector to all pixels
        fv = F.tanh(vcf + yf[:,:,None,None] )
        gv = F.sigmoid(vcg + yg[:,:,None,None] )
        vout = fv * gv
        
        
        hc2p = self.hconv(x_h) + self.v2h_conv1x1(vc2p)
        hcf, hcg = torch.split(hc2p,self.out_channels,dim=1)
        fh = F.tanh(hcf + yf[:,:,None,None] )
        gh = F.sigmoid(hcg + yg[:,:,None,None] )
        hout = fh * gh
        if self.skip_cxn:
            hout  = self.skip_conv1x1(hout) + x_h
            
        return vout,hout

class GatedPixelCNN(nn.Module):
    def __init__(self,num_layers=15, filt_size=5,in_channels=1,
                 num_fm=16,cond_size=3,im_range=256):
        
        super(GatedPixelCNN,self).__init__()
        self.first_layer = GatedLayer(in_channels=in_channels,out_channels=num_fm,
                                 filter_size=filt_size,mask_type="A",cond_size=3,skip_cxn=True)
        layers = []
        for i in range(num_layers - 1):
            layers.append(GatedLayer(in_channels=num_fm,out_channels=num_fm,
                                 filter_size=filt_size,mask_type="B",cond_size=3,skip_cxn=True))
        self.layers = nn.Sequential(*layers)
           
        self.last_layer = nn.Conv2d(num_fm,im_range,1)
        
        #self.apply(weights_init)
            
    def forward(self,x,y):
        xv,xh = self.first_layer(x,x,y)
        for layer in self.layers:
            xv,xh = layer(xv,xh,y)
        return self.last_layer(xh)
    
    
    def sample(self, y, shape=(21, 21), batch_size=4):
        x = torch.zeros((batch_size,1, *shape),
            dtype=torch.int64).float().to("cuda")

        h,w = shape
        for i in range(h):
            for j in range(w):
                # calc unnormalized p(x_ij|x<) logits
                pxij_logit = self.forward(x,y)[:,:,i,j]

                # compute normalized p(x_ij|x<) 
                pxij = F.softmax(pxij_logit,dim=1)
                # sample from p(x_ij|x<)
                x[:,0,i,j].copy_(torch.multinomial(pxij,1).squeeze())
        return x.type(torch.int64)
        
        
            
        
        
        


# In[ ]:


# for the 4th image experiment
# this will encode the 21x21x1 from bigger VQ-VAE and encode to 3x1 ze, which is then made discrete thru VQ operation with 3 separate tables
# then autoregressive decoder will decode back to 21x21 discrete latents, which then are looked up and passed thru deconv decoder
# of bigger autoencoder
class Encoder(nn.Module):
    def __init__(self,input_dim, hidden_dim,final_dim):
        super(Encoder,self).__init__()
        self.final_dim = final_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=4, stride=2, padding=1),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.Conv2d(in_channels=hidden_dim, out_channels=final_dim, kernel_size=3, stride=1, padding=1),
            nn.AvgPool2d(5)
            )
        self.apply(weights_init)
    def forward(self,x):
        ze = self.encoder(x)
        
        return ze.view(-1,self.final_dim,1)

    

class VQ(nn.Module):
    def __init__(self, K=512,D=3):
        super(VQ,self).__init__()
        self.e =  torch.randn(size=(K,D), requires_grad=True).float().to("cuda")
        
    def forward(self,ze):
        l2diff = (ze.transpose(2,1) - self.e[None,:])**2
        z = l2diff.argmin(1) # discrete
        zq = torch.cat((self.e[z[:,0],0,None], self.e[z[:,1],1,None], self.e[z[:,2],2,None]),dim=-1) # continuous
        return z,zq


class MetaPixelVQVAE(nn.Module):
    def __init__(self, weights_path, hidden_dim):
        super(MetaPixelVQVAE,self).__init__()
        self.VQVAE = AutoEncoder(3, 256).to("cuda")
        self.VQVAE.load_state_dict(torch.load(os.path.join(weights_path)))
        self.vqe = Encoder(256,hidden_dim,3).to("cuda")
        self.vq = VQ().to("cuda")
        self.decoder = GatedPixelCNN(im_range=512,cond_size=3).to("cuda")
    def forward(self,x):
        Z, ZE = self.VQVAE.encode(x)
        Z,ZE = Z.detach(),ZE.detach()
        ze = self.vqe(ZE)
        z,zq = self.vq(ze)
        Z_tild = self.decoder(Z[:,None].float(),zq)
        
        return Z_tild,Z, zq, ze
        
    
        
        
    


# In[ ]:



# m = MaskedConv2D("A",1,16,5)

# x = torch.randint(low=0,high=255,size=(8,1,32,32)).long()
# m(x.float())
# print(m.mask)

# tr = DataLoader(MNIST('../data', train=True, download=True, transform=ToTensor()),
#                      batch_size=128, shuffle=True, num_workers=1, pin_memory=True)

# pcnn = PixelCNN().to("cuda")
# h,w = 28,28

# opt = Adam(pcnn.parameters(),lr=0.01)

# for epoch in range(5):
#     t = tqdm.tqdm(tr)
#     for x,y in t:
#         opt.zero_grad()
#         x = x.to("cuda")
#         x_tild = pcnn(x)
#         loss= F.cross_entropy(x_tild,x.long().squeeze())
#         loss.backward()
#         opt.step()
#         t.set_description("epoch:%iloss:%6.4f"%(epoch,float(loss.data)))

# # x_tild = p(x.float())
# # x.size()

# # sampling
# sample = torch.zeros((1,1,h,w)).to("cuda")


# for i in range(h):
#     for j in range(w):
#         s_tild = pcnn(sample)
#         probs = F.softmax(s_tild,dim=1)[0,:,i,j]
#         sample[:,0,i,j] = torch.multinomial(probs,1)
        

# import matplotlib.pyplot as plt
# %matplotlib inline

# plt.imshow(sample[0,0],cmap="gray")

    

# vm = VMaskedConv2D(1,16,5)

# x = torch.randint(low=0,high=255,size=(8,1,32,32)).long()
#print(vm.mask)
# vm(x.float())

# hm = HMaskedConv2D("A",1,16,5)

# hm.mask

#x = torch.randint(low=0,high=255,size=(8,1,32,32))

#hm(x)

# gl = GatedLayer(1,8,5,"A",3,skip_cxn=True)

# x = torch.randint(low=0,high=255,size=(8,1,32,32))

# y = torch.randint(low=0,high=512,size=(8,3))

# v,h = gl(x,x,y)

# gl2 = GatedLayer(8,8,5,"B",3,skip_cxn=True)

# gl2(v,h,y)

