
# coding: utf-8

# In[2]:


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
from atari_data import get_data_loaders
from tensorboardX import SummaryWriter
from modules_meta_vqvae import MetaPixelVQVAE
from modules import to_scalar
from copy import deepcopy

import argparse
import sys

if __name__ == "__main__":
    tmp_argv = deepcopy(sys.argv)
    test_notebook = False
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [""]
        test_notebook= True

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default="Boxing-v0")
    parser.add_argument("--lr",type=float,default=3e-3)
    parser.add_argument("--K",type=int,default=512)
    parser.add_argument("--dim",type=int,default=256)
    parser.add_argument("--lambda_",type=int,default=1)
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--data_dir",type=str,default="../data")
    parser.add_argument("--max_batch_size",type=int,default=4)
    parser.add_argument("--savedir",type=str,default=".")
    parser.add_argument("--weights_path", type=str, default="./models/vqvae_dataset=Boxing-v0_lr=0.0003_K=512_dim=256_lambda_=1_decoder=deconv_batch_size=32_epochs=100_num_workers=4_num_frames=60000_resize_to=84/Boxing-v0_autoencoder.pt")
    parser.add_argument("--meta_hdim",type=int,default=128)

   
    args = parser.parse_args()
    
    sys.argv = tmp_argv

basename="meta_vqvae"
def mkstr(key):
    d = args.__dict__
    return "=".join([key,str(d[key])])


args.dataset = ("Small_" if test_notebook else "") + args.dataset
output_dirname = ("notebook_" if test_notebook else "") + "_".join([basename,mkstr("dataset"),mkstr("lr"),mkstr("meta_hdim"),mkstr("batch_size")])

saved_model_dir = os.path.join(args.savedir,"models/%s/%s" % (basename,output_dirname))

log_dir = os.path.join(args.savedir,'.%s_logs/%s'%(basename,output_dirname))
sample_dir = os.path.join(args.savedir,'samples/%s/%s'%(basename,output_dirname))
writer = SummaryWriter(log_dir=log_dir)

for dir_ in [saved_model_dir, args.data_dir, log_dir, sample_dir]:
    if not os.path.exists(dir_):
        os.makedirs(dir_)

DEVICE = torch.device('cuda') # torch.device('cpu'

def train():
    start_time = time.time()
    update_freq = int(args.batch_size / args.max_batch_size)
    train_loss = []
    for batch_idx, (x, _) in enumerate(train_loader):
        #x = torch.randint(0,255,(batch_size,3,84,84))
        do_update = batch_idx % update_freq == 0
        start_time = time.time()
        x = x.to("cuda")

        if do_update:
            opt.zero_grad()


        Z_tild,Z, zq, ze = model(x)
        zq.retain_grad()

        # PixelCNN loss
        loss_recons = F.cross_entropy(Z_tild,Z.long())
        loss_recons.backward(retain_graph=True)

        # Straight-through estimator
        ze.backward(zq.grad, retain_graph=True)

        # Vector quantization objective
    #     if do_update:
    #         vq.embedding.zero_grad()
        loss_vq = F.mse_loss(zq, ze.detach().squeeze())
        loss_vq.backward(retain_graph=True)

        # Commitment objective
        loss_commit = args.lambda_ * F.mse_loss(ze.squeeze(), zq.detach())
        loss_commit.backward()
        if do_update: # and batch_idx != 0:
            opt.step()

        train_loss.append(to_scalar([loss_recons, loss_vq]))
        #print(float(loss_recons.data))
    #     if (batch_idx + 1) % PRINT_INTERVAL == 0:
    #         print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
    #             batch_idx * len(x), len(train_loader.dataset),
    #             PRINT_INTERVAL * batch_idx / len(train_loader),
    #             np.asarray(train_loss)[-PRINT_INTERVAL:].mean(0),
    #             time.time() - start_time
    #         ))
    print("it took ",time.time() - start_time, " seconds")

    return np.mean(np.asarray(train_loss),axis=0)

def test():
    start_time = time.time()
    val_loss = []
    it = test_loader.__iter__()
    for batch_idx, (x, _) in enumerate(it):
#         x = torch.randint(0,255,(batch_size,3,84,84))
        x = x.to("cuda")
        Z_tild,Z, zq, ze = model(x)
        loss_recons = F.cross_entropy(Z_tild,Z.long())
        loss_vq = F.mse_loss(zq, ze.detach().squeeze())
        val_loss.append(to_scalar([loss_recons, loss_vq]))

#     print('\nValidation Completed!\tLoss: {} Time: {:5.3f}'.format(
#         np.asarray(val_loss).mean(0),
#         time.time() - start_time
#     ))
    return np.asarray(val_loss).mean(0)

def generate_samples():
    it = test_loader.__iter__()
    for i in range(2):
        x, _ = it.next()
        #x = torch.randint(0,255,(batch_size,3,84,84))
        x = x.to("cuda")
        _,_, zq, _ = model(x)

        ZQ = model.decoder.sample(zq)

        ZQ.size()

        X_tild, _ = model.VQVAE.decode(ZQ.squeeze())

        x_cat = torch.cat([x, X_tild], 0)
        images = (x_cat.cpu().data + 1) / 2


        save_image(
            images,
            sample_dir + '/reconstructions_metavqvae_%s_%i.png'%(args.dataset,i),
            nrow=args.max_batch_size
        )

        im_grid = make_grid(images, nrow=args.max_batch_size)
        writer.add_image("rec_" + str(i),  im_grid)



if __name__ == "__main__":

    model = MetaPixelVQVAE(args.weights_path, args.meta_hdim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader, test_loader = get_data_loaders(data_dir=args.data_dir, dataset=args.dataset, 
    max_batch_size=args.max_batch_size, num_workers=args.num_workers)
    
    BEST_LOSS = 999
    LAST_SAVED = -1
    for epoch in range(args.epochs):
        print("Epoch {}:".format(epoch))
        loss_rec,loss_vq = train()
        print("train loss: ",loss_rec)
        writer.add_scalar("loss_rec",loss_rec,global_step=epoch)
        writer.add_scalar("loss_vq",loss_vq,global_step=epoch)
        val_loss_rec,val_loss_vq = test()
        print("val loss: ",val_loss_rec)
        writer.add_scalar("val/loss_rec",val_loss_rec,global_step=epoch)
        writer.add_scalar("val/loss_vq",val_loss_vq,global_step=epoch)

        if val_loss_rec <= BEST_LOSS:
            BEST_LOSS = val_loss_rec
            LAST_SAVED = epoch
            print("Saving model!")
            torch.save(model.state_dict(), saved_model_dir + '/%s_%s_autoencoder.pt'%(args.dataset,basename))
        else:
            print("Not saving model! Last saved: {}".format(LAST_SAVED))

        generate_samples()


