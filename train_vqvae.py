
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from modules import AutoEncoder, to_scalar
import numpy as np
from torchvision.utils import save_image
import time
import os
import argparse
from tensorboardX import SummaryWriter
from atari_data import get_data_loaders


# In[6]:


import sys
if __name__ == "__main__":
    test_notebook = False
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [""]
        test_notebook= True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default="Small_Boxing-v0")
    parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--K",type=int,default=512)
    parser.add_argument("--dim",type=int,default=256)
    parser.add_argument("--lambda_",type=int,default=1)
    parser.add_argument("--decoder",type=str,choices=["deconv","pixelcnn"], default="deconv")
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--max_batch_size",type=int,default=4)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--num_frames",type=int,default=60000)
    parser.add_argument("--resize_to",type=int,default=84) # paper is 100
    parser.add_argument("--data_dir",type=str,default="../data")
    parser.add_argument("--savedir",type=str,default=".")
   
    args = parser.parse_args()


# In[7]:


update_freq = int(args.batch_size / args.max_batch_size)


# In[4]:


basename="vqvae"
def mkstr(key):
    d = args.__dict__
    return "=".join([key,str(d[key])])

output_dirname = "_".join([basename,mkstr("dataset"),mkstr("lr"),mkstr("batch_size")])

if test_notebook:
    output_dirname = "notebook_" + output_dirname
    
saved_model_dir = os.path.join(args.savedir,("models/%s" % output_dirname))

log_dir = os.path.join(args.savedir,'.%s_logs/%s'%(basename,output_dirname))
sample_dir = os.path.join(args.savedir,'%s_samples/%s'%(basename,output_dirname))
writer = SummaryWriter(log_dir=log_dir)

for dir_ in [saved_model_dir, args.data_dir, log_dir, sample_dir]:
    if not os.path.exists(dir_):
        os.makedirs(dir_)

env_list = ["Skiing-v0","Breakout-v0","Bowling-v0","Enduro-v0","UpNDown-v0","Boxing-v0"]

PRINT_INTERVAL = 100
INPUT_DIM = 3  # 3 (RGB) | 1 (Grayscale)
DEVICE = torch.device('cuda') # torch.device('cpu')


# In[9]:


train_loader, test_loader = get_data_loaders(data_dir=args.data_dir, dataset=args.dataset, 
max_batch_size=args.max_batch_size, num_workers=args.num_workers)


# In[ ]:


model = AutoEncoder(INPUT_DIM, args.dim, args.K).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

def train():
    train_loss = []
    for batch_idx, (x, _) in enumerate(train_loader):
        do_update = batch_idx % update_freq == 0
        start_time = time.time()
        x = x.to(DEVICE)

        if do_update:
            opt.zero_grad()

        x_tilde, z_e_x, z_q_x = model(x)
        
        z_q_x.retain_grad()

        loss_recons = F.mse_loss(x_tilde, x)
        loss_recons.backward(retain_graph=True)

        # Straight-through estimator
        z_e_x.backward(z_q_x.grad, retain_graph=True)

        # Vector quantization objective
        if do_update:
            model.embedding.zero_grad()
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        loss_vq.backward(retain_graph=True)

        # Commitment objective
        loss_commit = args.lambda_ * F.mse_loss(z_e_x, z_q_x.detach())
        loss_commit.backward()
        if do_update and batch_idx != 0:
            opt.step()

        train_loss.append(to_scalar([loss_recons, loss_vq]))

        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                PRINT_INTERVAL * batch_idx / len(train_loader),
                np.asarray(train_loss)[-PRINT_INTERVAL:].mean(0),
                time.time() - start_time
            ))
    
    return np.mean(np.asarray(train_loss),axis=0)

def test():
    start_time = time.time()
    val_loss = []
    for batch_idx, (x, _) in enumerate(test_loader):
        x = x.to(DEVICE)
        x_tilde, z_e_x, z_q_x = model(x)
        loss_recons = F.mse_loss(x_tilde, x)
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        val_loss.append(to_scalar([loss_recons, loss_vq]))

    print('\nValidation Completed!\tLoss: {} Time: {:5.3f}'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)


def generate_samples():
    x, _ = test_loader.__iter__().next()
    x = x[:32].to(DEVICE)
    x_tilde, _, _ = model(x)

    x_cat = torch.cat([x, x_tilde], 0)
    images = (x_cat.cpu().data + 1) / 2

    save_image(
        images,
        sample_dir + '/reconstructions_{}.png'.format(args.dataset),
        nrow=8
    )
    writer.add_image("rec",images)



if __name__ == "__main__":
    BEST_LOSS = 999
    LAST_SAVED = -1
    for epoch in range(1, args.epochs):
        print("Epoch {}:".format(epoch))
        loss = train()
        writer.add_scalar("loss_rec",loss[0],global_step=epoch)
        writer.add_scalar("loss_vq",loss[1],global_step=epoch)
        cur_loss, _ = test()

        if cur_loss <= BEST_LOSS:
            BEST_LOSS = cur_loss
            LAST_SAVED = epoch
            print("Saving model!")
            torch.save(model.state_dict(), saved_model_dir + '/{}_autoencoder.pt'.format(args.dataset))
        else:
            print("Not saving model! Last saved: {}".format(LAST_SAVED))

        generate_samples()

