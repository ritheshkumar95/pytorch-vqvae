
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from modules import AutoEncoder, to_scalar
import numpy as np
from torchvision.utils import save_image
import time
import os
import gym
from PIL import Image
from torchvision.transforms import Compose,Normalize,Resize,ToTensor
from matplotlib import pyplot as plt
#%matplotlib inline
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import argparse
from tensorboardX import SummaryWriter


# In[2]:


import sys
if __name__ == "__main__":
    test_notebook = False
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [""]
        test_notebook= True


# In[17]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=4)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--num_frames",type=int,default=60000)
    parser.add_argument("--resize_to",type=int,default=128) # paper is 100
    parser.add_argument("--data_dir",type=str,default="../data")
    parser.add_argument("--savedir",type=str,default=".")
    parser.add_argument("--dataset",type=str,default="Bowling-v0")
    parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--K",type=int,default=512)
    parser.add_argument("--dim",type=int,default=256)
    parser.add_argument("--lambda_",type=int,default=1)
    args = parser.parse_args()


# In[18]:


basename="vqvae"
def mkstr(key):
    d = args.__dict__
    return "=".join([key,str(d[key])])

output_dirname = "_".join([basename,*[mkstr(k) for k in args.__dict__ if "dir" not in k]])


# In[19]:


if test_notebook:
    output_dirname = "notebook_" + output_dirname
    
saved_model_dir = os.path.join(args.savedir,("models/%s" % output_dirname))

log_dir = os.path.join(args.savedir,'.%s_logs/%s'%(basename,output_dirname))

writer = SummaryWriter(log_dir=log_dir)

for dir_ in [saved_model_dir, args.data_dir, log_dir]:
    if not os.path.exists(dir_):
        os.makedirs(dir_)


# In[20]:


env_list = ["Skiing-v0","Tennis-v0","Pong-v0","Kangaroo-v0","CrazyClimber-v0","Breakout-v0","Bowling-v0"]

PRINT_INTERVAL = 100
INPUT_DIM = 3  # 3 (RGB) | 1 (Grayscale)
DEVICE = torch.device('cuda') # torch.device('cpu')



def convert_frame(state, new_shape=(64,64), cuda=True):
    state = Image.fromarray(state, 'RGB')

    if new_shape != (-1,-1):
        transforms = Compose([Resize(new_shape)])
        state = transforms(state)

    return state #Image


# In[21]:


def save_frames(env_name,imdir, num_frames=60000, resize_to=128):
    env = gym.make(env_name)
    state = env.reset()
    done = False
    t0 = time.time()
    for i in range(num_frames):
        action = env.action_space.sample()
        if done:
            state = env.reset()
            done = False
        else:
            state,r,done,_ = env.step(action)
        frame = convert_frame(state,new_shape=(resize_to, resize_to))
        frame.save(os.path.join(imdir,str(i) + ".jpg"))
    
    print(time.time() - t0)


# In[22]:


imdir = os.path.join(args.data_dir,args.dataset,str(0))
if not os.path.exists(imdir):
    os.makedirs(imdir)
    save_frames(args.dataset,imdir=imdir, num_frames=args.num_frames,resize_to=args.resize_to)


# In[23]:


# class FrameDataset(Dataset):
#     def __init__(self, fn,transform=None, just_frames=False):
#         self.just_frames = just_frames
#         self.frames, self.actions = torch.load(fn)

        
#         self.transform = transform

#     def __len__(self):
#         return len(self.frames)

#     def __getitem__(self, idx):
#         frame = self.frames[idx]
#         if self.transform:
#             frame = self.transform(frame)
#         if self.just_frames:
#             return frame
#         else:
#             action = self.actions[idx]
#             return frame,action


# In[24]:


transforms = Compose([ToTensor(),Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
rds = ImageFolder(os.path.dirname(imdir),transform=transforms)


# In[25]:


train_loader = DataLoader(rds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers) 


# In[26]:


model = AutoEncoder(INPUT_DIM, args.dim, args.K).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

def train():
    train_loss = []
    for batch_idx, (x, _) in enumerate(train_loader):
        start_time = time.time()
        x = x.to(DEVICE)

        opt.zero_grad()

        x_tilde, z_e_x, z_q_x = model(x)
        z_q_x.retain_grad()

        loss_recons = F.mse_loss(x_tilde, x)
        loss_recons.backward(retain_graph=True)

        # Straight-through estimator
        z_e_x.backward(z_q_x.grad, retain_graph=True)

        # Vector quantization objective
        model.embedding.zero_grad()
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        loss_vq.backward(retain_graph=True)

        # Commitment objective
        loss_commit = args.lambda_ * F.mse_loss(z_e_x, z_q_x.detach())
        loss_commit.backward()
        opt.step()

        train_loss.append(to_scalar([loss_recons, loss_vq]))

        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                PRINT_INTERVAL * batch_idx / len(train_loader),
                np.asarray(train_loss)[-PRINT_INTERVAL:].mean(0),
                time.time() - start_time
            ))


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
        'samples/reconstructions_{}.png'.format(DATASET),
        nrow=8
    )


# In[27]:


if __name__ == "__main__":
    BEST_LOSS = 999
    LAST_SAVED = -1
    for epoch in range(1, args.epochs):
        print("Epoch {}:".format(epoch))
        train()
        cur_loss, _ = test()

        if cur_loss <= BEST_LOSS:
            BEST_LOSS = cur_loss
            LAST_SAVED = epoch
            print("Saving model!")
            torch.save(model.state_dict(), 'models/{}_autoencoder.pt'.format(args.dataset))
        else:
            print("Not saving model! Last saved: {}".format(LAST_SAVED))

        generate_samples()

