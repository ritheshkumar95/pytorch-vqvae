
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
from torch.utils.data import DataLoader, Dataset

env_list = ["Skiing-v0","Tennis-v0","Pong-v0","Kangaroo-v0","CrazyClimber-v0","Breakout-v0","Bowling-v0"]

env_name = "Pong-v0"

BATCH_SIZE = 32
N_EPOCHS = 100
PRINT_INTERVAL = 100
DATASET =  "Pong-v0"#'CIFAR10'  # CIFAR10 | MNIST | FashionMNIST
NUM_WORKERS = 4
num_frames=60000
resize_to=(128,128)
INPUT_DIM = 3  # 3 (RGB) | 1 (Grayscale)
DIM = 256
K = 512
LAMDA = 1
LR = 3e-4

DEVICE = torch.device('cuda') # torch.device('cpu')

data_dir = "../data"

for dir_ in ["models", "samples", data_dir]:
    if not os.path.exists(dir_):
        os.mkdir(dir_)

def convert_frame(state, new_shape=(64,64), cuda=True):
    state = Image.fromarray(state, 'RGB')

    if new_shape != (-1,-1):
        transforms = Compose([Resize(new_shape)])
        state = transforms(state)

    return np.asarray(state)

def save_frames(env_name,num_frames=60000, resize_to=(128,128)):
    env = gym.make(env_name)
    state = env.reset()

    frames = np.zeros((num_frames,*resize_to,3),dtype=np.uint8)

    actions = np.zeros((num_frames,),dtype=np.uint8)

    t0 = time.time()
    for i in range(num_frames):
        action = env.action_space.sample()
        state,_,_,_ = env.step(action)
        framen = convert_frame(state,new_shape=resize_to)
        frames[i] = framen
        actions[i] = action
    print(time.time() - t0)

    fn = os.path.join(data_dir,env_name + ".npz")
    fna = os.path.join(data_dir,env_name + "actions_" + ".npz")
    np.savez(fn,frames=frames)
    np.savez(fna,actions=actions)

fn = os.path.join(data_dir,DATASET + ".npz")
fna = os.path.join(data_dir,env_name + "actions_" + ".npz")

if not os.path.exists(fn):
    save_frames(DATASET,num_frames=num_frames,resize_to=resize_to)

class FrameDataset(Dataset):
    def __init__(self, fn,fna, transform=None, just_frames=False):
        self.just_frames = just_frames
        frame_shape = (num_frames,*resize_to,3)
        self.frame_data = np.memmap(fn, dtype='uint8', mode='r', shape=frame_shape)
        if not self.just_frames:
            action_shape = (num_frames,)
            self.action_data = np.memmap(fna, dtype='uint8', mode='r', shape=action_shape)
        
        self.transform = transform

    def __len__(self):
        return self.frame_data.shape[0]

    def __getitem__(self, idx):
        frame = self.frame_data[idx]
        if self.transform:
            frame = self.transform(frame)
        if self.just_frames:
            return frame
        else:
            action = self.action_data[idx]
            return frame,action

transforms = Compose([ToTensor(),Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
rds = FrameDataset(fn,fna,transform=transforms)
train_loader = DataLoader(rds,batch_size=BATCH_SIZE,shuffle=True) # hopefully this will encourage vae to learn curves


model = AutoEncoder(INPUT_DIM, DIM, K).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

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
        loss_commit = LAMDA * F.mse_loss(z_e_x, z_q_x.detach())
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

BEST_LOSS = 999
LAST_SAVED = -1
for epoch in range(1, N_EPOCHS):
    print("Epoch {}:".format(epoch))
    train()
    cur_loss, _ = test()

    if cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch
        print("Saving model!")
        torch.save(model.state_dict(), 'models/{}_autoencoder.pt'.format(DATASET))
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))

    generate_samples()


