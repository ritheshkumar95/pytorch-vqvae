
# coding: utf-8

# In[2]:


import os
import gym
from PIL import Image
import time
from torchvision.transforms import Compose,Normalize,Resize,ToTensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


# In[3]:


def get_dirs(data_dir, dataset):
    basedir = os.path.join(data_dir,dataset)
    tr_dir = os.path.join(basedir, "train",str(0))
    val_dir = os.path.join(basedir,"val",str(0))
    test_dir = os.path.join(basedir,"test",str(0))
    return tr_dir, val_dir, test_dir


# In[4]:


def convert_frame(state, new_shape=(64,64), cuda=True):
    state = Image.fromarray(state, 'RGB')

    if new_shape != (-1,-1):
        transforms = Compose([Resize(new_shape)])
        state = transforms(state)

    return state #Image

def save_frames(env_name,tr_dir,val_dir,test_dir, num_frames=60000, resize_to=128):
    for dir_ in [tr_dir, val_dir, test_dir]:
        os.makedirs(dir_)
    env = gym.make(env_name)
    state = env.reset()
    done = False
    t0 = time.time()
    imdir = tr_dir
    for i in range(num_frames):
        if i == int(0.8 * num_frames):
            imdir = val_dir
        if i== int(0.9* num_frames):
            imdir = test_dir
        action = env.action_space.sample()
        if done:
            state = env.reset()
            done = False
        else:
            state,r,done,_ = env.step(action)
        frame = convert_frame(state,new_shape=(resize_to, resize_to))
        frame.save(os.path.join(imdir,str(i) + ".jpg"))
    
    print(time.time() - t0)


# In[16]:


def get_data_loaders(data_dir = "../data",dataset = "Small_Boxing-v0",max_batch_size = 4,num_workers =4):

    tr_dir, val_dir, test_dir = get_dirs(data_dir, dataset)
    save_out_frames = False

    for dir_ in [tr_dir, val_dir, test_dir]:
        if not os.path.exists(dir_):
            save_out_frames = True
    if save_out_frames:
        print("saving the raw frames. you gotta wait, dood")
        save_frames(dataset,tr_dir,val_dir,test_dir, num_frames=num_frames,resize_to=resize_to)
    else:
        print("Already Saved")

    transforms = Compose([ToTensor(),Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

    train_set = ImageFolder(os.path.dirname(tr_dir),transform=transforms)
    train_loader = DataLoader(train_set,batch_size=max_batch_size,shuffle=True,num_workers=num_workers) 

    test_set = ImageFolder(os.path.dirname(test_dir),transform=transforms)
    test_loader = DataLoader(test_set,batch_size=max_batch_size,shuffle=True,num_workers=num_workers) 
    return train_loader, test_loader

