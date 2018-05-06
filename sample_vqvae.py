
# coding: utf-8

# In[1]:


from torchvision.transforms import Compose,Normalize,Resize,ToTensor

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from modules import AutoEncoder, GatedPixelCNN, to_scalar
import numpy as np
from torchvision.utils import save_image
import time
import argparse
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tensorboardX import SummaryWriter

import sys
if __name__ == "__main__":
    test_notebook = False
    if "ipykernel_launcher" in sys.argv[0]:
        sys.argv = [""]
        test_notebook= True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default="Boxing-v0")
    parser.add_argument("--lr",type=float,default=3e-4)
    parser.add_argument("--K",type=int,default=512)
    parser.add_argument("--dim",type=int,default=256)
    parser.add_argument("--lambda_",type=int,default=1)
    parser.add_argument("--prior",type=str,choices=["uniform","pixelcnn"],default="pixelcnn")
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--num_workers",type=int,default=4)
    parser.add_argument("--num_frames",type=int,default=60000)
    parser.add_argument("--resize_to",type=int,default=84) # paper is 100
    parser.add_argument("--data_dir",type=str,default="../data")
    parser.add_argument("--savedir",type=str,default=".")
    parser.add_argument("--latent_shape",type=int,default=21)
    parser.add_argument("--max_batch_size",type=int,default=4)
    parser.add_argument("--weights_dir", type=str, default="./models/vqvae_dataset=Boxing-v0_lr=0.0003_K=512_dim=256_lambda_=1_decoder=deconv_batch_size=32_epochs=100_num_workers=4_num_frames=60000_resize_to=84")
   
    args = parser.parse_args()



update_freq = int(args.batch_size / args.max_batch_size)

basename="vqvae_test_time"
def mkstr(key):
    d = args.__dict__
    return "=".join([key,str(d[key])])
output_dirname = basename + "_" + mkstr("lr") + "_vae_params" + "_".join(os.path.basename(args.weights_dir).split("_")[1:]) 








# In[2]:


output_dirname


# In[2]:


if test_notebook:
    output_dirname = "notebook_" + output_dirname
    
saved_model_dir = os.path.join(args.savedir,("models/%s" % output_dirname))

log_dir = os.path.join(args.savedir,'.%s_logs/%s'%(basename,output_dirname))
sample_dir = os.path.join(args.savedir,'%s_samples/%s'%(basename,output_dirname))
writer = SummaryWriter(log_dir=log_dir)

for dir_ in [saved_model_dir, args.data_dir, log_dir, sample_dir]:
    if not os.path.exists(dir_):
        os.makedirs(dir_)

BATCH_SIZE = 32
N_EPOCHS = 100
PRINT_INTERVAL = 100
ALWAYS_SAVE = True

NUM_WORKERS = 4

 # (8, 8) -> 32x32 images, (7, 7) -> 28x28 images
INPUT_DIM = 3  # 3 (RGB) | 1 (Grayscale)
DIM = 128
VAE_DIM = 256
N_LAYERS = 15
K = 512
LR = 3e-4

DEVICE = torch.device('cuda') # torch.device('cpu')

train_loader, test_loader = get_data_loaders(data_dir=args.data_dir, dataset=args.dataset, 
max_batch_size=args.max_batch_size, num_workers=args.num_workers)


autoencoder = AutoEncoder(INPUT_DIM, args.dim, args.K).to(DEVICE)
autoencoder.load_state_dict(
    torch.load(args.weights_dir + '/{}_autoencoder.pt'.format(args.dataset))
)
autoencoder.eval()

model = GatedPixelCNN(args.K, args.dim, N_LAYERS).to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

def train():
    train_loss = []
    for batch_idx, (x, label) in enumerate(train_loader):
        do_update = batch_idx % update_freq == 0
        start_time = time.time()
        x = x.to(DEVICE)
        label = label.to(DEVICE)

        # Get the latent codes for image x
        latents, _ = autoencoder.encode(x)

        # Train PixelCNN with latent codes
        latents = latents.detach()
        logits = model(latents, label)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(
            logits.view(-1, K),
            latents.view(-1)
        )

        if do_update:
            opt.zero_grad()
        loss.backward()
        if do_update and batch_idx != 0:
            opt.step()

        train_loss.append(to_scalar(loss))

        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                PRINT_INTERVAL * batch_idx / len(train_loader),
                np.asarray(train_loss)[-PRINT_INTERVAL:].mean(0),
                time.time() - start_time
            ))
    return np.mean(np.asarray(train_loss))


def test():
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x = x.to(DEVICE)
            label = label.to(DEVICE)

            latents, _ = autoencoder.encode(x)
            logits = model(latents.detach(), label)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(
                logits.view(-1, K),
                latents.view(-1)
            )
            val_loss.append(to_scalar(loss))

    print('Validation Completed!\tLoss: {} Time: {}'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)


def generate_samples():
    label = torch.arange(2).expand(2, 2).contiguous().view(-1)
    label = label.to(device=DEVICE, dtype=torch.int64)

    latents = model.generate(label, shape=(args.latent_shape,args.latent_shape), batch_size=4)
    x_tilde, _ = autoencoder.decode(latents)
    print(x_tilde.size())
    images = (x_tilde.cpu().data + 1) / 2

    save_image(
        images,
        sample_dir + '/samples_{}.png'.format(args.dataset),
        nrow=10
    )


def generate_reconstructions():
    x, _ = test_loader.__iter__().next()
    x = x[:32].to(DEVICE)

    latents, _ = autoencoder.encode(x)
    x_tilde, _ = autoencoder.decode(latents)
    x_cat = torch.cat([x, x_tilde], 0)
    images = (x_cat.cpu().data + 1) / 2

    save_image(
        images,
        'samples/reconstructions_{}.png'.format(args.dataset),
        nrow=8
    )

BEST_LOSS = 999
LAST_SAVED = -1
generate_reconstructions()
for epoch in range(1, args.epochs):
    print("\nEpoch {}:".format(epoch))
    loss = train()
    writer.add_scalar("train/autoreg_prior_loss",loss,global_step=epoch)
    
    cur_loss = test()
    writer.add_scalar("val/autoreg_prior_loss",cur_loss,global_step=epoch)
    if ALWAYS_SAVE or cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch

        print("Saving model!")
        torch.save(model.state_dict(), saved_model_dir + '/{}_pixelcnn.pt'.format(args.dataset))
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))

    generate_samples()

