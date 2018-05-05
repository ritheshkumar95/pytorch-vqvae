import numpy as np
import time

import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal

from torchvision import datasets, transforms
from torchvision.utils import save_image

from modules import VAE


BATCH_SIZE = 32
N_EPOCHS = 100
PRINT_INTERVAL = 500
DATASET = 'FashionMNIST'  # CIFAR10 | MNIST | FashionMNIST
NUM_WORKERS = 4

INPUT_DIM = 1
DIM = 256
Z_DIM = 128
LR = 1e-3


preproc_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    eval('datasets.'+DATASET)(
        '../data/{}/'.format(DATASET), train=True, download=True,
        transform=preproc_transform,
    ), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    eval('datasets.'+DATASET)(
        '../data/{}/'.format(DATASET), train=False,
        transform=preproc_transform
    ), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

model = VAE(INPUT_DIM, DIM, Z_DIM).cuda()
print(model)
opt = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)


def train():
    train_loss = []
    model.train()
    for batch_idx, (x, _) in enumerate(train_loader):
        start_time = time.time()
        x = x.cuda()

        x_tilde, kl_d = model(x)
        loss_recons = F.mse_loss(x_tilde, x, size_average=False) / x.size(0)
        loss = loss_recons + kl_d

        nll = -Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(x)
        log_px = nll.mean().item() - np.log(128) + kl_d.item()
        log_px /= np.log(2)

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append([log_px, loss.item()])

        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {:5.3f} ms/batch'.format(
                batch_idx * len(x), len(train_loader.dataset),
                PRINT_INTERVAL * batch_idx / len(train_loader),
                np.asarray(train_loss)[-PRINT_INTERVAL:].mean(0),
                1000 * (time.time() - start_time)
            ))


def test():
    start_time = time.time()
    val_loss = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.cuda()
            x_tilde, kl_d = model(x)
            loss_recons = F.mse_loss(x_tilde, x, size_average=False) / x.size(0)
            loss = loss_recons + kl_d
            val_loss.append(loss.item())

    print('\nValidation Completed!\tLoss: {:5.4f} Time: {:5.3f} s'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)


def generate_reconstructions():
    model.eval()
    x, _ = test_loader.__iter__().next()
    x = x[:32].cuda()
    x_tilde, kl_div = model(x)

    x_cat = torch.cat([x, x_tilde], 0)
    images = (x_cat.cpu().data + 1) / 2

    save_image(
        images,
        'samples/vae_reconstructions_{}.png'.format(DATASET),
        nrow=8
    )


def generate_samples():
    model.eval()
    z_e_x = torch.randn(64, Z_DIM, 1, 1).cuda()
    x_tilde = model.decoder(z_e_x)

    images = (x_tilde.cpu().data + 1) / 2

    save_image(
        images,
        'samples/vae_samples_{}.png'.format(DATASET),
        nrow=8
    )


BEST_LOSS = 99999
LAST_SAVED = -1
for epoch in range(1, N_EPOCHS):
    print("Epoch {}:".format(epoch))
    train()
    cur_loss = test()

    if cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch
        print("Saving model!")
        torch.save(model.state_dict(), 'models/{}_vae.pt'.format(DATASET))
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))

    generate_reconstructions()
    generate_samples()
