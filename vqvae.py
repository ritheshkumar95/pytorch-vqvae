import numpy as np
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.distributions.normal import Normal

from modules import VectorQuantizedAE, to_scalar


BATCH_SIZE = 32
N_EPOCHS = 100
PRINT_INTERVAL = 100
DATASET = 'CIFAR10'  # CIFAR10 | MNIST | FashionMNIST
NUM_WORKERS = 4

INPUT_DIM = 3  # 3 (RGB) | 1 (Grayscale)
DIM = 256
K = 512
LAMDA = 1
LR = 3e-4

DEVICE = torch.device('cuda')  # torch.device('cpu')


# Create directories if they don't exist
Path('models').mkdir(exist_ok=True)
Path('samples').mkdir(exist_ok=True)

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
        transform=preproc_transform,
    ), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

model = VectorQuantizedAE(INPUT_DIM, DIM, K).to(DEVICE)
print(model)
opt = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)


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
        model.codebook.zero_grad()
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        loss_vq.backward(retain_graph=True)

        # Commitment objective
        loss_commit = LAMDA * F.mse_loss(z_e_x, z_q_x.detach())
        loss_commit.backward()
        opt.step()

        nll = -Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(x)
        log_px = nll.mean().item() - np.log(128) + np.log(K)
        log_px /= np.log(2)

        train_loss.append(
            [log_px] + to_scalar([loss_recons, loss_vq])
        )

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
        'samples/vqvae_reconstructions_{}.png'.format(DATASET),
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
        torch.save(model.state_dict(), 'models/{}_vqvae.pt'.format(DATASET))
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))

    generate_samples()
