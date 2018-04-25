import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from modules import AutoEncoder, to_scalar
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import time


BATCH_SIZE = 128
NUM_WORKERS = 4
LR = 2e-4
K = 256
LAMDA = 0.25
PRINT_INTERVAL = 100
N_EPOCHS = 100


preproc_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        '../data/cifar10/', train=True, download=True,
        transform=preproc_transform,
    ), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        '../data/cifar10/', train=False,
        transform=preproc_transform
    ), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

model = AutoEncoder(K).cuda()
opt = torch.optim.Adam(model.parameters(), lr=LR)


def train():
    train_loss = []
    for batch_idx, (data, _) in enumerate(train_loader):
        start_time = time.time()
        x = Variable(data, requires_grad=False).cuda()

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

        if (batch_idx + 1) % 100 == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                np.asarray(train_loss)[-100:].mean(0),
                time.time() - start_time
            ))


def test():
    start_time = time.time()
    val_loss = []
    for batch_idx, (data, _) in enumerate(test_loader):
        x = Variable(data, volatile=True).cuda()
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
    x = Variable(x[:32]).cuda()
    x_tilde, _, _ = model(x)

    x_cat = torch.cat([x, x_tilde], 0)
    images = x_cat.cpu().data
    save_image(images, './sample_cifar.png', nrow=8)


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
        torch.save(model.state_dict(), 'best_autoencoder.pt')
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))

    generate_samples()
