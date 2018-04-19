import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from modules import AutoEncoder, to_scalar
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import time


kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        '../data/cifar10/', train=True, download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    ), batch_size=64, shuffle=False, **kwargs
)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        '../data/cifar10/', train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
    ), batch_size=32, shuffle=False, **kwargs
)
test_data = list(test_loader)

model = AutoEncoder().cuda()
opt = torch.optim.Adam(model.parameters(), lr=3e-4)


def train(epoch):
    train_loss = []
    for batch_idx, (data, _) in enumerate(train_loader):
        start_time = time.time()
        x = Variable(data, requires_grad=False).cuda()

        opt.zero_grad()

        x_tilde, z_e_x, z_q_x = model(x)
        z_q_x.retain_grad()

        loss_recons = F.l1_loss(x_tilde, x)
        loss_recons.backward(retain_graph=True)

        # Straight-through estimator
        z_e_x.backward(z_q_x.grad, retain_graph=True)

        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        loss_vq.backward(retain_graph=True)

        # Commitment objective
        loss_commit = 0.25 * F.mse_loss(z_e_x, z_q_x.detach())
        loss_commit.backward()
        opt.step()

        train_loss.append(to_scalar([loss_recons, loss_vq]))

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                np.asarray(train_loss)[-100:].mean(0),
                time.time() - start_time
            ))


def test():
    x = Variable(test_data[0][0]).cuda()
    x_tilde, _, _ = model(x)
    x_tilde = (x_tilde+1)/2
    x = (x+1)/2

    x_cat = torch.cat([x, x_tilde], 0)
    images = x_cat.cpu().data
    save_image(images, './sample_cifar.png', nrow=8)


for i in range(100):
    train(i)
    test()
