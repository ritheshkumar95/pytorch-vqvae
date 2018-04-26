import torch
import torch.nn as nn
from torchvision import datasets, transforms
from modules import AutoEncoder, GatedPixelCNN, to_scalar
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import time


BATCH_SIZE = 64
NUM_WORKERS = 4
LR = 3e-4
K = 512
DIM = 64
N_LAYERS = 15
PRINT_INTERVAL = 100
N_EPOCHS = 100
ALWAYS_SAVE = True


preproc_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

autoencoder = AutoEncoder(K).cuda()
autoencoder.load_state_dict(
    torch.load('best_autoencoder.pt')
)
autoencoder.eval()

model = GatedPixelCNN(K, DIM, N_LAYERS).cuda()
criterion = nn.CrossEntropyLoss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=LR)


def train():
    train_loss = []
    for batch_idx, (data, _) in enumerate(train_loader):
        start_time = time.time()
        x = Variable(data, volatile=True).cuda()

        # Get the latent codes for image x
        latents, _ = autoencoder.encode(x)

        # Train PixelCNN with latent codes
        latents = Variable(latents.data)
        logits = model(latents)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(
            logits.view(-1, K),
            latents.view(-1)
        )

        opt.zero_grad()        
        loss.backward()
        opt.step()

        train_loss.append(to_scalar(loss))

        if (batch_idx + 1) % 100 == 0:
            print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
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
        latents, _ = autoencoder.encode(x)
        latents = Variable(latents.data, volatile=True)
        logits = model(latents)
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
    latents = model.generate()
    x_tilde, _ = autoencoder.decode(latents)
    images = (x_tilde.cpu().data + 1) / 2
    save_image(images, './sample_pixelcnn_cifar.png', nrow=8)


def generate_reconstructions():
    x, _ = test_loader.__iter__().next()
    x = Variable(x[:32]).cuda()
    latents, _ = autoencoder.encode(x)
    x_tilde, _ = autoencoder.decode(latents)
    x_cat = torch.cat([x, x_tilde], 0)
    images = (x_cat.cpu().data + 1) / 2
    save_image(images, './sample_cifar.png', nrow=8)


BEST_LOSS = 999
LAST_SAVED = -1
generate_reconstructions()
for epoch in range(1, N_EPOCHS):
    print("\nEpoch {}:".format(epoch))
    train()
    cur_loss = test()

    if ALWAYS_SAVE or cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch

        print("Saving model!")
        torch.save(model.state_dict(), 'best_pixelcnn.pt')
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))

    generate_samples()
