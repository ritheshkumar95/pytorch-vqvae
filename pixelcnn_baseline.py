import torch
import torch.nn as nn
from torchvision import datasets, transforms
from modules import GatedPixelCNN
import numpy as np
from torchvision.utils import save_image
import time


BATCH_SIZE = 32
N_EPOCHS = 100
PRINT_INTERVAL = 100
ALWAYS_SAVE = True
DATASET = 'FashionMNIST'  # CIFAR10 | MNIST | FashionMNIST
NUM_WORKERS = 4

IMAGE_SHAPE = (28, 28)  # (32, 32) | (28, 28)
INPUT_DIM = 3  # 3 (RGB) | 1 (Grayscale)
K = 256
DIM = 64
N_LAYERS = 15
LR = 3e-4


train_loader = torch.utils.data.DataLoader(
    eval('datasets.'+DATASET)(
        '../data/{}/'.format(DATASET), train=True, download=True,
        transform=transforms.ToTensor(),
    ), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    eval('datasets.'+DATASET)(
        '../data/{}/'.format(DATASET), train=False,
        transform=transforms.ToTensor(),
    ), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

model = GatedPixelCNN(K, DIM, N_LAYERS).cuda()
criterion = nn.CrossEntropyLoss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=LR)


def train():
    train_loss = []
    for batch_idx, (x, label) in enumerate(train_loader):
        start_time = time.time()
        x = (x[:, 0] * (K-1)).long().cuda()
        label = label.cuda()

        # Train PixelCNN with images
        logits = model(x, label)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(
            logits.view(-1, K),
            x.view(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append(loss.item())

        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                PRINT_INTERVAL * batch_idx / len(train_loader),
                np.asarray(train_loss)[-PRINT_INTERVAL:].mean(0),
                time.time() - start_time
            ))


def test():
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for batch_idx, (x, label) in enumerate(test_loader):
            x = (x[:, 0] * (K-1)).long().cuda()
            label = label.cuda()

            logits = model(x, label)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(
                logits.view(-1, K),
                x.view(-1)
            )
            val_loss.append(loss.item())

    print('Validation Completed!\tLoss: {} Time: {}'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)


def generate_samples():
    label = torch.arange(10).expand(10, 10).contiguous().view(-1)
    label = label.long().cuda()

    x_tilde = model.generate(label, shape=IMAGE_SHAPE, batch_size=100)
    images = x_tilde.cpu().data.float() / (K - 1)

    save_image(
        images[:, None],
        'samples/pixelcnn_baseline_samples_{}.png'.format(DATASET),
        nrow=10
    )


BEST_LOSS = 999
LAST_SAVED = -1
for epoch in range(1, N_EPOCHS):
    print("\nEpoch {}:".format(epoch))
    train()
    cur_loss = test()

    if ALWAYS_SAVE or cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch

        print("Saving model!")
        torch.save(model.state_dict(), 'models/{}_pixelcnn.pt'.format(DATASET))
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))

    generate_samples()
