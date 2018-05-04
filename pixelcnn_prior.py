import torch
import torch.nn as nn
from torchvision import datasets, transforms
from modules import VectorQuantizedVAE, GatedPixelCNN
import numpy as np
from torchvision.utils import save_image
import time


BATCH_SIZE = 32
N_EPOCHS = 100
PRINT_INTERVAL = 100
ALWAYS_SAVE = True
DATASET = 'MNIST'  # CIFAR10 | MNIST | FashionMNIST
NUM_WORKERS = 4

LATENT_SHAPE = (7, 7)  # (8, 8) -> 32x32 images, (7, 7) -> 28x28 images
INPUT_DIM = 1  # 3 (RGB) | 1 (Grayscale)
DIM = 64
VAE_DIM = 256
N_LAYERS = 15
K = 512
LR = 1e-3

DEVICE = torch.device('cuda') # torch.device('cpu')

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

autoencoder = VectorQuantizedVAE(INPUT_DIM, VAE_DIM, K).to(DEVICE)
autoencoder.load_state_dict(
    torch.load('models/{}_vqvae.pt'.format(DATASET))
)
autoencoder.eval()

model = GatedPixelCNN(K, DIM, N_LAYERS).to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)


def train():
    train_loss = []
    for batch_idx, (x, label) in enumerate(train_loader):
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
            x = x.to(DEVICE)
            label = label.to(DEVICE)

            latents, _ = autoencoder.encode(x)
            logits = model(latents.detach(), label)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(
                logits.view(-1, K),
                latents.view(-1)
            )
            val_loss.append(loss.item())

    print('Validation Completed!\tLoss: {} Time: {}'.format(
        np.asarray(val_loss).mean(0),
        time.time() - start_time
    ))
    return np.asarray(val_loss).mean(0)


def generate_samples():
    label = torch.arange(10).expand(10, 10).contiguous().view(-1)
    label = label.to(device=DEVICE, dtype=torch.int64)

    latents = model.generate(label, shape=LATENT_SHAPE, batch_size=100)
    x_tilde, _ = autoencoder.decode(latents)
    images = (x_tilde.cpu().data + 1) / 2

    save_image(
        images,
        'samples/samples_{}.png'.format(DATASET),
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
