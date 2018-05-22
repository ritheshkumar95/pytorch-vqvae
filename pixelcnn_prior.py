import numpy as np
import torch
import torch.nn.functional as F
import json
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from modules import VectorQuantizedVAE, GatedPixelCNN
from datasets import MiniImagenet

from tensorboardX import SummaryWriter

def train(data_loader, model, prior, optimizer, args, writer):
    for images, labels in data_loader:
        with torch.no_grad():
            images = images.to(args.device)
            latents = model.encode(images)
            latents = latents.detach()

        labels = labels.to(args.device)
        logits = prior(latents, labels)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        optimizer.zero_grad()
        loss = F.cross_entropy(logits.view(-1, args.k),
                               latents.view(-1))
        loss.backward()

        # Logs
        writer.add_scalar('loss/train', loss.item(), args.steps)

        optimizer.step()
        args.steps += 1

def test(data_loader, model, prior, args, writer):
    with torch.no_grad():
        loss = 0.
        for images, labels in data_loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            latents = model.encode(images)
            latents = latents.detach()
            logits = prior(latents, labels)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss += F.cross_entropy(logits.view(-1, args.k),
                                    latents.view(-1))

        loss /= len(data_loader)

    # Logs
    writer.add_scalar('loss/valid', loss.item(), args.steps)

    return loss.item()

def main(args):
    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_filename = './models/{0}/prior.pt'.format(args.output_folder)

    if args.dataset in ['mnist', 'fashion-mnist', 'cifar10']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if args.dataset == 'mnist':
            # Define the train & test datasets
            train_dataset = datasets.MNIST(args.data_folder, train=True,
                download=True, transform=transform)
            test_dataset = datasets.MNIST(args.data_folder, train=False,
                transform=transform)
            num_channels = 1
        elif args.dataset == 'fashion-mnist':
            # Define the train & test datasets
            train_dataset = datasets.FashionMNIST(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(args.data_folder,
                train=False, transform=transform)
            num_channels = 1
        elif args.dataset == 'cifar10':
            # Define the train & test datasets
            train_dataset = datasets.CIFAR10(args.data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(args.data_folder,
                train=False, transform=transform)
            num_channels = 3
        valid_dataset = test_dataset
    elif args.dataset == 'miniimagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Define the train, valid & test datasets
        train_dataset = MiniImagenet(args.data_folder, train=True,
            download=True, transform=transform)
        valid_dataset = MiniImagenet(args.data_folder, valid=True,
            download=True, transform=transform)
        test_dataset = MiniImagenet(args.data_folder, test=True,
            download=True, transform=transform)
        num_channels = 3

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=16, shuffle=True)

    # Save the label encoder
    with open('./models/{0}/labels.json'.format(args.output_folder), 'w') as f:
        json.dump(train_dataset._label_encoder, f)

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(test_loader))
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    writer.add_image('original', fixed_grid, 0)

    model = VectorQuantizedVAE(num_channels, args.hidden_size_vae, args.k).to(args.device)
    with open(args.model, 'rb') as f:
        state_dict = torch.load(f)
        model.load_state_dict(state_dict)
    model.eval()

    prior = GatedPixelCNN(args.k, args.hidden_size_prior,
        args.num_layers, n_classes=len(train_dataset._label_encoder)).to(args.device)
    optimizer = torch.optim.Adam(prior.parameters(), lr=args.lr)

    best_loss = -1.
    for epoch in range(args.num_epochs):
        train(train_loader, model, prior, optimizer, args, writer)
        # The validation loss is not properly computed since
        # the classes in the train and valid splits of Mini-Imagenet
        # do not overlap.
        loss = test(valid_loader, model, prior, args, writer)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open(save_filename, 'wb') as f:
                torch.save(prior.state_dict(), f)

if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='PixelCNN Prior for VQ-VAE')

    # General
    parser.add_argument('--data-folder', type=str,
        help='name of the data folder')
    parser.add_argument('--dataset', type=str,
        help='name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet)')
    parser.add_argument('--model', type=str,
        help='filename containing the model')

    # Latent space
    parser.add_argument('--hidden-size-vae', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--hidden-size-prior', type=int, default=64,
        help='hidden size for the PixelCNN prior (default: 64)')
    parser.add_argument('--k', type=int, default=512,
        help='number of latent vectors (default: 512)')
    parser.add_argument('--num-layers', type=int, default=15,
        help='number of layers for the PixelCNN prior (default: 15)')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=128,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=3e-4,
        help='learning rate for Adam optimizer (default: 3e-4)')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='prior',
        help='name of the output folder (default: prior)')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
        help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--device', type=str, default='cpu',
        help='set the device (cpu or cuda, default: cpu)')

    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    if not os.path.exists('./models/{0}'.format(args.output_folder)):
        os.makedirs('./models/{0}'.format(args.output_folder))
    args.steps = 0

    main(args)
