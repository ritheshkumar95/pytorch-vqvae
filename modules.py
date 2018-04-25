import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def to_scalar(arr):
    if type(arr) == list:
        return [x.cpu().data.tolist()[0] for x in arr]
    else:
        return arr.cpu().data.tolist()[0]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        return x + self.block(x)


class AutoEncoder(nn.Module):
    def __init__(self, K=512):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.ReLU(True),
            ResBlock(256),
            ResBlock(256),
        )

        self.embedding = nn.Embedding(K, 256)
        self.embedding.weight.data.copy_(1./K * torch.randn(K, 256))

        self.decoder = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 3, 4, 2, 1),
            nn.Sigmoid()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)

        z_e_x_transp = z_e_x.permute(0, 2, 3, 1)  # (B, H, W, C)
        emb = self.embedding.weight.transpose(0, 1)  # (C, K)
        dists = torch.pow(
            z_e_x_transp.unsqueeze(4) - emb[None, None, None],
            2
        ).sum(-2)
        latents = dists.min(-1)[1]
        return latents, z_e_x

    def decode(self, latents):
        shp = latents.size() + (-1, )
        z_q_x = self.embedding(latents.view(latents.size(0), -1))  # (B * H * W, C)
        z_q_x = z_q_x.view(*shp).permute(0, 3, 1, 2)  # (B, C, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde, z_q_x

    def forward(self, x):
        latents, z_e_x = self.encode(x)
        x_tilde, z_q_x = self.decode(latents)
        return x_tilde, z_e_x, z_q_x


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class PixelCNN(nn.Module):
    def __init__(self, dim=64, n_layers=4):
        super().__init__()
        self.dim = 64

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(256, dim)

        # Building the PixelCNN layer by layer
        net = []

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            net.extend([
                MaskedConv2d(mask_type, dim, dim, 7, 1, 3, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(True)
            ])

        # Add the output layer
        net.append(nn.Conv2d(dim, 256, 1))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, W)
        return self.net(x)

    def generate(self, batch_size=64):
        x = Variable(
            torch.zeros(64, 8, 8).long()
        ).cuda()

        for i in range(8):
            for j in range(8):
                logits = self.forward(x)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x
