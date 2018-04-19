import torch
import torch.nn as nn


def to_scalar(arr):
    if type(arr) == list:
        return [x.cpu().data.tolist()[0] for x in arr]
    else:
        return arr.cpu().data.tolist()[0]


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        return x + self.block(x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            ResBlock(256),
            nn.BatchNorm2d(256),
            ResBlock(256),
            nn.BatchNorm2d(256)
        )

        self.embedding = nn.Embedding(512, 256)
        self.embedding.weight.data.copy_(1./512 * torch.randn(512, 256))

        self.decoder = nn.Sequential(
            ResBlock(256),
            nn.BatchNorm2d(256),
            ResBlock(256),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        z_e_x = self.encoder(x)
        B, C, H, W = z_e_x.size()

        z_e_x_transp = z_e_x.permute(0, 2, 3, 1)  # (B, H, W, C)
        emb = self.embedding.weight.transpose(0, 1)  # (C, K)
        dists = torch.pow(
            z_e_x_transp.unsqueeze(4) - emb[None, None, None],
            2
        ).sum(-2)
        latents = dists.min(-1)[1]

        z_q_x = self.embedding(latents.view(latents.size(0), -1))
        z_q_x = z_q_x.view(B, H, W, C).permute(0, 3, 1, 2)
        x_tilde = self.decoder(z_q_x)
        return x_tilde, z_e_x, z_q_x
