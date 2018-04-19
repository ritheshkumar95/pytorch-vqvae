import torch
import torch.nn as nn


def to_scalar(arr):
    if type(arr) == list:
        return [x.cpu().data.tolist()[0] for x in arr]
    else:
        return arr.cpu().data.tolist()[0]


def euclidean_distance(z_e_x, emb):
    dists = torch.pow(
        z_e_x.unsqueeze(1) - emb[None, :, :, None, None],
        2
    ).sum(2)
    return dists


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
        )

        self.embedding = nn.Embedding(512, 64)

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 1, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z_e_x = self.encoder(x)
        B, C, H, W = z_e_x.size()

        dists = euclidean_distance(z_e_x, self.embedding.weight)
        latents = dists.min(1)[1]

        shp = latents.size() + (-1, )
        z_q_x = self.embedding(latents.view(-1)).view(*shp)
        z_q_x = z_q_x.permute(0, 3, 1, 2)

        x_tilde = self.decoder(z_q_x)
        return x_tilde, z_e_x, z_q_x
