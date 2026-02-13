import torch.nn as nn
import torch.nn.functional as F
import torch


def snconvTrans2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.ConvTranspose2d(**kwargs), eps=eps)

def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)


class Decoder(nn.Module):
    def __init__(self, img_size=32, latent_dim=10, noise_dim=100):
        super(Decoder, self).__init__()
        in_channels = latent_dim + noise_dim
        self.linear = snlinear(in_features=in_channels, out_features=512)  # [bs, 512]
        self.deconv1 = snconvTrans2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1) # 512, 2, 2
        self.bn1 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(True))
        self.deconv2 = snconvTrans2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1) # 512, 4, 4
        self.bn2 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(True))
        self.deconv3 = snconvTrans2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) # 256, 8, 8
        self.bn3 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(True))
        self.deconv4 = snconvTrans2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)  # 128, 16, 16
        self.bn4 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(True))
        self.deconv5 = snconvTrans2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1) # 3, 32, 32
        self.tanh = nn.Tanh()

    def forward(self, z):
        # z = torch.cat([z0, z1], dim=1)
        code = self.linear(z)  # [bs, 512]
        code = F.relu(code.unsqueeze(-1).unsqueeze(-1), True)  # [bs, 512, 1, 1]
        out = self.bn1(self.deconv1(code))
        out = self.bn2(self.deconv2(out))
        out = self.bn3(self.deconv3(out))
        out = self.bn4(self.deconv4(out))
        out = self.tanh(self.deconv5(out))
        return out


class Decoder_64(nn.Module):
    def __init__(self, img_size=64, latent_dim=10, noise_dim=100):
        super(Decoder_64, self).__init__()
        in_channels = latent_dim + noise_dim
        self.linear = snlinear(in_features=in_channels, out_features=512)  # [bs, 512]
        self.deconv1 = snconvTrans2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1) # 512, 2, 2
        self.bn1 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(True))  # todo: leaky relu?
        self.deconv2 = snconvTrans2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1) # 512, 4, 4
        self.bn2 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(True))
        self.deconv3 = snconvTrans2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) # 256, 8, 8
        self.bn3 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(True))
        self.deconv4 = snconvTrans2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)  # 128, 16, 16
        self.bn4 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(True))
        self.deconv5 = snconvTrans2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) # 3, 32, 32
        self.bn5 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(True))
        self.deconv6 = snconvTrans2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1)  # 3, 64, 64
        self.tanh = nn.Tanh()

    def forward(self, z):
        # z = torch.cat([z0, z1], dim=1)
        code = self.linear(z)  # [bs, 512]
        code = F.relu(code.unsqueeze(-1).unsqueeze(-1), True)  # [bs, 512, 1, 1]
        out = self.bn1(self.deconv1(code))
        out = self.bn2(self.deconv2(out))
        out = self.bn3(self.deconv3(out))
        out = self.bn4(self.deconv4(out))
        out = self.bn5(self.deconv5(out))
        out = self.tanh(self.deconv6(out))
        return out


class Decoder_128(nn.Module):
    def __init__(self, img_size=128, latent_dim=10, noise_dim=100):
        super(Decoder_128, self).__init__()
        in_channels = latent_dim + noise_dim
        self.linear = snlinear(in_features=in_channels, out_features=512)  # [bs, 512]
        self.deconv1 = snconvTrans2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1) # 512, 2, 2
        self.bn1 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(True))  # todo: leaky relu?
        self.deconv2 = snconvTrans2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1) # 512, 4, 4
        self.bn2 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(True))
        self.deconv3 = snconvTrans2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, padding=1) # 256, 8, 8
        self.bn3 = nn.Sequential(nn.BatchNorm2d(512), nn.ReLU(True))
        self.deconv4 = snconvTrans2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)  # 128, 16, 16
        self.bn4 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(True))
        self.deconv5 = snconvTrans2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) # 128, 32, 32
        self.bn5 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(True))
        self.deconv6 = snconvTrans2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1) # 64, 64, 64
        self.bn6 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(True))
        self.deconv7 = snconvTrans2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1)  # 3, 128, 128
        self.tanh = nn.Tanh()

    def forward(self, z):
        # z = torch.cat([z0, z1], dim=1)
        code = self.linear(z)  # [bs, 512]
        code = F.relu(code.unsqueeze(-1).unsqueeze(-1), True)  # [bs, 512, 1, 1]
        out = self.bn1(self.deconv1(code))
        out = self.bn2(self.deconv2(out))
        out = self.bn3(self.deconv3(out))
        out = self.bn4(self.deconv4(out))
        out = self.bn5(self.deconv5(out))
        out = self.bn6(self.deconv6(out))
        out = self.tanh(self.deconv7(out))
        return out


class Decoder_112(nn.Module):
    def __init__(self, img_size=112, latent_dim=10, noise_dim=100):
        super(Decoder_112, self).__init__()
        in_channels = latent_dim + noise_dim
        self.linear = snlinear(in_features=in_channels, out_features=512 * 7 * 7)
        self.deconv1 = snconvTrans2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)  # 256, 14, 14
        self.bn1 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(True))
        self.deconv2 = snconvTrans2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)  # 128, 28, 28
        self.bn2 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(True))
        self.deconv3 = snconvTrans2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)   # 64, 56, 56
        self.bn3 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(True))
        self.deconv4 = snconvTrans2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1)     # 3, 112, 112
        self.tanh = nn.Tanh()

    def forward(self, z):
        code = self.linear(z)  # [bs, 512*7*7]
        code = code.view(code.size(0), 512, 7, 7)
        out = self.bn1(self.deconv1(code))
        out = self.bn2(self.deconv2(out))
        out = self.bn3(self.deconv3(out))
        out = self.tanh(self.deconv4(out))
        return out


class Decoder_224(nn.Module):
    def __init__(self, img_size=224, latent_dim=10, noise_dim=100):
        super(Decoder_224, self).__init__()
        in_channels = latent_dim + noise_dim
        self.linear = snlinear(in_features=in_channels, out_features=512 * 7 * 7)
        self.deconv1 = snconvTrans2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)  # 256, 14, 14
        self.bn1 = nn.Sequential(nn.BatchNorm2d(256), nn.ReLU(True))
        self.deconv2 = snconvTrans2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)  # 128, 28, 28
        self.bn2 = nn.Sequential(nn.BatchNorm2d(128), nn.ReLU(True))
        self.deconv3 = snconvTrans2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)   # 64, 56, 56
        self.bn3 = nn.Sequential(nn.BatchNorm2d(64), nn.ReLU(True))
        self.deconv4 = snconvTrans2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)    # 32, 112, 112
        self.bn4 = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(True))
        self.deconv5 = snconvTrans2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1)     # 3, 224, 224
        self.tanh = nn.Tanh()

    def forward(self, z):
        code = self.linear(z)  # [bs, 512*7*7]
        code = code.view(code.size(0), 512, 7, 7)
        out = self.bn1(self.deconv1(code))
        out = self.bn2(self.deconv2(out))
        out = self.bn3(self.deconv3(out))
        out = self.bn4(self.deconv4(out))
        out = self.tanh(self.deconv5(out))
        return out


