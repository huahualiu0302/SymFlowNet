"""Network definitions of SymFlowNet and of the reference models.

Generator: encoder-decoder network of SymFlowNet (Fig. 3 of the paper).
    ConvBlock I (7 x 7) -> n_down ConvBlock I with stride 2 -> ResnetBlocks
    -> n_down DeconvBlocks with CBAM -> ConvBlock II (7 x 7, Tanh).
    With n_down = 3 and six residual blocks the receptive field is 275 x 275 pixels,
    which covers a whole 200 x 200 slice.
SNDiscriminator: PatchGAN discriminator with spectral normalization, used only by the
    CycleGAN reference models.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolution, optional instance normalization and activation."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=2, padding=1, activation='lrelu', norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_ch) if norm else None
        self.act = {'lrelu': nn.LeakyReLU(0.2, True), 'relu': nn.ReLU(True), 'tanh': nn.Tanh(),
                    'none': nn.Identity()}[activation]

    def forward(self, x):
        out = self.conv(x)
        if self.norm is not None:
            out = self.norm(out)
        return self.act(out)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction), nn.ReLU(),
                                nn.Linear(channels // reduction, channels), nn.Sigmoid())

    def forward(self, x):
        avg = self.fc(self.avg_pool(x).flatten(1))
        mx = self.fc(self.max_pool(x).flatten(1))
        return x * (avg + mx)[:, :, None, None]


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional block attention module: channel attention followed by spatial attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


class DeconvBlock(nn.Module):
    """Transposed convolution, instance normalization, CBAM and LeakyReLU."""

    def __init__(self, in_ch, out_ch, use_attention=True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 3, 2, 1, 1)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.cbam = CBAM(out_ch) if use_attention else None
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.norm(self.deconv(x))
        if self.cbam is not None:
            out = self.cbam(out)
        return self.act(out)


class ResnetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3, 1, 0), nn.InstanceNorm2d(channels), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3, 1, 0), nn.InstanceNorm2d(channels))

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, in_ch, ngf=32, out_ch=1, n_resnet=6, use_attention=True, n_down=3):
        super().__init__()
        self.pad = nn.ReflectionPad2d(3)
        self.conv_in = ConvBlock(in_ch, ngf, kernel_size=7, stride=1, padding=0)
        ch = [ngf * 2 ** k for k in range(n_down + 1)]
        self.downs = nn.Sequential(*[ConvBlock(ch[k], ch[k + 1]) for k in range(n_down)])
        self.resnet_blocks = nn.Sequential(*[ResnetBlock(ch[-1]) for _ in range(n_resnet)])
        self.ups = nn.Sequential(*[DeconvBlock(ch[k + 1], ch[k], use_attention) for k in reversed(range(n_down))])
        self.conv_out = ConvBlock(ngf, out_ch, kernel_size=7, stride=1, padding=0, activation='tanh', norm=False)

    def forward(self, x):
        x = self.downs(self.conv_in(self.pad(x)))
        x = self.ups(self.resnet_blocks(x))
        return self.conv_out(self.pad(x))


class FlowNet(nn.Module):
    """Generator whose output is multiplied by the pore image, so the velocity is exactly zero in the grains.

    in_ch = 1: the input is the pore image only (SymFlowNet, separate networks, ablations).
    in_ch = 2: the input is a velocity field and the pore image (translation models).
    """

    def __init__(self, in_ch=1, use_attention=True, ngf=32, n_resnet=6, n_down=3):
        super().__init__()
        self.in_ch = in_ch
        self.g = Generator(in_ch, ngf, 1, n_resnet, use_attention, n_down)

    def forward(self, field, pore):
        x = pore if self.in_ch == 1 else torch.cat([field, pore], 1)
        return self.g(x) * pore


class Transposed(nn.Module):
    """y-direction predictor obtained from an x-direction network through Eq. (1): u_y(m) = [u_x(m^T)]^T."""

    def __init__(self, net):
        super().__init__()
        self.net = [net]          # kept in a list so that the parameters are not registered twice

    def forward(self, field, pore):
        f = None if field is None else field.transpose(-1, -2)
        return self.net[0](f, pore.transpose(-1, -2)).transpose(-1, -2)


class SNDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral normalization (CycleGAN reference models)."""

    def __init__(self, in_ch=2, ndf=64):
        super().__init__()
        sn = nn.utils.spectral_norm
        ch = [in_ch, ndf, ndf * 2, ndf * 4, ndf * 8]
        layers = []
        for k in range(4):
            layers += [sn(nn.Conv2d(ch[k], ch[k + 1], 4, 2 if k < 3 else 1, 1)), nn.LeakyReLU(0.2, True)]
        layers.append(sn(nn.Conv2d(ch[4], 1, 4, 1, 1)))
        self.m = nn.Sequential(*layers)

    def forward(self, x):
        return self.m(x)


class ImagePool:
    """History of generated samples for the discriminator updates (Zhu et al., 2017)."""

    def __init__(self, size=50):
        self.size = size
        self.images = []

    def query(self, images):
        import random
        if self.size == 0:
            return images
        out = []
        for img in images.detach():
            img = img[None]
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif random.random() > 0.5:
                k = random.randint(0, self.size - 1)
                out.append(self.images[k].clone())
                self.images[k] = img
            else:
                out.append(img)
        return torch.cat(out, 0)
