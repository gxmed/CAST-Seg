import torch
import torch.nn as nn
from einops import repeat
from transformers import AutoModel
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))
    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, (2, 2), 2)
        self.nConvs = _make_nConv(in_channels, out_channels, 2, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)
        return self.nConvs(x)


# Vision Encoder
class VisionModel(nn.Module):
    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()
        self.model = AutoModel.from_pretrained(vision_type, output_hidden_states=True)
        self.spatial_dim = project_dim

    def forward(self, x):
        output = self.model(x, output_hidden_states=True)
        return output['hidden_states']


class BackgroundDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.up4 = UpBlock(768, 384, nb_Conv=2)
        self.up3 = UpBlock(384, 192, nb_Conv=2)
        self.up2 = UpBlock(192, 96, nb_Conv=2)

    def forward(self, x):
        x3 = self.up4(x[-1], x[-2])
        x2 = self.up3(x3, x[-3])
        x1 = self.up2(x2, x[-4])
        return [x3, x2, x1]


class ForegroundDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.up4 = UpBlock(768, 384, nb_Conv=2)
        self.up3 = UpBlock(384, 192, nb_Conv=2)
        self.up2 = UpBlock(192, 96, nb_Conv=2)

    def forward(self, x):
        x3 = self.up4(x[-1], x[-2])
        x2 = self.up3(x3, x[-3])
        x1 = self.up2(x2, x[-4])
        return [x3, x2, x1]


class CASTSegNetwork(nn.Module):
    """ConvNeXt encoder with complementary foreground/background decoders.

    TSL, MDAA, ACC, ASR and WEMA are training-stage components and are kept
    outside this inference network, so deployment only requires this module.
    """
    def __init__(self, vision_type, project_dim=768):
        super(CASTSegNetwork, self).__init__()
        self.encoder = VisionModel(vision_type, project_dim)
        self.foreground_decoder = ForegroundDecoder(input_dim=768, hidden_dim=128, output_dim=1)
        self.background_decoder = BackgroundDecoder(input_dim=768, hidden_dim=128, output_dim=1)
        self.decoder1 = SubpixelUpsample(2, 96, 24, 4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

    def forward(self, data):
        # Text is spatialized by TSL into CAM priors before this network.
        image, gt = data

        if image.shape[1] == 1:
            image = repeat(image, 'b 1 h w -> b c h w', c=3)

        image_features = self.encoder(image)
        image_features = [feat for feat in image_features]

        # Inference is image-only; the training-time expert and text encoder are decoupled.
        fg_output_img = self.foreground_decoder(image_features)
        fg_output_preds = self.decoder1(fg_output_img[-1])
        fg_output = self.out(fg_output_preds).sigmoid()

        bg_output_neg = self.background_decoder(image_features)
        bg_output = self.decoder1(bg_output_neg[-1])
        bg_output = self.out(bg_output).sigmoid()

        img = [torch.mean(tensor, dim=(2, 3), keepdim=False) for tensor in fg_output_img]
        neg = [torch.mean(tensor, dim=(2, 3), keepdim=False) for tensor in bg_output_neg]

        img2 = [tensor / tensor.norm(p=2) for tensor in img]
        neg2 = [tensor / tensor.norm(p=2) for tensor in neg]

        return fg_output, bg_output, img2, neg2
