from pathlib import Path

import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .net import decoder as decoder_net
from .net import vgg as vgg_net
from .function import adaptive_instance_normalization, coral

class Options():
    def __init__(self, decoder='models/decoder.pth', vgg='models/vgg_normalised.pth', content_size=512, style_size=512, crop=256, save_ext='.png', output='result'):
        self.decoder = os.path.join(os.path.dirname(os.path.abspath(__file__)), decoder)
        self.vgg = os.path.join(os.path.dirname(os.path.abspath(__file__)), vgg)
        self.content_size=content_size
        self.style_size=style_size
        self.crop=crop
        self.save_ext=save_ext
        self.output=output
        self.preserve_color=True
        self.do_interpolation=False
        self.alpha=1.0


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, device, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

class AdaINGenerator():
    def __init__(self):

        opts = Options()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        output_dir = Path(opts.output)
        output_dir.mkdir(exist_ok=True, parents=True)


        decoder = decoder_net
        vgg = vgg_net

        decoder.eval()
        vgg.eval()

        decoder.load_state_dict(torch.load(opts.decoder))
        vgg.load_state_dict(torch.load(opts.vgg))
        vgg = nn.Sequential(*list(vgg.children())[:31])

        vgg.to(device)
        decoder.to(device)

        self.opts = opts
        self.vgg = vgg
        self.decoder = decoder
        self.device = device

        self.content_tf = test_transform(opts.content_size, opts.crop)
        self.style_tf = test_transform(opts.style_size, opts.crop)

    def adain_generate(self, content, style):

        outputs=[]
        output_names=[]
        
        if os.path.isdir(content):
            content_dir = Path(content)
            content_paths = [f for f in content_dir.glob('*')]
        else:
            content_paths = [Path(content)]

        if os.path.isdir(style):
            style_dir = Path(style)
            style_paths = [f for f in style_dir.glob('*')]
        else:
            style_paths = style.split(',')
            if len(style_paths) == 1:
                style_paths = [Path(style)]
            else:
                interpolation_weights = [1 / len(style) for _ in style]
                
        for content_path in content_paths:
            if self.opts.do_interpolation:  # one content image, N style image
                style = torch.stack([self.style_tf(Image.open(str(p))) for p in style_paths])
                content = self.content_tf(Image.open(str(content_path))) \
                    .unsqueeze(0).expand_as(style)
                style = style.to(self.device)
                content = content.to(self.device)
                with torch.no_grad():
                        output = style_transfer(self.vgg, self.decoder, content, style, self.device, self.opts.alpha, interpolation_weights)
                output = output.cpu()
                output_name = self.opts.output + '/{:s}_interpolation{:s}'.format(
                    content_path.stem, self.opts.save_ext)
                
                ndarr = output[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                output = Image.fromarray(ndarr)
                output.save(output_name)

                outputs.append(output)
                output_names.append(output_name)

            else:  # process one content and one style
                for style_path in style_paths:
                    content = self.content_tf(Image.open(str(content_path)))
                    style = self.style_tf(Image.open(str(style_path)))
                    if self.opts.preserve_color:
                        style = coral(style, content)
                    style = style.to(self.device).unsqueeze(0)
                    content = content.to(self.device).unsqueeze(0)
                    with torch.no_grad():
                        output = style_transfer(self.vgg, self.decoder, content, style, self.device, self.opts.alpha)
                    output = output.cpu()

                    output_name = self.opts.output + '/{:s}_stylized_{:s}{:s}'.format(
                        content_path.stem, style_path.stem, self.opts.save_ext)

                    ndarr = output[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                    output = Image.fromarray(ndarr)
                    output.save(output_name)

                    outputs.append(output)
                    output_names.append(output_name)

        return outputs, output_names