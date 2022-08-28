"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from .utils import pytorch03_to_pytorch04
from torch.autograd import Variable
import torchvision.utils as vutils
import torch
import os
from torchvision import transforms
from PIL import Image
from .options import Options
from .tester import MUNIT_Tester


class MunitGenerator():
    def __init__(self):
        opts = Options()
        torch.manual_seed(opts.seed)
        if not os.path.exists(opts.output_path):
            os.makedirs(opts.output_path)

        # Load experiment setting
        opts.num_style = 1 if opts.style != '' else opts.num_style

        tester = MUNIT_Tester(opts)
        try:
            state_dict = torch.load(opts.checkpoint, map_location=torch.device('cpu'))
            tester.gen_a.load_state_dict(state_dict['a'])
            tester.gen_b.load_state_dict(state_dict['b'])
        except:
            state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.tester)
            tester.gen_a.load_state_dict(state_dict['a'])
            tester.gen_b.load_state_dict(state_dict['b'])

        self.opts =  opts
        self.tester = tester

    def munit_generate(self, input_img_path, style_img_path, n='', convert_to='B'):   
        tester = self.tester
        opts = self.opts
        if convert_to == 'A':
            opts.a2b = 0
        tester.eval()
        encode = tester.gen_a.encode if opts.a2b else tester.gen_b.encode # encode function
        style_encode = tester.gen_b.encode if opts.a2b else tester.gen_a.encode # encode function
        decode = tester.gen_b.decode if opts.a2b else tester.gen_a.decode # decode function

        new_size = opts.new_size

        with torch.no_grad():
            transform = transforms.Compose([transforms.Resize(new_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            image = Variable(transform(Image.open(input_img_path).convert('RGB')).unsqueeze(0))
            style_image = Variable(transform(Image.open(style_img_path).convert('RGB')).unsqueeze(0)) if opts.style != '' else None

            # Start testing
            content, _ = encode(image)

            style_rand = Variable(torch.randn(opts.num_style, opts.gen['style_dim'], 1, 1))
            
            if opts.style != '':
                _, style = style_encode(style_image)
            else:
                style = style_rand

            for j in range(opts.num_style):
                s = style[j].unsqueeze(0)
                outputs = decode(content, s)
                outputs = (outputs + 1) / 2.
                path = os.path.join(opts.output_path, 'output{:03d}_{}.jpg'.format(j, n))
                vutils.save_image(outputs.data, path, padding=0, normalize=True)
            if not opts.output_only:
                # also save input images
                vutils.save_image(image.data, os.path.join(opts.output_path, 'input.jpg'), padding=0, normalize=True)
            
            output_array = outputs.data[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            output_img = Image.fromarray(output_array)

        return output_img, path