import os

defaults = {
            'checkpoint':  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints/photo2sansu.pt'),
            'style': '',
            'a2b': 1,
            'seed': 10,
            'num_style': 1,
            'synchronized': True,
            'output_only': True,
            'output_path': './result',
            'model': 'MUNIT',
            'batch_size': 1,

            'gen': {
                'dim': 64,                     # number of filters in the bottommost layer
                'mlp_dim': 256,                # number of filters in MLP
                'style_dim': 8,                # length of style code
                'activ': 'relu',                 # activation function [relu/lrelu/prelu/selu/tanh]
                'n_downsample': 2,             # number of downsampling layers in content encoder
                'n_res': 4,                    # number of residual blocks in content encoder/decoder
                'pad_type': 'reflect'          # padding type [zero/reflect]
            },

            
            # data options
            'input_dim_a': 3,                              # number of image channels [1/3]
            'input_dim_b': 3,                              # number of image channels [1/3]
            'new_size': 256,                               # first resize the shortest image side to this size
            'crop_image_height': 256,                      # random crop image of this height
            'crop_image_width': 256,                       # random crop image of this width
        }

class Options():

    def __init__(self):
        self.checkpoint = defaults['checkpoint']
        self.style = defaults['style']
        self.a2b = defaults['a2b']
        self.seed = defaults['seed']
        self.num_style = defaults['num_style']
        self.synchronized = defaults['synchronized']
        self.output_only = defaults['output_only']
        self.output_path = defaults['output_path']
        self.tester = defaults['model']

        self.batch_size = 1
        self.gen = defaults['gen']
        self.input_dim_a = defaults['input_dim_a']
        self.input_dim_b = defaults['input_dim_b']
        self.new_size = defaults['new_size']
        self.crop_image_height = defaults['crop_image_height']
        self.crop_image_width = defaults['crop_image_width']