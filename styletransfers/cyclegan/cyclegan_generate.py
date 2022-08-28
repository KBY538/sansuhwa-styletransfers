import os
from .models import create_model
from .util import util
import cv2
import torchvision.transforms as transforms


defaults = {
                'name':'experiment_name',
                'gpu_ids': [],
                'checkpoints_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints'),
                'model': 'cycle_gan',
                'input_nc': 3,
                'output_nc': 3,
                'ngf': 64,
                'ndf': 64,
                'netD': 'basic',
                'netG': 'resnet_9blocks',
                'n_layers_D': 3,
                'norm': 'instance',
                'init_type': 'normal',
                'init_gain': 0.02,
                'no_dropout': True,
                'dataset_mode': 'unaligned',
                'direction': 'AtoB',
                'serial_batches': True,
                'num_threads': 4,
                'batch_size': 1,
                'load_size': 286,
                'crop_size': 256,
                'max_dataset_size': float("inf"),
                'preprocess': 'resize_and_crop',
                'no_flip': True,
                'display_winsize': 256,
                'epoch': 'latest',
                'load_iter': 0,
                'verbose': True,
                'suffix':'',
                'results_dir': './results/',
                'aspect_ratio': 1.0,
                'phase': 'test',
                'eval': True,
                'num_test': 500,
        }

class Options():

    def __init__(self, dataroot):
        self.dataroot = dataroot
        self.isTrain = False
        self.name = defaults['name']
        self.gpu_ids = defaults['gpu_ids']
        self.checkpoints_dir = defaults['checkpoints_dir']
        self.model = defaults['model']
        self.input_nc = defaults['input_nc']
        self.output_nc = defaults['output_nc']
        self.ngf = defaults['ngf']
        self.ndf = defaults['ndf']
        self.netD = defaults['netD']
        self.netG = defaults['netG']
        self.n_layers_D = defaults['n_layers_D']
        self.norm = defaults['norm']
        self.init_type = defaults['init_type']
        self.init_gain = defaults['init_gain']
        self.no_dropout = defaults['no_dropout']
        self.dataset_mode = defaults['dataset_mode']
        self.direction = defaults['direction']
        self.serial_batches = defaults['serial_batches']
        self.num_threads = defaults['num_threads']
        self.batch_size = defaults['batch_size']
        self.load_size = defaults['load_size']
        self.crop_size = defaults['crop_size']
        self.max_dataset_size = defaults['max_dataset_size']
        self.preprocess = defaults['preprocess']
        self.no_flip = defaults['no_flip']
        self.display_winsize = defaults['display_winsize']
        self.epoch = defaults['epoch']
        self.load_iter = defaults['load_iter']
        self.verbose = defaults['verbose']
        self.model_suffix = defaults['suffix']
        self.results_dir = defaults['results_dir']
        self.aspect_ratio = defaults['aspect_ratio']
        self.phase = defaults['phase']
        self.eval = defaults['eval']
        self.num_test = defaults['num_test']


class CycleGANGenerator():
    def __init__(self, data_root='./datasets'):
        opt = Options(data_root)
        opt.name = 'photo2sansu'
        
        self.opt = opt
        
        self.model = create_model(opt)    
        self.model.setup(opt)              
        self.model.eval()

    def cyclegan_generate(self, image_dir, save_dir, image_name, convert_to=['B']):
        util.mkdirs(save_dir)
        img = cv2.imread(image_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size))
        transform = transforms.ToTensor()
        transformed_img = transform(img)
        transformed_img = transformed_img.unsqueeze(0)

        im_data = self.model.generate_one(transformed_img, fake=convert_to[0])
        im = util.tensor2im(im_data)
        save_path = os.path.join(save_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=self.opt.aspect_ratio)

        return im, save_path