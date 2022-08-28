from .networks import AdaINGen
from torch.autograd import Variable
import torch.nn as nn

class MUNIT_Tester(nn.Module):
    def __init__(self, hyperparameters):
        super(MUNIT_Tester, self).__init__()
        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters.input_dim_a, hyperparameters.gen)  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters.input_dim_b, hyperparameters.gen)  # auto-encoder for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters.gen['style_dim']

        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())


    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake = self.gen_a.encode(x_a)
        c_b, s_b_fake = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a)
        x_ab = self.gen_b.decode(c_a, s_b)
        return x_ab, x_ba
        