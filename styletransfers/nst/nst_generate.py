from .utils import get_device, image_loader, get_cnn_normalization_mean, get_cnn_normalization_std, get_content_layers_default
from .nets import get_style_model_and_losses
import torch
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms

class NST():
    def __init__(self, num_steps=300,
                style_weight = 1000000, content_weight=1):

        self.device = get_device()
        self.cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(self.device).eval()
        self.num_steps = num_steps
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.normalization_mean = get_cnn_normalization_mean(self.device)
        self.normalization_std = get_cnn_normalization_std(self.device)
    
    def get_input_optimizer(self, input_img):
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def run_style_transfer(self, content_img, style_img, input_img, n='', save_path='result/'):

        content_img = image_loader(content_img, self.device)
        style_img = image_loader(style_img, self.device)
        input_img = image_loader(input_img, self.device)
        
        model, style_losses, content_losses = get_style_model_and_losses(self.cnn,
                                            self.normalization_mean, self.normalization_std, style_img, content_img)

        input_img.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = self.get_input_optimizer(input_img)

        run = [0]
        while run[0] <= self.num_steps:

            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= self.style_weight
                content_score *= self.content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1

                return style_score + content_score

            optimizer.step(closure)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        unloader = transforms.ToPILImage()
        output = input_img.cpu().clone()
        output = output.squeeze(0)
        output = unloader(output)

        save_path =save_path+'nst_result{}.png'.format(n)
        output.save(save_path)
        
        return output, save_path
