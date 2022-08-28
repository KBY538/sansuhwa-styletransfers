import torch
from PIL import Image
import torchvision.transforms as transforms

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_cnn_normalization_mean(device, mean=[0.485, 0.456, 0.406]):
    return torch.tensor(mean).to(device)

def get_cnn_normalization_std(device, std=[0.229, 0.224, 0.225]):
    return torch.tensor(std).to(device)

def get_content_layers_default():
    return ['conv_4']

def get_style_layers_default():
    return ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def image_loader(image_name, device, imsize=256):
    
    loader = transforms.Compose([
                             transforms.Resize((imsize, imsize)),
                             transforms.ToTensor()])

    image = Image.open(image_name)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

  
def gram_matrix(input):
    a, b, c, d = input.size()

    features = input.view(a * b, c * d)

    G = torch.mm(features, features.t())

    return G.div(a * b * c * d)