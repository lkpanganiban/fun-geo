'''
# Style Transfer
Implementing Style Transfer of Satellite Images

Requirements
- torch
- torchvision
- tqdm
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import os
import glob
from tqdm import tqdm

# Optional: help debug autograd errors
torch.autograd.set_detect_anomaly(True)

# ----- CONFIG -----
content_dir = "data/content_dir"
style_dir = "data/style_dir"
output_dir = "output_dir"
image_size = 998
model_name = "vgg19"
num_steps = 500
style_weight = 1e6
content_weight = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Image Utils -----
def load_image(img_path, size=None):
    image = Image.open(img_path).convert('RGB')
    if size:
        image = image.resize((size, size))
    transform = transforms.ToTensor()
    image = transform(image).unsqueeze(0).to(device)
    return image

def save_image(tensor, path):
    image = tensor.cpu().clone().squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(path)

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(c, h * w)
    G = torch.mm(features, features.t())
    return G / (c * h * w)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# ----- Replace all in-place ReLU -----
def replace_relu_with_new(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.ReLU):
            setattr(model, name, nn.ReLU(inplace=False))
        elif isinstance(layer, nn.Sequential):
            replace_relu_with_new(layer)
        elif hasattr(layer, 'children'):
            replace_relu_with_new(layer)

# ----- Model Selection -----
def get_model(name="vgg19"):
    if name.startswith("vgg"):
        cnn = getattr(models, name)(pretrained=True).features.to(device).eval()
        normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        content_layers = ['conv_4']
        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    elif name.startswith("resnet"):
        resnet = getattr(models, name)(pretrained=True).to(device).eval()
        cnn = nn.Sequential(*list(resnet.children())[:6])
        normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        content_layers = ['0']
        style_layers = ['0', '1', '2']
    else:
        raise ValueError("Unsupported model")
    return cnn, normalization_mean, normalization_std, content_layers, style_layers

# ----- Build Transfer Model -----
def build_transfer_model(cnn, norm_mean, norm_std, style_img, content_img, content_layers, style_layers):
    normalization = transforms.Normalize(norm_mean, norm_std)
    model = nn.Sequential(normalization)
    style_losses, content_losses = [], []

    i = 0
    for layer in cnn.children():
        name = f"conv_{i}"
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

        i += 1

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (StyleLoss, ContentLoss)):
            break
    return model[:i+1], style_losses, content_losses

# ----- Transfer Execution -----
def stylize(content_img, style_img, cnn, norm_mean, norm_std, content_layers, style_layers):
    input_img = content_img.clone()
    model, style_losses, content_losses = build_transfer_model(
        cnn, norm_mean, norm_std, style_img, content_img, content_layers, style_layers
    )
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_weight * style_score + content_weight * content_score
            loss.backward(retain_graph=True)
            run[0] += 1
            return loss
        if run[0] % 100 == 0:
            print(f"Step {run[0]}/{num_steps}")
        optimizer.step(closure)

    return input_img.clamp(0, 1)

# ----- Main Loop -----
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

content_paths = glob.glob(os.path.join(content_dir, "*.jpg"))
style_paths = glob.glob(os.path.join(style_dir, "*.jpg"))
cnn, norm_mean, norm_std, content_layers, style_layers = get_model(model_name)
replace_relu_with_new(cnn)  # crucial fix!

print(f"Processing {len(content_paths)} content images with {len(style_paths)} style images...")

for content_path in tqdm(content_paths):
    content_img = load_image(content_path, size=image_size)
    for style_path in style_paths:
        style_img = load_image(style_path, size=image_size)
        output_img = stylize(content_img, style_img, cnn, norm_mean, norm_std, content_layers, style_layers)

        content_name = os.path.splitext(os.path.basename(content_path))[0]
        style_name = os.path.splitext(os.path.basename(style_path))[0]
        out_name = f"{content_name}_stylized_{style_name}.jpg"
        save_path = os.path.join(output_dir, out_name)
        save_image(output_img, save_path)
