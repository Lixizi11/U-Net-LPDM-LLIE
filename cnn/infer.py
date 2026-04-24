# cnn/infer.py
import os
import torch
from PIL import Image
from torchvision import transforms
from model import CNNEnhancer

device = "cuda"

model = CNNEnhancer().to(device)
model.load_state_dict(torch.load("weights/cnn.pth", weights_only=True))
model.eval()

t = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

in_dir = r"D:\fyp_project\supporting_document\lol_dataset\eval15\low"
out_dir = r"D:\fyp_project\supporting_document\lol_dataset\eval15\cnn_enhanced"
os.makedirs(out_dir, exist_ok=True)

for name in os.listdir(in_dir):
    img = Image.open(os.path.join(in_dir, name)).convert("RGB")
    x = t(img).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x)

    out = transforms.ToPILImage()(y.squeeze().cpu())
    out.save(os.path.join(out_dir, name))