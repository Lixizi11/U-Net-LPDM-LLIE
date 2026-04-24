# cnn/train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import LOLDataset
from .model import CNNEnhancer
from utils import laplacian_loss

device = "cuda"

dataset = LOLDataset(
    "lol_dataset/our485/low",
    "lol_dataset/our485/high"
)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = CNNEnhancer().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(50):
    for low, high in loader:
        low, high = low.to(device), high.to(device)

        pred = model(low)

        loss = (
            F.l1_loss(pred, high)
            + 0.1 * laplacian_loss(pred, high)   # ⭐结构先验
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

    print("Epoch:", epoch, "Loss:", loss.item())

torch.save(model.state_dict(), "weights/cnn.pth")