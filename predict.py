import os

import clip
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

workdir = "./"
data_dir = os.path.join(workdir, "data")
results_dir = os.path.join(workdir, "results")

backbone = "ViT-L/14"
model, preprocess = clip.load(backbone, device=device)

dataset = CIFAR100(data_dir, train=False, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

classes = dataset.classes
prompts = [f"A photo of a {c.replace('_', ' ')}." for c in classes]
text = clip.tokenize(prompts).to(device)

results = {"prediction": [], "label": []}
for data in tqdm(dataloader):
    image, label = data

    image = image.to(device)
    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)

    prediction = logits_per_image.argmax(dim=-1)

    results["prediction"].extend(prediction.cpu().numpy().tolist())
    results["label"].extend(label.numpy().tolist())

df = pd.DataFrame(results)
df.to_csv(os.path.join(results_dir, "cifar100_predictions.csv"), index=False)
