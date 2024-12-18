import json
import os
from math import ceil

import clip
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

workdir = "./"
data_dir = os.path.join(workdir, "data")
results_dir = os.path.join(workdir, "results")

backbone = "ViT-L/14"
model, preprocess = clip.load(backbone, device=device)

dataset = CIFAR100(data_dir, train=True, transform=preprocess)
test_dataset = CIFAR100(data_dir, train=False, transform=preprocess)

classes = dataset.classes
prompts = [f"A photo of a {c.replace('_', ' ')}." for c in classes]
text = clip.tokenize(prompts).to(device)

n_cal = 512
idx = np.random.choice(len(dataset), n_cal, replace=False)
cal_dataset = Subset(dataset, idx)
cal_dataloader = DataLoader(cal_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

alpha = 0.20
scores = []
for data in tqdm(cal_dataloader):
    image, target = data

    image = image.to(device)
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
    probs = logits_per_image.softmax(dim=-1)

    scores.extend(1 - torch.gather(probs.cpu(), 1, target.view(-1, 1)).numpy())

q_level = ceil((n_cal + 1) * (1 - alpha)) / n_cal
s_cal = np.quantile(scores, q_level, method="higher")

test_prediction, test_score, test_label = [], [], []
for data in tqdm(test_dataloader, desc="Testing"):
    image, target = data

    image = image.to(device)
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
    probs = torch.softmax(logits_per_image, dim=-1)
    prediction_set = probs >= 1 - s_cal
    score = torch.sum(prediction_set, dim=-1)

    prediction = torch.multinomial(prediction_set.float() + 1e-08, 1)
    test_prediction.extend(prediction.squeeze().cpu().numpy().tolist())
    test_score.extend(score.cpu().numpy().tolist())
    test_label.extend(target.cpu().numpy().tolist())

test_prediction = np.array(test_prediction)
test_score = np.array(test_score)
test_label = np.array(test_label)

r = 0.20
ub = (1 - alpha) / (1 - r)

test_selected = (test_score > 0) * (test_score <= ub)
test_selective_risk = np.sum(
    test_prediction[test_selected] != test_label[test_selected]
) / np.sum(test_selected)


cp_results = {
    "alpha": alpha,
    "lambda": r,
    "n_cal": n_cal,
    "s_cal": s_cal.item(),
    "test_risk": test_selective_risk,
    "test_coverage": np.sum(test_selected) / len(test_selected),
}
json.dump(cp_results, open(os.path.join(results_dir, "cp_results.json"), "w"))
