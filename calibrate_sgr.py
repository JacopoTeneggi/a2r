import json
import os
from math import ceil

import clip
import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import binom
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

workdir = "./"
data_dir = os.path.join(workdir, "data")
results_dir = os.path.join(workdir, "results")


def _hoeffding_plus(r, loss, n):
    h1 = lambda u: u * np.log(u / r) + (1 - u) * np.log((1 - u) / (1 - r))
    return -n * h1(np.maximum(r, loss))


def _bentkus_plus(r, loss, n):
    return np.log(np.maximum(binom.cdf(np.floor(n * loss), n, r), 1e-10)) + 1


def hoeffding_bentkus_bound(n, delta, loss, maxiter=1000):
    def _tailprob(r):
        hoeffding_mu = _hoeffding_plus(r, loss, n)
        bentkus_mu = _bentkus_plus(r, loss, n)
        return np.minimum(hoeffding_mu, bentkus_mu) - np.log(delta)

    if _tailprob(1 - 1e-10) > 0:
        return 1
    else:
        try:
            return brentq(_tailprob, loss, 1 - 1e-10, maxiter=maxiter)
        except:
            print(f"BRENTQ RUNTIME ERROR at muhat={loss}")
            return 1.0


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

prediction, confidence, label = [], [], []
for data in tqdm(cal_dataloader, desc="Predicting on calibration set"):
    image, target = data

    image = image.to(device)
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
    probs = torch.softmax(logits_per_image, dim=-1)
    max_probs = torch.amax(probs, dim=-1)

    prediction.extend(logits_per_image.argmax(dim=-1).cpu().numpy().tolist())
    confidence.extend(max_probs.cpu().numpy().tolist())
    label.extend(target.numpy().tolist())

prediction = np.array(prediction)
confidence = np.array(confidence)
label = np.array(label)

sorted_confidence = np.sort(confidence)
risk, delta = 0.20, 0.01
z_min, z_max = 0, n_cal - 1
turns = ceil(np.log2(n_cal))
for _ in range(turns):
    z = np.ceil((z_min + z_max) / 2).astype(int)
    threshold = sorted_confidence[z]

    selected = confidence >= threshold
    selective_risk = np.sum(prediction[selected] != label[selected]) / np.sum(selected)
    bound = hoeffding_bentkus_bound(np.sum(selected), delta / turns, selective_risk)

    if bound < risk:
        z_max = z
    else:
        z_min = z

test_prediction, test_confidence, test_label = [], [], []
for data in tqdm(test_dataloader, desc="Testing"):
    image, target = data

    image = image.to(device)
    with torch.no_grad():
        logits_per_image, _ = model(image, text)
    probs = torch.softmax(logits_per_image, dim=-1)
    max_probs = torch.amax(probs, dim=-1)

    test_prediction.extend(logits_per_image.argmax(dim=-1).cpu().numpy().tolist())
    test_confidence.extend(max_probs.cpu().numpy().tolist())
    test_label.extend(target.numpy().tolist())

test_prediction = np.array(test_prediction)
test_confidence = np.array(test_confidence)
test_label = np.array(test_label)

test_selected = test_confidence >= threshold
test_selective_risk = np.sum(
    test_prediction[test_selected] != test_label[test_selected]
) / np.sum(test_selected)


sgr_results = {
    "nominal_risk": risk,
    "delta": delta,
    "n_cal": n_cal,
    "threshold": threshold,
    "risk": selective_risk,
    "bound": bound,
    "test_risk": test_selective_risk,
    "test_coverage": np.sum(test_selected) / len(test_selected),
}
json.dump(sgr_results, open(os.path.join(results_dir, "sgr_results.json"), "w"))
