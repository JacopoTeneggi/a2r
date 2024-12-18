import json
import os
import pickle

import clip
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR100
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng()

workdir = "./"
data_dir = os.path.join(workdir, "data")
results_dir = os.path.join(workdir, "results")


class SelectiveRiskMartingale:
    def __init__(self, tolerance):
        self.tolerance = tolerance

        self.loss = np.array([])
        self.wealth = np.array([1])

        self.fraction = np.array([0])
        self.min_fraction = 0
        self.max_fraction = 1
        self.a = 1

    def update(self, loss):
        payoff = loss - self.tolerance

        fraction = self.fraction[-1]
        bet = 1 + fraction * payoff
        wealth = self.wealth[-1] * bet

        z = payoff / bet
        self.a += z**2
        fraction = max(
            self.min_fraction,
            min(self.max_fraction, fraction + 2 / (2 - np.log(3)) * z / self.a),
        )

        self.loss = np.append(self.loss, loss)
        self.wealth = np.append(self.wealth, wealth)
        self.fraction = np.append(self.fraction, fraction)


def main():
    model, preprocess = clip.load("ViT-L/14", device=device)
    sgr_results = json.load(open(os.path.join(results_dir, "sgr_results.json"), "r"))
    tolerance = sgr_results["nominal_risk"]
    threshold = sgr_results["threshold"]

    dataset = CIFAR100(data_dir, train=False, transform=preprocess)
    classes = dataset.classes
    class_prompts = [f"A photo of a {class_name}" for class_name in classes]
    class_texts = clip.tokenize(class_prompts).to(device)

    @torch.no_grad()
    def score_fn(image):
        logits_per_image, _ = model(image, class_texts)
        probs = torch.softmax(logits_per_image, dim=-1)
        prediction = torch.argmax(probs, dim=-1).cpu()
        max_probs = torch.amax(probs, dim=-1).cpu()
        return prediction, max_probs

    prediction, confidence, label = [], [], []
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    for data in tqdm(dataloader, desc="Scoring test dataset"):
        image, _label = data
        image = image.to(device)
        _prediction, _confidence = score_fn(image)
        prediction.extend(_prediction.cpu().numpy().tolist())
        confidence.extend(_confidence.cpu().numpy().tolist())
        label.extend(_label.cpu().numpy().tolist())

    prediction = np.array(prediction)
    confidence = np.array(confidence)
    label = np.array(label)

    selected = confidence >= threshold
    loss = prediction != label
    scores = selected * loss

    r = 100
    results = {}
    for shift in ["none", 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        results[shift] = {"test_risk": None, "test_coverage": None, "tests": []}
        if shift == "none":
            shifted_scores = scores
        else:
            shift_fn = lambda image: T.functional.solarize(image, shift)
            shifted_preprocess = T.Compose([shift_fn, preprocess])

            shifted_dataset = CIFAR100(
                data_dir, train=False, transform=shifted_preprocess
            )

            shifted_prediction, shifted_confidence = [], []
            shifted_dataloader = DataLoader(
                shifted_dataset, batch_size=32, shuffle=False
            )
            for data in tqdm(
                shifted_dataloader,
                desc=f"Scoring shifted test dataset (shift = {shift})",
            ):
                image, _ = data
                image = image.to(device)
                _prediction, _confidence = score_fn(image)
                shifted_prediction.extend(_prediction.cpu().numpy().tolist())
                shifted_confidence.extend(_confidence.cpu().numpy().tolist())

            shifted_prediction = np.array(shifted_prediction)
            shifted_confidence = np.array(shifted_confidence)

            shifted_selected = shifted_confidence >= threshold
            shifted_loss = shifted_prediction != label
            shifted_scores = shifted_selected * shifted_loss

            results[shift]["test_risk"] = np.sum(shifted_scores) / np.sum(
                shifted_selected
            )
            results[shift]["test_coverage"] = np.sum(shifted_selected) / len(dataset)

        for _ in tqdm(range(r)):
            pi = rng.permutation(len(dataset))

            k = SelectiveRiskMartingale(tolerance)

            t_max = 512
            t_change = 128
            for t, idx in enumerate(pi[:t_max]):
                if t < t_change:
                    score = scores[idx]
                else:
                    score = shifted_scores[idx]
                k.update(score)

            results[shift]["tests"].append(k)

    pickle.dump(results, open(os.path.join(results_dir, "sgr_results.pkl"), "wb"))


if __name__ == "__main__":
    main()
