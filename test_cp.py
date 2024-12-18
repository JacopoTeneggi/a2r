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


class ConformalMartingale:
    def __init__(self, ref_dist):
        self.ref_dist = ref_dist
        self.score = np.array([])
        self.gamma = np.array([])
        self.wealth = np.array([1])

        self.fraction = np.array([0])
        self.min_fraction = -1
        self.max_fraction = 1
        self.a = 1

    def conformal_p_value(self, score, tau=None):
        tau = tau or rng.random()
        return (
            (self.ref_dist < score).sum() + tau * (1 + (self.ref_dist == score).sum())
        ) / (len(self.ref_dist) + 1)

    def update(self, score, tau=None):
        gamma = self.conformal_p_value(score, tau=tau)
        payoff = 2 * (gamma - 0.5)

        fraction = self.fraction[-1]
        bet = 1 + fraction * payoff
        wealth = self.wealth[-1] * bet

        z = payoff / bet
        self.a += z**2
        fraction = max(
            self.min_fraction,
            min(self.max_fraction, fraction + 2 / (2 - np.log(3)) * z / self.a),
        )

        self.score = np.append(self.score, score)
        self.gamma = np.append(self.gamma, gamma)
        self.wealth = np.append(self.wealth, wealth)
        self.fraction = np.append(self.fraction, fraction)


def main():
    model, preprocess = clip.load("ViT-L/14", device=device)
    cp_results = json.load(open(os.path.join(results_dir, "cp_results.json"), "r"))
    s_cal = cp_results["s_cal"]

    dataset = CIFAR100(data_dir, train=False, transform=preprocess)
    classes = dataset.classes
    class_prompts = [f"A photo of a {class_name}" for class_name in classes]
    class_texts = clip.tokenize(class_prompts).to(device)

    @torch.no_grad()
    def score_fn(image):
        logits_per_image, _ = model(image, class_texts)
        probs = torch.softmax(logits_per_image, dim=-1)
        prediction_set = probs > 1 - s_cal
        return torch.sum(prediction_set, dim=-1)

    scores = []
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    for data in tqdm(dataloader, desc="Scoring test dataset"):
        image, _ = data
        image = image.to(device)
        score = score_fn(image)
        scores.extend(score.cpu().numpy().tolist())
    scores = np.array(scores)

    r = 100
    results = {}
    for shift in ["none", 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        results[shift] = []
        if shift == "none":
            shifted_scores = scores
        else:
            shift_fn = lambda image: T.functional.solarize(image, shift)
            shifted_preprocess = T.Compose([shift_fn, preprocess])

            shifted_dataset = CIFAR100(
                data_dir, train=False, transform=shifted_preprocess
            )
            shifted_scores = []
            shifted_dataloader = DataLoader(
                shifted_dataset, batch_size=32, shuffle=False
            )
            for data in tqdm(
                shifted_dataloader,
                desc=f"Scoring shifted test dataset (shift = {shift})",
            ):
                image, _ = data
                image = image.to(device)
                score = score_fn(image)
                shifted_scores.extend(score.cpu().numpy().tolist())
            shifted_scores = np.array(shifted_scores)

        for _ in tqdm(range(r)):
            pi = rng.permutation(len(dataset))

            t_init = 256
            k = ConformalMartingale(scores[pi[:t_init]])

            t_max = 512
            t_change = 128
            for t, idx in enumerate(pi[t_init : t_init + t_max]):
                tau = rng.random()
                if t < t_change:
                    score = scores[idx]
                else:
                    score = shifted_scores[idx]
                k.update(score, tau=tau)

            results[shift].append(k)

    pickle.dump(results, open(os.path.join(results_dir, "cp_results.pkl"), "wb"))


if __name__ == "__main__":
    main()
