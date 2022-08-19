import torch
import matplotlib.pyplot as plt
import numpy as np

def check_accuracy(loader, model, device="cpu"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            preds = torch.sigmoid(model(data))
            preds = (preds > 0.5).float()
            num_correct += torch.sum(preds == target).item()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds * target).sum()) / (preds.sum() + target.sum()) + 1e-8
    
    model.train()
    
    return num_correct / num_pixels, dice_score / len(loader)