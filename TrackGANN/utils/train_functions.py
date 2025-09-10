import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from TrackGANN.src.loggers import log_to_file




def balanced_focal_loss(pred, label, pos_weight, neg_weight, gamma, sample_prob):
    with torch.no_grad():
        is_class_0 = (label == 0)
        is_class_1 = (label == 1)
        rand_mask = (torch.rand_like(pred) < sample_prob)
        active_mask = (is_class_0 & rand_mask) | is_class_1

    pred_active = pred[active_mask]
    label_active = label[active_mask]

    p_t = torch.where(label_active == 1, pred_active, 1 - pred_active)
    log_p_t = torch.log(torch.clamp(p_t, min=1e-7))
    weights = label_active * pos_weight + (1 - label_active) * neg_weight
    loss = -weights * (1 - p_t)**gamma * log_p_t

    return loss.mean()


def balanced_cross_entropy(pred, label, pos_weight=1, neg_weight=0.4):
    weights = label * pos_weight + (1 - label) * neg_weight
    loss = F.binary_cross_entropy(pred, label, weight=weights, reduction='mean') 
    return loss


@log_to_file()
def train(model, loader, optimizer, criterion, device, **kwargs):
    model.train()
    total_loss = 0

    for data in tqdm(loader, desc="Training", unit="timeslice"):

        data = data.to(device)
        optimizer.zero_grad()
        pred, label, edge_index = model(data)
        loss = criterion(pred, label)
        loss.backward()         
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)


@log_to_file()
def evaluate(model, loader, criterion, device, threshold=0.5,  **kwargs):
    model.eval()

    total_loss = 0
    all_true_labels = []
    all_pred_labels = []


    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluation", unit="timeslice"):
            data = data.to(device)
            pred, label, edge_index = model(data)
            loss = criterion(pred,label)
            total_loss += loss.item()
            
            all_true_labels.append(label.cpu().numpy())
            all_pred_labels.append((pred >= threshold).cpu().numpy())

    
    all_true_labels = np.concatenate(all_true_labels)
    all_pred_labels = np.concatenate(all_pred_labels)
    
    true_positive = np.sum((all_pred_labels == 1) & (all_true_labels == 1))
    true_negative = np.sum((all_pred_labels == 0) & (all_true_labels == 0))
    false_positive = np.sum((all_pred_labels == 1) & (all_true_labels == 0))
    false_negative = np.sum((all_pred_labels == 0) & (all_true_labels == 1))
    
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    purity = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    efficiency = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    

    return total_loss/len(loader), accuracy, purity, efficiency