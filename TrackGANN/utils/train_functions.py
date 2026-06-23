import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from TrackGANN.src.loggers import log_to_file



def balanced_focal_loss(edge_attr, edge_label, pos_weight, neg_weight, gamma):
    ''' A function that calculates the average error across all instances for the BFL. '''

    p_t = torch.where(edge_label == 1, edge_attr, 1 - edge_attr)
    log_p_t = torch.log(torch.clamp(p_t, min=1e-7))
    weights = edge_label * pos_weight + (1 - edge_label) * neg_weight
    loss = (-weights * (1 - p_t)**gamma * log_p_t).mean()

    return loss


#########################################################################

def embeddings_distance_loss(node_attr, edge_attr, edge_label, edge_index, pos_weight, neg_weight, emb_margin, emb_alpha):

    x_i = node_attr[edge_index[0]]
    x_j = node_attr[edge_index[1]]

    dxij = torch.linalg.vector_norm(x_i - x_j, ord=2, dim=1)
    
    pos_mask = edge_label.eq(1).squeeze()
    neg_mask = edge_label.eq(0).squeeze()

    emb_pos = torch.tensor(0.0, dtype=edge_attr.dtype, device=edge_attr.device)
    emb_neg = torch.tensor(0.0, dtype=edge_attr.dtype, device=edge_attr.device)

    emb_pos = dxij[pos_mask].pow(2).mean()
    emb_neg = torch.clamp(emb_margin - dxij[neg_mask], min=0.0).pow(2).mean()

    loss = emb_alpha * (emb_pos + (neg_weight / pos_weight) * emb_neg)

    return loss


def node_degree_loss(node_attr, edge_attr, edge_label, edge_index, degree_alpha):

    indices = edge_index[1]
    num_nodes = node_attr.size(0)

    soft_degrees = torch.zeros(num_nodes, dtype=edge_attr.dtype, device=edge_attr.device)
    true_degrees = torch.zeros(num_nodes, dtype=edge_label.dtype, device=edge_label.device)

    soft_degrees.scatter_add_(0, indices, edge_attr.squeeze())
    true_degrees.scatter_add_(0, indices, edge_label.squeeze().to(edge_label.dtype))

    loss = degree_alpha * F.mse_loss(soft_degrees, true_degrees.to(soft_degrees.dtype))

    return loss


def full_loss(node_attr, edge_attr, edge_label, edge_index,
              pos_weight, neg_weight, gamma,
              emb_margin, emb_alpha, degree_alpha):
    
    bf_loss = balanced_focal_loss(edge_attr, edge_label, pos_weight, neg_weight, gamma)
    emb_loss = embeddings_distance_loss(node_attr, edge_attr, edge_label, edge_index, pos_weight, neg_weight, emb_margin, emb_alpha)
    nd_loss = node_degree_loss(node_attr, edge_attr, edge_label, edge_index, degree_alpha)

    return bf_loss, emb_loss, nd_loss, bf_loss + emb_loss + nd_loss

#########################################################################



def balanced_cross_entropy(pred, label, pos_weight=1, neg_weight=0.4):
    ''' A function that calculates the average error across all instances for the BCE '''

    weights = label * pos_weight + (1 - label) * neg_weight
    loss = F.binary_cross_entropy(pred, label, weight=weights, reduction='mean') 
    return loss




@log_to_file()
def train(model, loader, optimizer, criterion, device, **kwargs):
    ''' Trains the model. Requires specifying the model, optimizer, 
    loss function, dataset, and the device for training. The entire 
    training process is logged using a decorator. '''

    model.train()
    total_loss = 0

    #######################################
    total_bf_loss = 0
    total_emb_loss = 0
    total_nd_loss = 0
    #######################################

    for data in tqdm(loader, desc="Training", unit="timeslice"):

        data = data.to(device)
        optimizer.zero_grad()
        node_attr, edge_attr, edge_label, edge_index = model(data)

        #######################################
        bf_loss, emb_loss, nd_loss, loss = criterion(node_attr, edge_attr, edge_label, edge_index)
        #######################################

        loss.backward()         
        optimizer.step()
        total_loss += loss.item()

        #######################################
        total_bf_loss += bf_loss.item()
        total_emb_loss += emb_loss.item()
        total_nd_loss += nd_loss.item()
        #######################################

    print(f'bf_loss: {total_bf_loss/total_loss:.4f}; emb_loss: {total_emb_loss/total_loss:.4f}; nd_loss: {total_nd_loss/total_loss:.4f}')
    return total_loss / len(loader)


@log_to_file()
def evaluate(model, loader, criterion, device, threshold=0.5,  **kwargs):
    ''' Calculates quality metrics with a given threshold and 
    loss function on a specified test dataset, which is also 
    provided as a function argument. '''


    model.eval()

    total_loss = 0

    #######################################
    total_bf_loss = 0
    total_emb_loss = 0
    total_nd_loss = 0
    #######################################

    all_true_labels = []
    all_pred_labels = []


    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluation", unit="timeslice"):
            data = data.to(device)
            node_attr, edge_attr, edge_label, edge_index = model(data)
            bf_loss, emb_loss, nd_loss, loss = criterion(node_attr, edge_attr, edge_label, edge_index)
            total_loss += loss.item()

            
            #######################################
            total_bf_loss += bf_loss.item()
            total_emb_loss += emb_loss.item()
            total_nd_loss += nd_loss.item()
            #######################################
            
            all_true_labels.append(edge_label.cpu().numpy())
            all_pred_labels.append((edge_attr >= threshold).cpu().numpy())

    
    all_true_labels = np.concatenate(all_true_labels)
    all_pred_labels = np.concatenate(all_pred_labels)
    
    true_positive = np.sum((all_pred_labels == 1) & (all_true_labels == 1))
    true_negative = np.sum((all_pred_labels == 0) & (all_true_labels == 0))
    false_positive = np.sum((all_pred_labels == 1) & (all_true_labels == 0))
    false_negative = np.sum((all_pred_labels == 0) & (all_true_labels == 1))
    
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
    purity = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    efficiency = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    
    print(f'bf_loss: {total_bf_loss/total_loss:.4f}; emb_loss: {total_emb_loss/total_loss:.4f}; nd_loss: {total_nd_loss/total_loss:.4f}')
    return total_loss/len(loader), accuracy, purity, efficiency