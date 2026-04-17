import torch
import random
import torch_geometric
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch import nn



def del_pairs(edge_index):
    ''' Removes duplicate edge indices, converting an undirected graph into a directed one. '''

    unique_pairs = {}
    edge_index = edge_index.T
    
    for pair in edge_index:
        sorted_pair = tuple(sorted(pair.tolist()))
        
        
        if sorted_pair not in unique_pairs:
            unique_pairs[sorted_pair] = pair.tolist()
    
    return torch.tensor(list(unique_pairs.values())).T.long().contiguous().to('cuda')


def timeslice_split(dataset):
    ''' Splits the graph consisting of hits into separate tracks with their coordinates, labels, and drift times. '''

    tm = dataset
    tracks = []
    splt_num1 = []
    splt_num2 = []

    edge_index1 = del_pairs(tm.edge_index)
    for n in range(len(edge_index1[0])-1):
        if edge_index1[1][n]+1 != edge_index1[1][n+1]:
            splt_num1.append(int(edge_index1[1][n]+1))

    splt_num1.append(int(max(edge_index1[1]))+1)
    splt_num2.append(int(0))
    
    for n in range(len(splt_num1)-1):
        splt_num2.append(int(splt_num1[n]))

    splt_num1 = np.array(splt_num1)
    splt_num2 = np.array(splt_num2)
    num = splt_num1-splt_num2-1

    for i in range(len(num)):
        x = tm.x[splt_num2[i]:splt_num1[i]].to('cuda')
        y = tm.y[splt_num2[i]:splt_num1[i]].to('cuda')
        t = tm.t[splt_num2[i]:splt_num1[i]].to('cuda')


        data = Data(x=x, y=y, t=t)
        tracks.append(data)
    
    return tracks



def tracks_pool(tracks):
    ''' Averages the feature vectors of each hit within a track. 
    Takes as input a list of tensors containing feature vectors of hits. '''

    represent = []
    labels = []
    time_intervals = []

    for track in tracks:
            time_intervals.append(track.t[0])
            labels.append((track.y[0]))
            batch = torch.zeros(track.x.size(0), dtype=torch.long).to('cuda')
            pool = torch_geometric.nn.global_mean_pool(track.x,batch).to('cuda')
            represent.append(pool)
    
    pool_result = torch.stack(represent).float().squeeze().to('cuda')
    labels = torch.tensor(labels).float().to('cuda')
    time_intervals = torch.stack(time_intervals).float().squeeze().to('cuda')

    return pool_result, labels, time_intervals










def del_pairs_GPU(edge_index):

    edge_index = edge_index.to('cuda')
    
    sorted_edges, _ = torch.sort(edge_index, dim=0)
    unique_edges = torch.unique(sorted_edges, dim=1)

    return unique_edges.contiguous()




def tracks_pool_GPU(dataset):


    x = dataset.x.to('cuda')
    y = dataset.y.to('cuda')
    t = dataset.t.to('cuda')
    edge_index = dataset.edge_index.to('cuda')

    clear_edge_index = del_pairs_GPU(edge_index)

    dst_nodes = clear_edge_index[1]
    num_nodes = int(dst_nodes[-1] + 1) 

    diff = dst_nodes[1:] - dst_nodes[:-1]
    breaks = torch.nonzero(diff != 1).view(-1) +1

    
    node_boundaries = torch.cat([torch.tensor([0+1], device='cuda'), dst_nodes[breaks].long(), torch.tensor([num_nodes+1], device='cuda')])
    
    lengths = node_boundaries[1:] - node_boundaries[:-1]

    batch_tensor = torch.repeat_interleave(torch.arange(len(lengths), device='cuda'), lengths)

    
    track_starts = torch.cat([torch.tensor([0], device='cuda'), dst_nodes[breaks].long(), torch.tensor([num_nodes-1], device='cuda')])
    labels = y[track_starts[:-1]].float()
    time_intervals = t[track_starts[:-1]].float()
    pool_result = torch_geometric.nn.global_mean_pool(x, batch_tensor)

    return pool_result, labels, time_intervals












def norm_layer(norm_type, features):
  ''' A function that returns the normalization function based on a given key. '''

  if norm_type == "layer":
      return nn.LayerNorm(features)
  elif norm_type == "batch":
      return nn.BatchNorm1d(features)
  elif norm_type == "instance":
      return nn.InstanceNorm1d(features)
  elif norm_type is None:
      return nn.Identity()
  else:
      raise ValueError(f"Unsupported normalization type: {norm_type}")
  












def represent_to_graph_with_times(encoded_tracks, times, labels):
    ''' Builds a supergraph where nodes are encoded tracks, and 
    edges are formed based on the intersection of drift time intervals. '''

    hmatrix = encoded_tracks.float().to('cuda')
    edge_index = []
    edge_labels = []

    for i in range(len(encoded_tracks)):
        for j in range(len(encoded_tracks)):
            if i != j and [j, i] not in edge_index:
                if times[i][1] > times[j][0] and times[i][0] < times[j][1]:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_labels.append(1 if labels[i] == labels[j] else 0)
                    edge_labels.append(1 if labels[i] == labels[j] else 0)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to('cuda')
    edge_labels = torch.tensor(edge_labels if labels is not None else [0]*len(edge_index[0]), 
                             dtype=torch.float).to('cuda')

    return Data(x=hmatrix, edge_index=edge_index, y=edge_labels)














def represent_to_graph_with_times_GPU(encoded_tracks, times, labels):


    device = encoded_tracks.device
    N = encoded_tracks.size(0)

    time_left = times[:,0].view(-1, 1)
    time_right = times[:,1].view(-1, 1)

    cond1 = (time_left < time_right.T)
    cond2 = (time_right > time_left.T)

    mask = cond1 & cond2 & ~torch.eye(N,dtype=torch.bool, device=device)

    edge_index = torch.nonzero(mask).T.contiguous()


    if labels is not None:
        row, col = edge_index[0], edge_index[1]
        edge_labels = (labels[row] == labels[col]).float()
    else:
        edge_labels = torch.zeros(edge_index.size(1), device=device)


    return Data(x=encoded_tracks.float(), edge_index=edge_index, y=edge_labels)
