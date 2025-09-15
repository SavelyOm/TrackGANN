import torch
import random
import torch_geometric
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch import nn



def del_pairs(edge_index):

    unique_pairs = {}
    edge_index = edge_index.T
    
    for pair in edge_index:
        sorted_pair = tuple(sorted(pair.tolist()))
        
        
        if sorted_pair not in unique_pairs:
            unique_pairs[sorted_pair] = pair.tolist()
    
    return torch.tensor(list(unique_pairs.values())).T.long().contiguous().to('cuda')


def timeslice_split(dataset):

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


def norm_layer(norm_type, features):
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