import torch
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from pathlib import Path



def save_graph_to_npz(G, filename):
    ''' Saves a NetworkX graph in npz format with the filename specified as a function argument. '''

    mapping = {old_label: new_label for new_label, old_label in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)

    nodes = list(G.nodes())
    
    nodes_features = np.array([G.nodes[n]['node_feature'] for n in nodes])
    nodes_labels = np.array([G.nodes[n]['node_label'] for n in nodes])
    nodes_time_interval = np.array([G.nodes[n]['node_time_interval'] for n in nodes])

    edges = []
    for e in G.edges():
        edges.append(e)

    
    edges = np.array(edges)
    
    np.savez(filename,
             nodes=nodes,
             nodes_features=nodes_features,
             nodes_labels=nodes_labels,
             nodes_time_interval=nodes_time_interval,
             edges=edges)
    

def save_model(model, optimizer, epoch, dir_name, model_name):
    ''' Saves the trained model along with its optimizer and the number of the last training epoch. '''

    model_path = Path(dir_name) / "models" / f"{model_name}_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)




