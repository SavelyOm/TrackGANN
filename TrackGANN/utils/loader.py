import torch
import networkx as nx
import numpy as np

from torch_geometric.data import Data 
from pathlib import Path



def model_loader(model, optimizer, model_dir_name, model_name):
    ''' Loads the model and optimizer from the specified directory. '''
    
    models_dir = Path(model_dir_name) / "models"

    checkpoint_path = models_dir / f"{model_name}.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=True)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    
    return model, optimizer, last_epoch

def load_npz_to_pyg(filename):
    ''' This function/module loads a graph saved in the npz 
    format and converts it into PyTorch Geometric data containers. '''

    with np.load(filename, allow_pickle=True) as data:
        x = torch.from_numpy(data['nodes_features']).float()
        edge_index = torch.from_numpy(data['edges']).long().t().contiguous()
        y = torch.from_numpy(data['nodes_labels']).float()
        t = torch.from_numpy(data['nodes_time_interval']).float()
    
    return Data(x=x, edge_index=edge_index, y=y, t=t)


def load_graph_from_npz(filename):
    ''' This function/module loads a graph saved in the npz 
    format and converts it into Graph from networkx. '''

    data = np.load(filename, allow_pickle=True)
    
    G = nx.DiGraph()
    
    for i, node in enumerate(data['nodes']):
        G.add_node(node, feature=data['nodes_features'][i], label=data['nodes_labels'][i])
        G.add_edges_from(data['edges'])
    
    return G
