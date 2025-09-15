import torch
import math
import pandas as pd
import numpy as np
import networkx as nx
from torch_geometric.data import Data


def csv_to_graph(csv_file):
    ''' A converter for a CSV files in the format 
    [x, y, z, track_id, event_id, t_left, t_right] '''
    timeslice = pd.read_csv(csv_file, sep=' ')

    tm_nodes = []
    tm_edges = []
    tm_labels = []
    tm_pos = []
    time_interval = []

    for i in range(len(timeslice)):
        hit = [timeslice['x'][i], timeslice['y'][i], timeslice['z'][i]]
        if "event_id" in timeslice.columns:
            label = timeslice['event_id'][i]
        else:
            label = float('nan')

        time = [timeslice['time_left'][i], timeslice['time_right'][i]]
    
        tm_pos.append(i)
        tm_nodes.append(hit)
        tm_labels.append(label)
        time_interval.append(time)

    for i in range(len(timeslice)-1):
        track_id = timeslice['track_id']
        
        if track_id[i] == track_id[i+1]:
            tm_edges.append((i,i+1))
            tm_edges.append((i+1,i))

    
    G = nx.DiGraph()
    for k in range(len(tm_nodes)):
        G.add_node(k, node_feature=tm_nodes[k], node_label = tm_labels[k], node_time_interval = time_interval[k])
    G.add_edges_from(tm_edges)

    return G


def dict_to_graph(timeslice,
                  format):
    ''' A converter for a list of dictionaries in the format 
    list(dict(hit: [x,y,z], track_id: tr_id, drift_time: [t_left, t_right], event_id: ev_id)) '''

    sorted_timeslice = sorted(timeslice, key=lambda x: (x["track_id"], math.sqrt(x["x"]**2+x["y"]**2+x["z"]**2)))
    tm_nodes = []
    tm_edges = []
    tm_labels = []
    tm_pos = []
    time_interval = []
    for i in range(len(timeslice)):
        tm_dict = sorted_timeslice[i]
        if "event_id" in tm_dict:
            label = tm_dict['event_id']
        else:
            label = float('nan')

        time = tm_dict['drift_time']
        hit = tm_dict['hit']

        tm_pos.append(i)
        tm_nodes.append(hit)
        tm_labels.append(label)
        time_interval.append(time)
    
    for i in range(len(timeslice)-1):
        tm_dict = sorted_timeslice[i]
        track_id = tm_dict["track_id"]
        
        if track_id[i] == track_id[i+1]:
            tm_edges.append((i,i+1))
            tm_edges.append((i+1,i))
    
    if format == 'nx_Graph':
        G = nx.DiGraph()
        for k in range(len(tm_nodes)):
            G.add_node(k, node_feature=tm_nodes[k], node_label = tm_labels[k], node_time_interval = time_interval[k])
        G.add_edges_from(tm_edges)

        return G
    
    if format == 'pygeometric':
        pyg_graph = Data(x=tm_nodes, edge_index=tm_edges, t=time_interval, y=tm_labels)

        return pyg_graph