import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path

from TrackGANN.src.louvian import LouvainMethod, result2graph, compairsion


def full_evaluate(model, loader, device, name_of_directory, plot_label, threshold=0.5):
    log_filename = Path(name_of_directory) / f'{plot_label}.csv'
    log_header = "Result"

    model.eval()
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluation", unit="timeslice"):
            data = data.to(device)
            pred, label, edge_index = model(data)
            graph=result2graph(pred=pred,edge_index=edge_index, threshold=threshold)
            louvian = LouvainMethod(graph=graph)
            louvian.compute()
            clusters = louvian.clusters()
            result = compairsion(clusters, log_filename=log_filename, log_header=log_header)

            
    return result