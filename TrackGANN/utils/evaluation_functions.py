import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import defaultdict

from TrackGANN.src.louvian import LouvainMethod, result2graph, compairsion


#========================================================================================================#
def GetTrueComm(name_of_directory):
    true_labels_list = []
    folder_path_toCSV = Path(name_of_directory) / Path(name_of_directory)/ "Timeslices"

    csv_files = [f for f in os.listdir(folder_path_toCSV) if f.endswith('.csv')]
    csv_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    for filename in csv_files:
        print(filename)
        file_path = os.path.join(folder_path_toCSV, filename)
        df = pd.read_csv(file_path, sep=' ', index_col=False)
        true_labels_dict = dict(zip(df['track_id'], df['event_id']))
        true_labels_list.append(true_labels_dict)
    return true_labels_list
#========================================================================================================#



def full_evaluate(model, loader, device, name_of_directory, plot_label, threshold=0.5):
    log_filename = Path(name_of_directory) / f'{plot_label}.csv'
    log_header = "Result"


    #========================================================================================================#
    try:
        true_labels_list = GetTrueComm(name_of_directory)
        i=0 
        ari_arr, nmi_arr = [],[]
    except:
        pass
    #========================================================================================================#


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

            #========================================================================================================#
            try:
                true_comm = true_labels_list[i]

                node_to_predicted = {}
                for comm_id, nodes in result.items():
                    for node in nodes:
                        node_to_predicted[node] = comm_id

                nodes_list = list(node_to_predicted.keys())
                predicted_vector = [node_to_predicted[node] for node in nodes_list]
                try:
                    true_vector = [true_comm[node] for node in nodes_list]
                except:
                    print(true_comm)
                    print(true_comm)

                # Adjusted Rand Index (ARI)
                ari = adjusted_rand_score(true_vector, predicted_vector)
                print(f"Adjusted Rand Index: {ari:.4f}")

                # Normalized Mutual Information (NMI)
                nmi = normalized_mutual_info_score(true_vector, predicted_vector)
                print(f"Normalized Mutual Information: {nmi:.4f}")

                ari_arr.append(ari)
                nmi_arr.append(nmi)


                i+=1
            except:
                pass
            #========================================================================================================#
    try:
        ari_center = np.sum(ari_arr)/len(loader)
        nmi_center = np.sum(nmi_arr)/len(loader)

        ari_std = np.std(ari_arr)
        nmi_std = np.std(nmi_arr)

        status = f'ari: {ari_center} +- {ari_std}\n nmi: {nmi_center} +- {nmi_std}'
        print(status)
    except:
        pass

    return result   