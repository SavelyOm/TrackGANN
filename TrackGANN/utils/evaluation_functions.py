import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import pandas as pd

from TrackGANN.src.louvian import LouvainMethod, result2graph, compairsion


def GetTrueComm(name_of_directory):
    ''' A function to convert labeled data into a dictionary of the following format: dict(track_id: event_id) '''
    true_labels_list = []
    folder_path_toCSV = Path(name_of_directory) / Path(name_of_directory)/ "Timeslices"

    csv_files = [f for f in os.listdir(folder_path_toCSV) if f.endswith('.csv')]
    csv_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    for filename in csv_files:
        file_path = os.path.join(folder_path_toCSV, filename)
        df = pd.read_csv(file_path, sep=' ', index_col=False)
        true_labels_dict = dict(zip(df['track_id'], df['event_id']))
        true_labels_list.append(true_labels_dict)
    return true_labels_list



def pairwise_precision_recall(true_labels, pred_labels):
    """
    Compute pairwise precision and recall for sets.
    
    Args:
        true_labels: dict {node_id: true_cluster_id}
        pred_labels: dict {node_id: pred_cluster_id}
    
    Returns:
        precision, recall, f1
    """

    common_nodes = set(true_labels.keys()) & set(pred_labels.keys())
    
    if len(common_nodes) <= 1:
        return 0.0, 0.0, 0.0
    

    nodes_list = list(common_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}
    

    true_vector = [true_labels[node] for node in nodes_list]
    pred_vector = [pred_labels[node] for node in nodes_list]
    
    n_nodes = len(nodes_list)
    TP, TN, FP, FN = 0, 0, 0, 0
    

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            true_same = (true_vector[i] == true_vector[j])
            pred_same = (pred_vector[i] == pred_vector[j])
            
            if true_same and pred_same:
                TP += 1
            elif not true_same and not pred_same:
                TN += 1
            elif not true_same and pred_same:
                FP += 1
            elif true_same and not pred_same:
                FN += 1
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return accuracy, precision, recall, f1


def full_evaluate(model, loader, device, name_of_directory, plot_label, threshold=0.5):
    ''' This function performs a complete sorting of tracks 
    by events from start to finish. The function takes graph 
    data as input, processes it through a neural network model, 
    and finally performs full clustering using the Louvain Method. 
    The output is a dictionary of the form dict(event_id: list[tracks]). '''
    
    log_filename = Path(name_of_directory) / f'{plot_label}.csv'
    log_header = "Result"

    accuracy_arr, precision_arr, recall_arr, f1_arr = [], [], [], []
    
    try:
        true_labels_list = GetTrueComm(name_of_directory)
        i=0 
    except:
        pass



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


            if true_labels_list is not None and i < len(true_labels_list):
                true_comm = true_labels_list[i]

                node_to_predicted = {}
                for comm_id, nodes in result.items():
                    for node in nodes:
                        node_to_predicted[node] = comm_id

                try:
                    accuracy, precision, recall, f1 = pairwise_precision_recall(
                        {node: true_comm[node] for node in node_to_predicted.keys()},
                        node_to_predicted
                    )
                    
                    precision_arr.append(precision)
                    recall_arr.append(recall)
                    f1_arr.append(f1)
                    accuracy_arr.append(accuracy)
                    
                    print(f"\nBatch {i}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                    
                except KeyError as e:
                    print(f"KeyError in batch {i}: {e}")
                
                i += 1

        try:
            if len(precision_arr) > 0:
                accuracy_center = np.mean(accuracy_arr)
                precision_center = np.mean(precision_arr)
                recall_center = np.mean(recall_arr)
                f1_center = np.mean(f1_arr)

                accuracy_std = np.std(accuracy_arr)
                precision_std = np.std(precision_arr)
                recall_std = np.std(recall_arr)
                f1_std = np.std(f1_arr)
                
                status = (
                    f'Accuracy: {accuracy_center:.4f} ± {accuracy_std:.4f}'
                    f'Precision: {precision_center:.4f} ± {precision_std:.4f}\n'
                    f'Recall: {recall_center:.4f} ± {recall_std:.4f}\n'
                    f'F1: {f1_center:.4f} ± {f1_std:.4f}'
                )
                print("\n" + "="*50)
                print("FINAL RESULTS (Precision, Recall, F1):")
                print(status)
                print("="*50)
                
                
                return result
            else:
                print("No metrics were computed (no true labels available).")
                return result
            
        except Exception as e:
            print(f"Error computing final metrics: {e}")
            return result

