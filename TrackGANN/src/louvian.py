import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from collections import defaultdict

from .loggers import log_to_file


class GraphMethod:
    """Базовый класс для математических методов обработки графов"""
    def __init__(self, graph):
        self.graph = graph 
        self.result = None    

    def compute(self):
        """Основной метод вычислений"""
        raise NotImplementedError("Метод compute должен быть реализован в подклассе")

    def visualize_result(self):
        """Визуализация результатов"""
        if self.result is None:
            raise ValueError("Сначала выполните compute()")
        print(f"Результат вычислений: {self.result}")



class LouvainMethod(GraphMethod):
    """Реализация алгоритма Лувена для кластеризации графа"""
    def __init__(self, graph):
        self.graph = graph

    def modularity(self, adj_matrix: torch.Tensor, clusters: torch.Tensor):

        N = adj_matrix.size(0)
        m = adj_matrix.sum().item() / 2
        degrees = adj_matrix.sum(dim=1)

        self.Q = 0
    
        for i in range(N):
            for j in range(N):
                if clusters[i] == clusters[j]:
                    A_ij = adj_matrix[i, j]
                    k_i_k_j = degrees[i] * degrees[j]
                    self.Q += A_ij - (k_i_k_j / (2 * m))
        self.Q /= (2 * m)  # Нормализаци

        return self.Q

    def louvain_optimize(self, adj_matrix: torch.Tensor, clusters: torch.Tensor, max_iter: int = 20):
        device = adj_matrix.device
        """Итеративная оптимизация Лувена"""
        self.communities = clusters.clone().to(device)
        self.N = adj_matrix.size(0)
        
        for _ in range(max_iter):
            improved = False
            node_sequence = torch.randperm(self.N)
            
            for node in node_sequence:

                current_comm = self.communities[node].to(device)
                neighbors = torch.where(adj_matrix[node] > 0)[0]
                unique_comms = torch.unique(self.communities[neighbors]).to(device)
                
                
                best_comm = current_comm
                max_delta = 0.0
                Q_init = self.modularity(adj_matrix, self.communities)
                
                # Проверяем все соседние сообщества
                for comm in unique_comms:
                    if comm == current_comm:
                        continue
                        
                    # Пробное перемещение
                    test_communities = self.communities.clone()
                    test_communities[node] = comm
                    Q_new = self.modularity(adj_matrix, test_communities)
                    
                    if (delta := Q_new - Q_init) > max_delta:
                        max_delta = delta
                        best_comm = comm
                
                # Применяем лучшее перемещение
                if max_delta > 0:
                    self.communities[node] = best_comm
                    improved = True
            
            if not improved:
                break
        
        return self.communities

    def graph_constructer(self, communities: torch.Tensor):
        self.G = nx.Graph()
        for k in range(len(communities)):
            self.G.add_node(k, node_label = communities[k])
        
        edges = []
        for i in range(len(communities)):
            for j in range(len(communities)):
                if (i != j):
                    if (communities[i] == communities[j]):
                        edges.append([i,j])
        self.G.add_edges_from(edges)

        return self.G

    def compute(self):

        adj_matrix = to_dense_adj(self.graph.edge_index)[0]
        clusters = self.graph.x

        self.communities = self.louvain_optimize(adj_matrix, clusters)
        self.result = self.graph_constructer(self.communities)

        return self.result
    
    def clusters(self):
        return self.communities


    def visualize_result(self):
        colors = [self.G.nodes[i]['node_label'] for i in self.G.nodes]
        pos = nx.spring_layout(self.G, seed=42)

        plt.figure(figsize=(12, 8))
        nx.draw(self.G, pos, node_color=colors, cmap='tab20', node_size=200, 
            edge_color='gray', alpha=0.7, with_labels=True)
        plt.title(f'Граф после отчистки')
        plt.show()


def result2graph(pred, edge_index, threshold):
    edge_index_t = edge_index.t() 
    true_edges=torch.where(pred > threshold)[0]
    mask = torch.zeros(edge_index_t.shape[0], dtype=torch.bool, device='cuda:0')
    mask[true_edges] = True

    filtered_edge_index_t = edge_index_t[mask]
    filtered_edge_index=filtered_edge_index_t.t()
    nodes = torch.arange(0,max(filtered_edge_index[0].unique())+1,1)
    
    return Data(x=nodes,edge_index=filtered_edge_index)

@log_to_file()
def compairsion(tensor, **kwargs):

    cluster_to_nodes1 = defaultdict(list)

    for node_idx, cluster in enumerate(tensor):
        cluster_to_nodes1[cluster.item()].append(node_idx)
    
    return cluster_to_nodes1


#graph=result2graph(pred=pred,edge_index=edge_index)
#louvian = LouvainMethod(graph=graph)
#louvian.compute()
#clusters = louvian.clusters()
#print(clusters)
#compairsion(clusters)