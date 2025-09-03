import torch
from torch import nn 
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.nn import  GATConv

#from src.config import TRAINCONFIG
from TrackGANN.utils.model_functions import represent_to_graph_with_times, timeslice_split, tracks_pool, norm_layer



class CustomMLP(nn.Module):
   def __init__(self, inpsize, hiddsize, outpsize, act, norm, dropout, sigma):
      super(CustomMLP, self).__init__()

      firstlay = []
      hiddlay = []
      outlay = []

      firstlay.append(nn.Linear(inpsize, hiddsize[0]))
      firstlay.append(norm_layer(norm, hiddsize[0]))
      firstlay.append(act)
      firstlay.append(nn.Dropout(dropout))
      
      for n in range(1, len(hiddsize)):
        hiddlay.append(nn.Linear(hiddsize[n-1], hiddsize[n]))
        hiddlay.append(norm_layer(norm, hiddsize[n]))
        hiddlay.append(act)
        hiddlay.append(nn.Dropout(dropout))

      outlay.append(nn.Linear(hiddsize[-1], outpsize))

      if sigma == True:
        outlay.append(nn.Sigmoid())


      self.fn1 = nn.Sequential(*firstlay)
      self.fn2 = nn.Sequential(*hiddlay)
      self.fn3 = nn.Sequential(*outlay)
    
   def forward(self, x):
      x = self.fn1(x)
      x = self.fn2(x)
      x = self.fn3(x)
      return x
   
class GAT(nn.Module):
  def __init__(self, num_features, predconvdim, convdim, encoder_convolutions):
    super(GAT, self).__init__()

    firstlay = []
    convlay = []

    firstlay.append(GATConv(num_features, predconvdim))
    firstlay.append(nn.Tanh())
    firstlay.append(GATConv(predconvdim, convdim))

    convlay.append(GATConv(convdim, convdim))
    convlay.append(nn.Tanh())
    convlay.append(GATConv(convdim, convdim))

    self.conv1 = nn.Sequential(*firstlay)
    self.conv2 = nn.Sequential(*convlay)
    self.enc_conf = encoder_convolutions


  def forward(self, data):
  
    x, edge_index, t, y = data.x, data.edge_index, data.t, data.y

    x = self.conv1[0](x, edge_index)  # GATConv
    x = self.conv1[1](x)              # Tanh
    x = self.conv1[2](x, edge_index)  # GATConv

    for _ in range(self.enc_conf):
        x = self.conv2[0](x, edge_index)  # GATConv
        x = self.conv2[1](x)              # Tanh
        x = self.conv2[2](x, edge_index)  # GATConv
     
    return Data(x=x, edge_index=edge_index, y=y, t=t)
  
class Encoder(nn.Module):
  def __init__(self, num_features, predconvdim, convdim, indim, hiddim, outdim, encoder_convolutions, act, norm, dropout, sigma):
    super(Encoder, self).__init__()
    self.gcn = GAT(num_features=num_features, predconvdim=predconvdim, convdim=convdim, encoder_convolutions=encoder_convolutions)
    self.mlp = CustomMLP(inpsize=indim, hiddsize=hiddim, outpsize=outdim, act=act, norm=norm, dropout=dropout, sigma=sigma)

  def forward(self, data):
    data = self.gcn(data)
    tracks = timeslice_split(data)
    x, labels, time_intervals = tracks_pool(tracks)
    x = self.mlp(x)
    
    return x, time_intervals, labels




class EdgeWeightedGraphConv(MessagePassing):
    def __init__(self, node_features, num_iterations, edge_weight_layers, edge_update_layers, node_update_layers,  edge_features=1):
        super(EdgeWeightedGraphConv, self).__init__(aggr='add')
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_iterations = num_iterations 

        self.edge_weight_mlp = CustomMLP(2*node_features, edge_weight_layers, edge_features, act=nn.Tanh(), norm="layer", dropout=0.1, sigma=True)
        self.edge_update_mlp = CustomMLP(2*node_features + edge_features, edge_update_layers, edge_features, act=nn.Tanh(), norm="layer", dropout=0.1, sigma=True)
        self.node_update = CustomMLP(2*node_features, node_update_layers, node_features, act=nn.Tanh(), norm="layer", dropout=0.1, sigma=False)

    
        
    def forward(self, data):
        x, edge_index, y = data.x, data.edge_index, data.y
        row, col = edge_index

        
        
        edge_attr = self.edge_weight_mlp(torch.cat([x[row], x[col]], dim=-1))  
        for _ in range(self.num_iterations):
            

            updated_x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            edge_attr = self.edge_update_mlp(torch.cat([updated_x[row], updated_x[col], edge_attr], dim=-1))          
            x = updated_x + x
        
        edge_attr = edge_attr.squeeze()
        return edge_attr, y, edge_index

    def message(self, x_j, edge_attr):
        return x_j * edge_attr
        
    def update(self, aggr_out, x):
        out = torch.cat([aggr_out, x], dim=-1)  
        return self.node_update(out)



class TracksNN(nn.Module):
    def __init__(self, hidden_linear_layers, out_linear_layer, encoder_convolutions,
                 edge_weight_layers, edge_update_layers, node_update_layers, classifier_convolutions):
      super(TracksNN, self).__init__()
      self.encoder = Encoder(num_features=3, predconvdim=32, convdim=64, indim=64, hiddim = hidden_linear_layers, outdim=out_linear_layer, encoder_convolutions=encoder_convolutions, act=nn.Tanh(), norm="layer", dropout=0.1, sigma=False).to('cuda')
      self.classifier = EdgeWeightedGraphConv(node_features=out_linear_layer, num_iterations=classifier_convolutions, 
                                              edge_weight_layers=edge_weight_layers, edge_update_layers=edge_update_layers, node_update_layers=node_update_layers).to('cuda')
       
    
    def forward(self, data):
        x, time_intervals, y = self.encoder(data)
        graph = represent_to_graph_with_times(x, time_intervals, y).to('cuda')
        x, y, edge_index = self.classifier(graph)
        return x, y, edge_index
    





