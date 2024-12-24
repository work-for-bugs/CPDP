import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.nn import GatedGraphConv

from graph_transformer_layers import GraphTransformerLayer
from mlp_readout import MLPReadout


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_edge_types, num_steps=8):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_edge_types = max_edge_types

        # 门控图神经网络模块，用于从图中提取高阶特征
        self.ggnn = GatedGraphConv(in_features=input_dim, out_features=output_dim,
                                   n_steps=num_steps, n_etypes=max_edge_types)
        # 基础的图卷积模块，用于节点特征更新
        self.gcn = GCNConv(in_features=input_dim, out_features=output_dim)
        # 图Transformer层，堆叠了多层（n_layers=3）
        n_layers = 3
        num_heads = 10
        self.n_layers = n_layers
        self.gtn = nn.ModuleList([GraphTransformerLayer(input_dim, output_dim, 
                                                        num_heads=num_heads, dropout=0.2,
                                                        max_edge_types=max_edge_types,
                                                        layer_norm=False, batch_norm=True, residual=True)
                                    for _ in range(n_layers - 1)])
        self.MPL_layer = MLPReadout(output_dim, 2)
        self.sigmoid = nn.Sigmoid()

        ffn_ratio = 2
        self.concat_dim = output_dim
        self.RepLK = nn.Sequential(
            nn.BatchNorm1d(self.concat_dim),
            nn.Conv1d(self.concat_dim, self.concat_dim * ffn_ratio, kernel_size=1, stride=1, padding=0, groups=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(self.concat_dim * ffn_ratio, self.concat_dim, kernel_size=1, stride=1, padding=0, groups=1, dilation=1),
        )
        k = 3

        self.Avgpool1 =  nn.Sequential(
            nn.ReLU(),
            nn.AvgPool1d(k, stride= k),
            nn.Dropout(0.1)
        )
        self.ConvFFN = nn.Sequential(
            nn.BatchNorm1d(self.concat_dim),
            nn.Conv1d(self.concat_dim, self.concat_dim * ffn_ratio, kernel_size=1, stride=1, padding=0, groups=1),
            nn.GELU(),
            nn.Conv1d(self.concat_dim  * ffn_ratio, self.concat_dim, kernel_size=1, stride=1, padding=0, groups=1),
        )
        self.Avgpool2 =  nn.Sequential(
            nn.ReLU(),
            nn.AvgPool1d(k, stride= k),
            nn.Dropout(0.1)
        )

    def forward(self, batch, cuda=False):
        graph, features, edge_types = batch.get_network_inputs(cuda=cuda)
        graph = graph.to(torch.device('cuda:0'))
        for conv in self.gtn:
            features = conv(graph, features, edge_types)
        outputs = batch.de_batchify_graphs(features)
        
        outputs = outputs.transpose(1, 2)
        ''' 
              Layer1
        '''
        outputs += self.RepLK(outputs)
        outputs = self.Avgpool1(outputs)
        outputs += self.ConvFFN(outputs)
        outputs = self.Avgpool2(outputs)
        '''
              Layer2
        ''' 
        outputs = outputs.transpose(1, 2)
        outputs = self.MPL_layer(outputs.sum(dim=1))
        outputs = nn.Softmax(dim=1)(outputs)
        return outputs
    
    