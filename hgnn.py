from torch_geometric.nn import HeteroConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Linear, Module, ModuleList
from torch.nn.functional import relu, dropout
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.dense.linear import Linear


class EdgeAwareSAGEConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        aggr="mean",
        edge_dropout=0.2,
        edge_hidden_dim=16,
        init_edge_scale=0.5,
        learnable_edge_scale=True,
        use_root_weight=True,
        normalize= False,
        bias=True,
    ):
        super().__init__(aggr=aggr)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.edge_dropout = edge_dropout
        self.edge_hidden_dim = edge_hidden_dim
        self.use_root_weight = use_root_weight
        self.normalize = normalize
        self._init_edge_scale = float(init_edge_scale)

        self.lin_node = Linear(in_channels[0], out_channels, bias=False)

        if use_root_weight:
            self.lin_root = Linear(in_channels[1], out_channels, bias=bias)
        else:
            self.lin_root = None

        self.edge_norm = nn.LayerNorm(edge_dim)

        self.edge_mlp = nn.Sequential(
            Linear(edge_dim, edge_hidden_dim),
            nn.ReLU(),
            nn.Dropout(edge_dropout),
            Linear(edge_hidden_dim, out_channels),
        )

        if learnable_edge_scale:
            self.edge_scale = nn.Parameter(torch.tensor(self._init_edge_scale))
        else:
            self.register_buffer("edge_scale", torch.tensor(self._init_edge_scale))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_node.reset_parameters()

        if self.lin_root is not None:
            self.lin_root.reset_parameters()

        self.edge_norm.reset_parameters()

        for module in self.edge_mlp:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        with torch.no_grad():
            self.edge_scale.fill_(self._init_edge_scale)


    def forward(self, x, edge_index, edge_attr, size=None):
        if isinstance(x, tuple):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        edge_attr = self.edge_norm(edge_attr)
        edge_attr = F.dropout(edge_attr, p=self.edge_dropout, training=self.training)

        out = self.propagate(
            edge_index=edge_index,
            x=(x_src, x_dst),
            edge_attr=edge_attr,
            size=size,
        )

        if self.lin_root is not None and x_dst is not None:
            out = out + self.lin_root(x_dst)

        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)

        return out

    def message(self, x_j, edge_attr):
        node_msg = self.lin_node(x_j)
        edge_msg = self.edge_mlp(edge_attr)
        return node_msg + self.edge_scale * edge_msg



class SAGE_HGNN_edge_features(Module):
    def __init__(self, data, parameters):
        super().__init__()
        self.output_size = data.get('y').shape[-1]

        prefix = data.get('prefix')
        self.ata_edge_dim = prefix['activity', 'act_to_act', 'activity'].edge_attr.shape[-1]
        self.rta_edge_dim = prefix['resource', 'res_to_act', 'activity'].edge_attr.shape[-1]
        self.rtr_edge_dim = prefix['resource', 'res_to_res', 'resource'].edge_attr.shape[-1]

        self.dropout = parameters.get('dropout')
        self.layers_size = parameters.get('layers_size')
        self.num_hidden = parameters.get('hidden_layers')
        self.aggregator = parameters.get('aggregator')
        self.linear_size = self.layers_size*2

        if parameters.get('readout') == 'add':
            self.readout = global_add_pool
        elif parameters.get('readout') == 'mean':
            self.readout = global_mean_pool
        elif parameters.get('readout') == 'max':
            self.readout = global_max_pool

        self.convs = ModuleList()
        self.convs.append(HeteroConv({
            ('activity', 'act_to_act', 'activity'): EdgeAwareSAGEConv((-1, -1), self.layers_size, self.ata_edge_dim,
                                                                      edge_dropout=self.dropout),
            ('resource', 'res_to_act', 'activity'): EdgeAwareSAGEConv((-1, -1), self.layers_size, self.rta_edge_dim,
                                                                      edge_dropout=self.dropout),
            ('resource', 'res_to_res', 'resource'): EdgeAwareSAGEConv((-1, -1), self.layers_size, self.rtr_edge_dim,
                                                                      edge_dropout=self.dropout),
        }, aggr=self.aggregator))

        for i in range(self.num_hidden):
            self.convs.append(HeteroConv({
            ('activity', 'act_to_act', 'activity'): EdgeAwareSAGEConv((-1, -1), self.layers_size, self.ata_edge_dim, 
                                                                      edge_dropout=self.dropout),
            ('resource', 'res_to_act', 'activity'): EdgeAwareSAGEConv((-1, -1), self.layers_size, self.rta_edge_dim,
                                                                      edge_dropout=self.dropout),
            ('resource', 'res_to_res', 'resource'): EdgeAwareSAGEConv((-1, -1), self.layers_size, self.rtr_edge_dim, 
                                                                      edge_dropout=self.dropout),
            }, aggr=self.aggregator))

        self.convs.append(HeteroConv({
            ('activity', 'act_to_act', 'activity'): EdgeAwareSAGEConv((-1, -1), self.layers_size, self.ata_edge_dim,
                                                                      edge_dropout=self.dropout),
            ('resource', 'res_to_act', 'activity'): EdgeAwareSAGEConv((-1, -1), self.layers_size, self.rta_edge_dim,
                                                                      edge_dropout=self.dropout),
            ('resource', 'res_to_res', 'resource'): EdgeAwareSAGEConv((-1, -1), self.layers_size, self.rtr_edge_dim,
                                                                      edge_dropout=self.dropout),
        }, aggr=self.aggregator))

        self.context_conv = HeteroConv({
            ('activity', 'act_to_act', 'activity'): EdgeAwareSAGEConv((-1, -1), self.layers_size, self.ata_edge_dim,
                                                                      edge_dropout=self.dropout),
            ('resource', 'res_to_act', 'activity'): EdgeAwareSAGEConv((-1, -1), self.layers_size, self.rta_edge_dim,
                                                                      edge_dropout=self.dropout),
            ('resource', 'res_to_res', 'resource'): EdgeAwareSAGEConv((-1, -1), self.layers_size, self.rtr_edge_dim,
                                                                      edge_dropout=self.dropout),
        }, aggr=self.aggregator)

        self.lin1 =Linear(self.linear_size, self.linear_size)
        self.lin2 = Linear(self.linear_size, self.output_size)

    def forward(self, prefix, context):
        prefix_dict = prefix.x_dict
        edge_index_dict = prefix.edge_index_dict
        edge_attr_dict = prefix.edge_attr_dict

        context_dict = context.x_dict
        context_edge_index_dict = context.edge_index_dict
        context_edge_attr_dict = context.edge_attr_dict

        for conv in self.convs:
            prefix_dict = conv(prefix_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            prefix_dict = {k: relu(v) for k, v in prefix_dict.items()}
            prefix_dict = {k: dropout(v, p=self.dropout, training=self.training) for k, v in prefix_dict.items()}
        prefix_x_activity = self.readout(prefix_dict['activity'], prefix['activity'].batch)

        context_dict = self.context_conv(context_dict, context_edge_index_dict, edge_attr_dict=context_edge_attr_dict)
        context_dict = {k: relu(v) for k, v in context_dict.items()}
        context_dict = {k: dropout(v, p=self.dropout, training=self.training) for k, v in context_dict.items()}
        context_x_activity = self.readout(context_dict['activity'], context['activity'].batch)

        x_cat = torch.cat([prefix_x_activity, context_x_activity], dim=-1)
        x_cat = self.lin1(x_cat)
        x_cat = relu(x_cat)
        x_cat = dropout(x_cat, p=self.dropout, training=self.training)
        x_cat = self.lin2(x_cat)

        return x_cat
    