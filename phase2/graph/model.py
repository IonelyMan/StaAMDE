def build_hgt_model(metadata, hidden_channels: int, num_layers: int, heads: int, dropout: float):
    import torch.nn.functional as F
    from torch import nn
    from torch_geometric.nn import HGTConv, Linear

    class HGTGraphClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            node_types, _ = metadata
            self.node_types = list(node_types)
            self.dropout = dropout
            self.node_linears = nn.ModuleDict(
                {node_type: Linear(-1, hidden_channels) for node_type in self.node_types}
            )
            self.convs = nn.ModuleList(
                [
                    HGTConv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        metadata=metadata,
                        heads=heads,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 2),
            )

        def forward(self, data):
            x_dict = {}
            for node_type, x in data.x_dict.items():
                if node_type in self.node_linears:
                    x_dict[node_type] = F.relu(self.node_linears[node_type](x))

            for conv in self.convs:
                conv_out = conv(x_dict, data.edge_index_dict)
                next_x = {}
                for node_type, previous in x_dict.items():
                    value = conv_out.get(node_type)
                    if value is None:
                        value = previous
                    next_x[node_type] = F.dropout(F.relu(value), p=self.dropout, training=self.training)
                x_dict = next_x

            if "app" not in x_dict:
                raise ValueError("图中缺少 app 节点，无法做 APK 级分类")
            return self.classifier(x_dict["app"])

    return HGTGraphClassifier()


def build_ta_sgatv2_model(
    metadata,
    hidden_channels: int,
    num_layers: int,
    heads: int,
    dropout: float,
    type_embedding_dim: int = 16,
):
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch_geometric.nn import GATv2Conv, Linear, global_mean_pool

    class XAIDroidGATv2Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            node_types, edge_types = metadata
            self.node_types = list(node_types)
            self.edge_types = list(edge_types)
            self.model_type = "ta-sgatv2"
            self.dropout = dropout
            self.type_embedding_dim = type_embedding_dim if self.node_types else 0
            self.node_type_embedding = (
                nn.Embedding(max(len(self.node_types), 1), self.type_embedding_dim)
                if self.type_embedding_dim > 0
                else None
            )
            self.input_proj = Linear(-1, hidden_channels)
            self.convs = nn.ModuleList(
                [
                    GATv2Conv(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        heads=heads,
                        concat=False,
                        dropout=dropout,
                        add_self_loops=True,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 2),
            )

        def prepare_x(self, data):
            x = data.x.float()
            node_type = getattr(data, "node_type", None)
            if self.node_type_embedding is not None and node_type is not None:
                node_type = node_type.clamp(min=0, max=len(self.node_types) - 1)
                type_x = self.node_type_embedding(node_type)
                x = torch.cat([x, type_x], dim=-1)
            return F.elu(self.input_proj(x))

        def forward(self, data, return_attention_weights: bool = False):
            x = self.prepare_x(data)
            attention_weights = None
            for layer_index, conv in enumerate(self.convs):
                x = F.dropout(x, p=self.dropout, training=self.training)
                is_last = layer_index == len(self.convs) - 1
                if return_attention_weights and is_last:
                    x, attention_weights = conv(x, data.edge_index, return_attention_weights=True)
                else:
                    x = conv(x, data.edge_index)
                x = F.elu(x)

            batch = getattr(data, "batch", None)
            if batch is None:
                batch = x.new_zeros(x.size(0), dtype=torch.long)
            graph_x = global_mean_pool(x, batch)
            logits = self.classifier(graph_x)
            if return_attention_weights:
                return logits, attention_weights
            return logits

    return XAIDroidGATv2Classifier()


def build_model(
    metadata,
    hidden_channels: int,
    num_layers: int,
    heads: int,
    dropout: float,
    model_type: str = "ta-sgatv2",
    type_embedding_dim: int = 16,
):
    if model_type == "hgt":
        return build_hgt_model(metadata, hidden_channels, num_layers, heads, dropout)
    if model_type == "ta-sgatv2":
        return build_ta_sgatv2_model(metadata, hidden_channels, num_layers, heads, dropout, type_embedding_dim)
    raise ValueError(f"未知图模型类型: {model_type}")
