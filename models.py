"""Model zoo for the JM-LLM reproduction (pure PyTorch).

Graph convolutions accept either a dense or a torch sparse adjacency, so the
same code paths work for the 2k sample and the 258k-node full dataset.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def spmm(adj, x):
    """adj @ x supporting both dense and sparse torch tensors."""
    if adj.is_sparse:
        return torch.sparse.mm(adj, x)
    return adj @ x


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj_norm):
        return spmm(adj_norm, x @ self.weight) + self.bias


class GNNModel(nn.Module):
    """GCN backbone (reference: 4 layers, hidden 128, output 64)."""

    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_layers=4,
                 dropout=0.5):
        super().__init__()
        self.dropout = dropout
        if num_layers > 2:
            dims = [input_dim] + [hidden_dim] * (num_layers - 2) + [output_dim]
        else:
            dims = [input_dim, output_dim]
        self.layers = nn.ModuleList(
            GCNLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

    def forward(self, x, adj_norm):
        h = x
        for layer in self.layers[:-1]:
            h = F.dropout(F.relu(layer(h, adj_norm)), p=self.dropout,
                          training=self.training)
        return self.layers[-1](h, adj_norm)


class LSTMModel(nn.Module):
    """LSTM backbone (reference: 4 layers, hidden 128, dropout 0.5)."""

    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_layers=4,
                 dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h, _ = self.lstm(x.unsqueeze(1))
        return self.fc(h[:, -1])


class GNNClassifier(nn.Module):
    def __init__(self, gnn, emb_dim=64, num_classes=2):
        super().__init__()
        self.gnn = gnn
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x, adj_norm=None):
        return {"class_pred": self.fc(self.gnn(x, adj_norm))}


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_layers=4,
                 dropout=0.5, num_classes=2):
        super().__init__()
        self.backbone = LSTMModel(input_dim, hidden_dim, output_dim,
                                  num_layers, dropout)
        self.fc = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        return {"class_pred": self.fc(self.backbone(x))}


class CollaborativeModel(nn.Module):
    """Jordan-fused classifier shared by JM-GNN and JM-LSTM."""

    def __init__(self, backbone, backbone_out_dim, jordan, hidden_dim=128,
                 num_classes=2):
        super().__init__()
        self.backbone = backbone
        n_eig = len(jordan.eigenvalues)
        self.fusion = nn.Sequential(
            nn.Linear(backbone_out_dim + n_eig, hidden_dim), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.key_node_predictor = nn.Linear(hidden_dim, 1)
        # normalised modal weights |lambda_i| / max|lambda|
        eig = torch.tensor(jordan.eigenvalues, dtype=torch.float32)
        eig = eig / eig.abs().max().clamp_min(1e-12)
        self.register_buffer("jordan_feat", eig)

    def forward(self, x, adj_norm=None):
        h = self.backbone(x, adj_norm) if adj_norm is not None \
            else self.backbone(x)
        jfeat = self.jordan_feat.expand(h.size(0), -1)
        fused = self.fusion(torch.cat([h, jfeat], dim=1))
        return {"class_pred": self.classifier(fused),
                "key_node_scores": self.key_node_predictor(fused).squeeze(-1)}


# ---------------------------------------------------------------------------
# JM-LLM collaborators
# ---------------------------------------------------------------------------
class JordanModel(nn.Module):
    """Jordan recurrent network for threat-propagation prediction (Sec. 3.1)."""

    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.hidden = nn.Linear(input_dim + output_dim, 64)
        self.output = nn.Linear(64, output_dim)

    def forward(self, x):
        prev = torch.zeros(x.size(0), self.output_dim, device=x.device)
        return torch.sigmoid(self.output(torch.tanh(
            self.hidden(torch.cat([x, prev], dim=1)))))


class LMModel(nn.Module):
    """LLM-style encoder for semantic threat analysis (Sec. 3.2)."""

    def __init__(self, input_dim, output_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.encoder(x)


def batched_predict(model, X, batch=4096, softmax=False):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out = model(X[i:i + batch].to(DEVICE))
            if softmax:
                out = torch.softmax(out, dim=1)
            outs.append(out.cpu().numpy())
    return np.vstack(outs)
