"""Shared data pipeline and Jordan matrix tools for the JM-LLM reproduction.

Supports two UNSW-NB15 schemas:
- official 45-column split files (UNSW_NB15_training-set.csv / testing-set.csv)
- the raw 49-column sample shipped with the reference repository

All graph/Jordan operations are sparse so that the full dataset (~258k nodes)
can be handled without materialising an N x N dense matrix.
"""
from pathlib import Path
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, lobpcg
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

FULL_DIR = Path(os.environ.get(
    "UNSW_NB15_DIR", Path(__file__).resolve().parent / "data" / "unsw_full"))
FULL_TRAIN = FULL_DIR / "UNSW_NB15_training-set.csv"
FULL_TEST = FULL_DIR / "UNSW_NB15_testing-set.csv"
SAMPLE_CSV = Path(__file__).resolve().parent / "unsw_nb15_sample.csv"

# --- official 45-column training/testing schema ----------------------------
FULL_NUMERIC = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports",
]
# --- raw 49-column schema shipped with the reference repository ------------
RAW_NUMERIC = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "sload",
    "dload", "sloss", "dloss", "sintpkt", "dintpkt", "sjit", "djit", "swin",
    "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smeansz",
    "dmeansz", "trans_depth", "res_bdy_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports",
]
CATEGORICAL_FEATURES = ["proto", "service", "state"]


def _numeric_columns(df):
    cols = [c for c in FULL_NUMERIC if c in df.columns]
    if len(cols) < 10:  # fall back to the raw schema names
        cols = [c for c in RAW_NUMERIC if c in df.columns]
    return cols


def _clean(df):
    df = df.dropna(subset=["label"]).drop_duplicates().reset_index(drop=True)
    numerics = _numeric_columns(df)
    for col in numerics:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numerics] = df[numerics].fillna(0)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("-").astype(str)
    return df, numerics


def _fit_transform(train_df, test_df):
    """StandardScaler on numeric features + one-hot categoricals (fit on train)."""
    scaler = StandardScaler()
    num_tr = scaler.fit_transform(train_df[_numeric_columns(train_df)])
    num_te = scaler.transform(test_df[_numeric_columns(test_df)])

    cat_tr = pd.get_dummies(train_df[CATEGORICAL_FEATURES])
    cat_te = pd.get_dummies(test_df[CATEGORICAL_FEATURES]).reindex(
        columns=cat_tr.columns, fill_value=0)

    X_tr = np.hstack([num_tr, cat_tr.values.astype(np.float32)]).astype(np.float32)
    X_te = np.hstack([num_te, cat_te.values.astype(np.float32)]).astype(np.float32)
    return X_tr, X_te


def load_full_dataset():
    """Official 175,341 / 82,332 split. Returns (Xtr, ytr, Xte, yte)."""
    train_df = pd.read_csv(FULL_TRAIN)
    test_df = pd.read_csv(FULL_TEST)
    print(f"official training set: {train_df.shape}, testing set: {test_df.shape}")

    train_df, _ = _clean(train_df)
    test_df, _ = _clean(test_df)
    X_tr, X_te = _fit_transform(train_df, test_df)
    y_tr = train_df["label"].astype(int).values
    y_te = test_df["label"].astype(int).values
    print(f"features: {X_tr.shape[1]} | train label counts {np.bincount(y_tr)}, "
          f"test label counts {np.bincount(y_te)}")
    return X_tr, y_tr, X_te, y_te


def load_sample_dataset(data_path=SAMPLE_CSV):
    """Small bundled sample, split 60/20/20 like the reference protocol."""
    df = pd.read_csv(data_path)
    df, _ = _clean(df)
    print(f"sample dataset shape: {df.shape}")
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE,
                              stratify=df["label"].values)
    X, _ = _fit_transform(df, df)
    y = df["label"].astype(int).values
    return X[tr], y[tr], X[te], y[te]


# ---------------------------------------------------------------------------
# graph construction
# ---------------------------------------------------------------------------
def build_knn_adjacency(X, k_neighbors=3, verbose=True):
    """Symmetric-free KNN graph with gaussian-kernel weights."""
    adj = kneighbors_graph(X, k_neighbors, mode="distance", include_self=False,
                           n_jobs=-1)
    rows, cols = adj.nonzero()
    dist = np.asarray(adj[rows, cols]).ravel()
    weights = np.exp(-dist)
    adjacency = sp.csr_matrix((weights, (rows, cols)), shape=adj.shape)
    if verbose:
        print(f"KNN graph: {adj.shape[0]} nodes, {adjacency.nnz} edges")
    return adjacency


def normalized_adjacency_sparse(adjacency, add_self_loops=True):
    a = adjacency.tocsr().astype(np.float32)
    if add_self_loops:
        a = a + sp.eye(a.shape[0], format="csr", dtype=np.float32)
    a = 0.5 * (a + a.T)  # symmetrise
    deg = np.asarray(a.sum(axis=1)).ravel()
    d_inv_sqrt = sp.diags(np.power(np.maximum(deg, 1e-12), -0.5))
    return (d_inv_sqrt @ a @ d_inv_sqrt).tocsr().astype(np.float32)


def scipy_to_torch_sparse(mat):
    import torch
    coo = mat.tocoo()
    indices = np.vstack([coo.row, coo.col])
    values = coo.data.astype(np.float32)
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()


def make_masks(n, y=None, seed=RANDOM_STATE, val_size=0.2):
    idx = np.arange(n)
    train_idx, val_idx = train_test_split(
        idx, test_size=val_size, random_state=seed,
        stratify=y if y is not None else None)
    masks = {}
    for name, part in [("train", train_idx), ("val", val_idx)]:
        m = np.zeros(n, dtype=bool)
        m[part] = True
        masks[name] = m
    return masks


# ---------------------------------------------------------------------------
# Jordan decomposition
# ---------------------------------------------------------------------------
class JordanMatrixDecomposition:
    """Leading eigen-decomposition (approximate Jordan form) of the adjacency.

    Uses a sparse solver so that the full dataset can be processed, and exposes
    `apply()` which propagates a vector without forming an N x N matrix.
    """

    def __init__(self, adj_matrix, n_components=26, solver="arpack",
                 max_iter=20000, tol=1e-2, preprocess=True, use_pca_init=True,
                 n_pca_components=100):
        self.adj_matrix = adj_matrix
        self.n_components = n_components
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.preprocess = preprocess
        self.use_pca_init = use_pca_init
        self.n_pca_components = n_pca_components
        self.jordan_form, self.trans_matrix = self._compute_jordan_form()
        self.eigenvalues = np.diag(self.jordan_form)
        self._pinv_Q = None

    def _compute_jordan_form(self):
        matrix = self.adj_matrix.asfptype()
        if self.preprocess:
            matrix = matrix + sp.eye(matrix.shape[0], format="csr") * 1e-10
            matrix = 0.5 * (matrix + matrix.T)

        k = int(min(self.n_components, matrix.shape[0] - 2))
        if self.solver == "lobpcg" and self.use_pca_init:
            svd = TruncatedSVD(n_components=min(self.n_pca_components,
                                                matrix.shape[0] - 1),
                               random_state=RANDOM_STATE)
            X0 = svd.fit_transform(matrix)[:, :k]
            eigenvalues, eigenvectors = lobpcg(matrix, X0, largest=True,
                                               maxiter=self.max_iter,
                                               tol=self.tol, verbosityLevel=0)
        else:
            eigenvalues, eigenvectors = eigsh(matrix, k=k, which="LM",
                                              maxiter=self.max_iter, tol=self.tol)

        order = np.argsort(-np.abs(eigenvalues))
        return np.diag(eigenvalues[order]), eigenvectors[:, order]

    @property
    def pinv_Q(self):
        if self._pinv_Q is None:
            self._pinv_Q = np.linalg.pinv(self.trans_matrix)
        return self._pinv_Q

    def apply(self, v, t=1, continuous=True):
        """Propagate state vector v through the Jordan form (P @ v)."""
        coef = np.exp(self.eigenvalues * t) if continuous else self.eigenvalues ** t
        return self.trans_matrix @ (coef * (self.pinv_Q @ v))

    def predict_infection(self, initial_state, t=1, continuous=True):
        return self.apply(initial_state, t, continuous)
