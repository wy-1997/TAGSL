import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, pdb, sys, argparse
from dgc.utils import load_graph_data, normalize_adj, construct_filter, normalize_adj_torch, normalize_adj_torch_sparse
import scipy.sparse as sp
from model import *
from dgc.clustering import k_means
from dgc.eval import print_eval, match_cluster, print_eval_simple
from dgc.rand import setup_seed
from datetime import datetime
import time
from distutils.util import strtobool
import sys
# sys.path.insert(0, './DeProp')
# from DeProp.model import DeProp
setup_seed(42)
torch.autograd.set_detect_anomaly(True)
start_time = time.time()
print('start time:', datetime.now())
from utils import cluster_id2assignment, Cprop
cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import CyclicLR
from torch_geometric.nn.models.mlp import MLP



def spectral_clu(H, cluster_num):
    from sklearn.cluster import SpectralClustering
    clustering = SpectralClustering(n_clusters=cluster_num, assign_labels='kmeans').fit(H)
    cluster_id = clustering.labels_
    cluster_centers = np.zeros((cluster_num, H.shape[1]))
    for i in range(cluster_num):
        cluster_centers[i] = H[cluster_id==i].mean(0)
    return cluster_id, cluster_centers






parser = argparse.ArgumentParser(description='TAGSL graph clustering')
### training params ###
parser.add_argument('--dataset', type=str, default='texas', help='name of dataset')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--t_lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--a_lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--t_wd', type=float, default=5e-4, help='')
parser.add_argument('--a_wd', type=float, default=5e-4, help='')
parser.add_argument('--device', type=str, default='cuda', help='device')
parser.add_argument('--dropout', type=float, default=0.0, help='')
parser.add_argument('--fold', type=str, default='0-1-2-3-4', help='num of repeats, and seed of each repeat, separated by -')


parser.add_argument('--hidden_dim', type=int, default=512, help="hidden dimension")
parser.add_argument('--emb_dim', type=int, default=64, help="hidden dimension")


parser.add_argument('--input_encoder', type=str, default='svd', help='svd, lin, or mlp')
parser.add_argument('--svd_on_S', type=str, default='S_norm', help='obtain input features by svd on S or S_norm')
parser.add_argument('--svd_on_A', type=str, default='A_norm', help='obtain input features by svd on S or A_norm')


### attr_agg params ###
parser.add_argument('--attr_layers', type=int, default=5, help='')
parser.add_argument('--attr_alpha', type=float, default=0.5, help='')
parser.add_argument('--attr_r', type=float, default=1.0, help='the nnz ratio of attr simi mtx')
parser.add_argument('--attr_bin', type=int, default=0, help='the nnz ratio of attr simi mtx')
parser.add_argument('--attr_prop', type=str, default='sgc', help='sgc style, gcn style, or appnp style')
parser.add_argument('--attr_linear_trans', type=str, default='mlp', help='lin or mlp')



### fusion params ###
parser.add_argument('--fusion_norm', type=str, default='none', help='if l2-norm, l2 normalization on Ht and HA before fusion, if none, no post process')
parser.add_argument('--fusion_method', type=str, default='add', help='add, concat, max')
parser.add_argument('--fusion_beta', type=float, default=0.5, help='H = beta * H_t + (1-beta) * H_a')


### X_prop params ###
parser.add_argument('--xprop_layers', type=int, default=5, help='')
parser.add_argument('--xprop_alpha', type=float, default=0.2, help='')

### C_prop params ###
parser.add_argument('--cprop_layers', type=int, default=1, help='')
parser.add_argument('--cprop_alpha', type=float, default=1.0, help='')
parser.add_argument('--cprop_abl', type=int, default=0, help='')


### loss params ###
parser.add_argument('--loss_lambda_prop', type=float, default=1, help='')
parser.add_argument('--sharpening', type=float, default=1, help="")
parser.add_argument('--loss_lambda_kmeans', type=float, default=0.02, help='')
parser.add_argument('--kmeans_loss', type=str, default='cen', 
                    help='tr(ace), cen(troid contrastive), nod(e contrastive)')
parser.add_argument('--loss_lambda_SSG0', type=float, default=0.0001, help='')
parser.add_argument('--loss_lambda_SSG1', type=float, default=0.006, help='')
parser.add_argument('--loss_lambda_SSG2', type=float, default=0.006, help='')
parser.add_argument('--loss_lambda_SSG3', type=float, default=0.006, help='')
parser.add_argument('--temperature', type=float, default=2.0, help='') 
parser.add_argument('--clu_size', type=strtobool, default=True, help='') 
parser.add_argument('--norm', type=int, default=1, help='')
parser.add_argument('--rounding', type=int, default=0, help='')


### log params ###
parser.add_argument('--log_file', type=str, default=None, help='')
parser.add_argument('--log_fold_file', type=str, default=None, help='')
parser.add_argument('--save_model', type=strtobool, default=False, help='')






# =========================================================
# TAGSL topology-branch hyperparameters
# =========================================================

parser.add_argument(
    '--tagsl_alpha',
    type=float,
    default=1.0,
    help=(
        'Weight of the dynamic structural-operator term.'
    ),
)

parser.add_argument(
    '--tagsl_beta',
    type=float,
    default=1.0,
    help=(
        'Weight of the topology-aware tensor-affinity term.'
    ),
)

parser.add_argument(
    '--tagsl_landmark_size',
    type=int,
    default=100,
    help=(
        'Number of Nystrom landmark tuples.'
    ),
)

parser.add_argument(
    '--tagsl_backend',
    type=str,
    default='dense_exact',
    choices=[
        'dense_exact',
        'implicit_projector',
    ],
    help=(
        'Structural backend. dense_exact follows the explicit '
        'Q update and is intended for small/medium graphs. '
        'implicit_projector is the scalable engineering path.'
    ),
)

parser.add_argument(
    '--tagsl_affinity_mode',
    type=str,
    default='topology_aware',
    choices=[
        'feature_only',
        'one_hop',
        'two_hop',
        'topology_aware',
    ],
    help=(
        'Affinity mode used by the TAGSL Nystrom operator.'
    ),
)

parser.add_argument(
    '--tagsl_scale_mode',
    type=str,
    default='none',
    choices=[
        'none',
        'divide_by_m',
        'row_normalize',
    ],
    help=(
        'Output scaling convention for the Nystrom basis.'
    ),
)

parser.add_argument(
    '--tagsl_sigma',
    type=float,
    default=1.0,
    help=(
        'Sensitivity coefficient in the topology-aware '
        'tensor affinity.'
    ),
)

parser.add_argument(
    '--tagsl_node_chunk_size',
    type=int,
    default=2048,
    help=(
        'Number of graph nodes processed in each streaming '
        'Nystrom node block.'
    ),
)

parser.add_argument(
    '--tagsl_landmark_chunk_size',
    type=int,
    default=32,
    help=(
        'Number of landmark tuples processed in each '
        'streaming Nystrom landmark block.'
    ),
)

parser.add_argument(
    '--tagsl_q_lr',
    type=float,
    default=1e-3,
    help=(
        'Explicit learning rate for the TAGSL Q block.'
    ),
)

parser.add_argument(
    '--tagsl_h_lr',
    type=float,
    default=1e-3,
    help=(
        'Explicit learning rate for the TAGSL H block.'
    ),
)

parser.add_argument(
    '--tagsl_orthogonality_mode',
    type=str,
    default='column_strict',
    choices=[
        'column_strict',
        'shape_preserving',
    ],
    help=(
        'QR-retraction convention. column_strict enforces '
        'H.T @ H = I and requires N >= embedding_dim.'
    ),
)

parser.add_argument(
    '--tagsl_recon_weight',
    type=float,
    default=1.0,
    help=(
        'Weight of the coupling loss between differentiable '
        'topology features XW and the explicit TAGSL state H.'
    ),
)

parser.add_argument(
    '--tagsl_max_dense_elements',
    type=int,
    default=50000000,
    help=(
        'Maximum number of elements allowed in the explicit '
        'dense Q backend.'
    ),
)


parser.add_argument('--clu_check', type=int, default=1, help='')






args = parser.parse_args()
if args.log_file is None:
    args.log_file = 'res_log/'+args.dataset+'_grid_search.txt'
if args.log_fold_file is None:
    args.log_fold_file = 'res_fold_log/'+args.dataset+'_grid_search.txt'

for output_path in (args.log_file, args.log_fold_file):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

print(args)


# =========================================================
# TAGSL graph-preprocessing helpers
# =========================================================

def to_torch_sparse_coo_float32(
    matrix,
    device,
):
    """
    Convert a scipy sparse matrix, NumPy matrix, or torch
    tensor into a coalesced torch sparse COO tensor.

    The conversion never densifies a scipy sparse adjacency.
    """
    if torch.is_tensor(matrix):
        if matrix.layout == torch.sparse_coo:
            sparse_matrix = (
                matrix
                .coalesce()
                .to(
                    device=device,
                    dtype=torch.float32,
                )
            )

        elif matrix.layout == torch.strided:
            if matrix.ndim != 2:
                raise ValueError(
                    'A dense adjacency must be two-dimensional.'
                )

            sparse_matrix = (
                matrix
                .to(dtype=torch.float32)
                .to_sparse()
                .coalesce()
                .to(device)
            )

        else:
            raise TypeError(
                'Unsupported torch adjacency layout: '
                f'{matrix.layout}.'
            )

    elif sp.issparse(matrix):
        matrix_coo = (
            matrix
            .tocoo(copy=False)
        )

        indices_numpy = np.vstack(
            (
                matrix_coo.row,
                matrix_coo.col,
            )
        ).astype(
            np.int64,
            copy=False,
        )

        values_numpy = (
            matrix_coo.data
            .astype(
                np.float32,
                copy=False,
            )
        )

        indices = torch.from_numpy(
            indices_numpy
        )

        values = torch.from_numpy(
            values_numpy
        )

        sparse_matrix = (
            torch.sparse_coo_tensor(
                indices=indices,
                values=values,
                size=matrix_coo.shape,
                dtype=torch.float32,
            )
            .coalesce()
            .to(device)
        )

    else:
        dense_numpy = np.asarray(
            matrix,
            dtype=np.float32,
        )

        if dense_numpy.ndim != 2:
            raise ValueError(
                'A NumPy adjacency must be two-dimensional.'
            )

        row_indices, col_indices = (
            np.nonzero(
                dense_numpy
            )
        )

        indices_numpy = np.vstack(
            (
                row_indices,
                col_indices,
            )
        ).astype(
            np.int64,
            copy=False,
        )

        values_numpy = dense_numpy[
            row_indices,
            col_indices,
        ].astype(
            np.float32,
            copy=False,
        )

        indices = torch.from_numpy(
            indices_numpy
        )

        values = torch.from_numpy(
            values_numpy
        )

        sparse_matrix = (
            torch.sparse_coo_tensor(
                indices=indices,
                values=values,
                size=dense_numpy.shape,
                dtype=torch.float32,
            )
            .coalesce()
            .to(device)
        )

    if (
        sparse_matrix.ndim != 2
        or sparse_matrix.shape[0]
        != sparse_matrix.shape[1]
    ):
        raise ValueError(
            'The adjacency matrix must be square.'
        )

    return sparse_matrix


@torch.no_grad()
def remove_sparse_self_loops(
    adjacency,
):
    """
    Remove diagonal entries without densifying the graph.
    """
    adjacency = adjacency.coalesce()

    indices = adjacency.indices()
    values = adjacency.values()

    off_diagonal_mask = (
        indices[0] != indices[1]
    )

    adjacency_without_loops = (
        torch.sparse_coo_tensor(
            indices=indices[
                :,
                off_diagonal_mask,
            ],
            values=values[
                off_diagonal_mask
            ],
            size=adjacency.shape,
            device=adjacency.device,
            dtype=adjacency.dtype,
        )
        .coalesce()
    )

    return adjacency_without_loops






# =========================================================
# Load graph data
# =========================================================

X_loaded, true_labels, A_loaded = (
    load_graph_data(
        root_path='./dataset/',
        dataset_name=args.dataset,
        show_details=True,
    )
)



# =========================================================
# Explicit one-time graph symmetrization
#
# TAGSL uses one shared undirected adjacency for:
#   1. landmark generation;
#   2. one-hop topology support;
#   3. two-hop topology support A^2.
#
# Element-wise maximum preserves one-sided edge weights
# without doubling edges that are already symmetric.
# =========================================================

if sp.issparse(A_loaded):
    adjacency_nnz_before = int(
        A_loaded.nnz
    )

    A_loaded = (
        A_loaded
        .tocsr(copy=True)
        .maximum(
            A_loaded.T
        )
        .tocsr()
    )

    A_loaded.eliminate_zeros()

    adjacency_nnz_after = int(
        A_loaded.nnz
    )

else:
    A_loaded = np.asarray(
        A_loaded,
        dtype=np.float32,
    )

    if (
        A_loaded.ndim != 2
        or A_loaded.shape[0]
        != A_loaded.shape[1]
    ):
        raise ValueError(
            'A_loaded must be a square adjacency matrix.'
        )

    adjacency_nnz_before = int(
        np.count_nonzero(
            A_loaded
        )
    )

    A_loaded = np.maximum(
        A_loaded,
        A_loaded.T,
    )

    adjacency_nnz_after = int(
        np.count_nonzero(
            A_loaded
        )
    )

print(
    'Adjacency symmetrization: '
    f'nnz={adjacency_nnz_before} '
    f'-> {adjacency_nnz_after}'
)









# The benchmark provides the target number of clusters c.
# Node labels are still retained only for evaluation in the
# current experimental framework.
cluster_num = len(
    np.unique(
        true_labels
    )
)


# =========================================================
# Raw TAGSL adjacency
#
# A_raw_sym:
#   1. sparse COO;
#   2. float32;
#   3. no self-loops;
#   4. symmetric;
#   5. not degree-normalized.
#
# The same tensor will later be passed to:
#   - landmark-tuple generation;
#   - one-hop/two-hop topology support;
#   - TAGSL adjacency consistency.
# =========================================================

A_raw_sym = (
    to_torch_sparse_coo_float32(
        matrix=A_loaded,
        device=args.device,
    )
)

A_raw_sym = (
    remove_sparse_self_loops(
        A_raw_sym
    )
)

if not torch.isfinite(
    A_raw_sym.values()
).all():
    raise ValueError(
        'The raw adjacency contains NaN or Inf.'
    )

if torch.any(
    A_raw_sym.values() < 0
):
    raise ValueError(
        'TAGSL requires nonnegative adjacency weights.'
    )

# Validate rather than silently symmetrize. This prevents
# landmark generation and topology-support lookup from using
# two different graph conventions.
A_raw_sym = (
    tagsl_require_symmetric_adjacency(
        A_raw=A_raw_sym,
        atol=1e-8,
    )
)


# =========================================================
# Compatibility alias for the retained original framework
#
# The old attribute encoder and several existing helper
# functions still refer to the variable name A.
# =========================================================

A = A_raw_sym


# =========================================================
# Edge index derived from the validated raw graph
# =========================================================

edge_index = (
    A_raw_sym
    .indices()
    .detach()
    .clone()
)


# =========================================================
# Normalized adjacency variants retained for the original
# dual-view clustering framework
# =========================================================

A_norm = normalize_adj_torch_sparse(
    A,
    self_loop=True,
    symmetry=True,
)

A_no_loop_sym = normalize_adj_torch_sparse(
    A,
    self_loop=False,
    symmetry=False,
)


# =========================================================
# Node features and original attribute preprocessing
# =========================================================

X = torch.as_tensor(
    np.asarray(
        X_loaded,
        dtype=np.float32,
    ),
    dtype=torch.float32,
    device=args.device,
)

if X.ndim != 2:
    raise ValueError(
        'The node feature matrix X must have shape [N, D].'
    )

if X.shape[0] != A_raw_sym.shape[0]:
    raise ValueError(
        'X and A_raw_sym contain different numbers of nodes.'
    )

if not torch.isfinite(
    X
).all():
    raise ValueError(
        'The node feature matrix contains NaN or Inf.'
    )

X_norm = F.normalize(
    X,
    p=2,
    dim=1,
)


# =========================================================
# Original half-S normalization retained unchanged
#
# half_S_norm @ half_S_norm.T approximates S_norm.
# =========================================================

deg_vec = (
    X_norm
    @ X_norm.sum(
        dim=0
    )
)

deg_vec[
    deg_vec == 0
] = 1

deg_vec = deg_vec.pow(
    -0.5
)

half_S_norm = (
    deg_vec.unsqueeze(1)
    * X_norm
)


# =========================================================
# Early TAGSL configuration diagnostics
# =========================================================

num_nodes = int(
    A_raw_sym.shape[0]
)

dense_q_elements = (
    num_nodes - cluster_num
) * num_nodes

print(
    'TAGSL data preprocessing: '
    f'N={num_nodes}, '
    f'raw_edges={A_raw_sym._nnz()}, '
    f'features={X.shape[1]}, '
    f'clusters={cluster_num}'
)

print(
    'TAGSL configuration: '
    f'backend={args.tagsl_backend}, '
    f'affinity={args.tagsl_affinity_mode}, '
    f'landmarks={args.tagsl_landmark_size}, '
    f'dense_Q_elements={dense_q_elements}'
)

if (
    args.tagsl_backend
    == 'dense_exact'
    and dense_q_elements
    > args.tagsl_max_dense_elements
):
    raise ValueError(
        'The requested dense_exact backend would create '
        f'{dense_q_elements} Q elements, exceeding '
        '--tagsl_max_dense_elements='
        f'{args.tagsl_max_dense_elements}. '
        'Use --tagsl_backend implicit_projector for the '
        'scalable engineering path, or explicitly increase '
        'the guard after checking available memory.'
    )



def train():
    ##### train #####
    best_acc = 0
    best_res = []

    best_loss = 99999999

    # =====================================================
    # TAGSL efficiency log
    # =====================================================

    efficiency_log_path = (
        f'./res_log/'
        f'{args.dataset}_fold{fold}_efficiency_tagsl.csv'
    )
    
    # 确保目录存在
    if not os.path.exists('./res_log/'):
        os.makedirs('./res_log/')
        
    with open(efficiency_log_path, 'w') as f:
        f.write('Epoch,Time(s),Peak_GPU_Mem(MB)\n')
    
    # 重置 GPU 峰值显存统计，确保从训练开始计算
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()





    beta = args.fusion_beta
    for e in range(args.epochs):
        
        epoch_start_time = time.time()  # [新增 2] 计时开始
        optimizer_t.zero_grad()
        optimizer_a.zero_grad()

        
        

        # =================================================
        # TAGSL topology branch
        #
        # top_input remains the output of the retained
        # structural input encoder.
        # =================================================

        top_input = S_encoder()

        top_output = model(
            top_input,
            update_structure=True,
            tensor_mode=args.tagsl_affinity_mode,
            include_rank=False,
            return_diagnostics=False,
        )

        # Differentiable topology representation.
        #
        # This tensor enters the retained fusion and clustering
        # losses, allowing gradients to update both:
        #   1. the TAGSL projection W;
        #   2. S_encoder.
        H_t = top_output[
            'projected_features'
        ]

        # Explicit TAGSL structural state.
        #
        # H_t_struct is updated by the alternating structural
        # procedure and does not carry an autograd graph.
        H_t_struct = top_output[
            'H'
        ]

        # Coupling between the differentiable topology
        # representation XW and the explicit structural state H.
        loss_tagsl_recon = (
            model.projection_reconstruction_loss(
                top_output,
                reduction='mean',
            )
        )
        
        # =================================================
        # First-epoch TAGSL runtime diagnostics
        # =================================================

        if e == 0:
            print(
                'TAGSL runtime state: '
                f'backend={top_output["backend_type"]}, '
                f'tensor_mode={top_output["tensor_mode"]}, '
                f'scale_mode={top_output["scale_mode"]}, '
                f'structure_step={top_output["structure_step"]}, '
                f'projected_shape={tuple(H_t.shape)}, '
                f'structural_shape={tuple(H_t_struct.shape)}, '
                f'projected_requires_grad={H_t.requires_grad}, '
                f'structural_requires_grad={H_t_struct.requires_grad}'
            )

        # =================================================
        # Retained attribute branch and dual-view fusion
        # =================================================

        H_a = attr_model(
            A_encoder()
        )

        H = fusion_attr(
            H_t,
            H_a,
        )


        # =================================================
        # Numerical checks
        # =================================================

        if not torch.isfinite(
            H_t
        ).all():
            raise FloatingPointError(
                'Non-finite values were found in '
                'TAGSL projected_features.'
            )

        if not torch.isfinite(
            H_t_struct
        ).all():
            raise FloatingPointError(
                'Non-finite values were found in '
                'the explicit TAGSL structural state.'
            )

        if not torch.isfinite(
            loss_tagsl_recon
        ).all():
            raise FloatingPointError(
                'The TAGSL reconstruction loss is non-finite.'
            )

        if not torch.isfinite(
            H
        ).all():
            raise FloatingPointError(
                'Non-finite values were found in '
                'the fused representation H.'
            )


        loss_prop = torch.pow(1 - cos_sim(H, X_prop), args.sharpening).mean()

        if e%args.clu_check == 0:
            if args.rounding == 0:
                if e == 0:
                    cluster_ids,centers_xprop = k_means(X_prop.detach().cpu(), cluster_num, device='cpu', distance='cosine')
                    cluster_ids,centers = k_means(H.detach().cpu(), cluster_num, device='cpu', distance='cosine', centers=centers_xprop)
                else:
                    cluster_ids,centers = k_means(H.detach().cpu(), cluster_num, device='cpu', distance='cosine', centers='kmeans')
            else:
                print('rounding')
                cluster_ids,centers = k_means((torch.round(torch.tanh(H)*7)).detach().cpu(), cluster_num, device='cpu', distance='cosine', centers='kmeans')


        C0 = cluster_id2assignment(cluster_ids, cluster_num).to(args.device)
        # C = Cprop(C0, A, args)
        if args.cprop_abl == 1:
            print('cprop_abl')
            C = C0
        else:
            C = C_prop_model(C0)
        # C = C0
        C = F.normalize(C, p=2, dim=1)

        loss_kmeans =  kmeans_loss_fn(H, C, args)
        


        if args.norm == 1:
            print('norm')
            H_t_norm = (H_t - H_t.mean(dim=0)) / H_t.std(dim=0) / torch.sqrt(torch.tensor(H_t.shape[1]).to(H_t.device))
            H_a_norm = (H_a - H_a.mean(dim=0)) / H_a.std(dim=0) / torch.sqrt(torch.tensor(H_a.shape[1]).to(H_a.device))
        elif args.norm == 0:
            H_t_norm = H_t
            H_a_norm = H_a
        ort_loss = (ortho_loss_fn(H_t_norm) + ortho_loss_fn(H_a_norm))
        # inv_loss_o = args.loss_lambda_SSG1 * (H_t_norm - H_a_norm).pow(2).sum()
        inv_loss_o = F.mse_loss(H_t_norm, H_a_norm)
        # inv_loss_o = args.loss_lambda_SSG1 * (H_t_norm - H_a_norm).norm(p=2, dim=1).mean()
        inv_loss_n = (node_t_neighbor_a_loss_fn2(H_t_norm, H_a_norm, A_no_loop_sym) + node_t_neighbor_a_loss_fn2(H_a_norm, H_t_norm, A_no_loop_sym))
        # print(C.sum(0).mean(), C.sum(0).max())
        inv_loss_c = (node_t_cluster_a_loss_fn2(H_t_norm, H_a_norm, C, clu_size=args.clu_size) + node_t_cluster_a_loss_fn2(H_a_norm, H_t_norm, C, clu_size=args.clu_size))
        inv_loss = args.loss_lambda_SSG1 * inv_loss_o + args.loss_lambda_SSG2 * inv_loss_n + args.loss_lambda_SSG3 * inv_loss_c


        # =================================================
        # Hybrid objective
        #
        # Original dual-view objective
        # +
        # TAGSL structural reconstruction coupling
        # =================================================

        loss_original = (
            args.loss_lambda_prop
            * loss_prop
            + args.loss_lambda_kmeans
            * loss_kmeans
            + args.loss_lambda_SSG0
            * ort_loss
            + inv_loss
        )

        loss = (
            loss_original
            + args.tagsl_recon_weight
            * loss_tagsl_recon
        )
        loss.backward()
        optimizer_t.step()
        optimizer_a.step()


        ## evaluation
        print(
            (
                'epoch: %d, '
                'loss: %.3f, '
                'loss_original: %.3f, '
                'tagsl_recon: %.3f, '
                'loss_kmeans: %.3f, '
                'loss_prop: %.3f, '
                'ort_loss: %.3f,'
            )
            % (
                e,
                loss.item(),
                loss_original.item(),
                loss_tagsl_recon.item(),
                loss_kmeans.item(),
                loss_prop.item(),
                ort_loss.item(),
            ),
            end=' ',
        )
        print('inv_loss1: %.3f, inv_loss2: %.3f, inv_loss3: %.3f,' % (inv_loss_o, inv_loss_n, inv_loss_c), end=' ')
        predict_labels = torch.argmax(C, dim=1)
        res = print_eval_simple(predict_labels.cpu().numpy(), true_labels)
        # === [新增 3] 记录时间和显存 ===
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            torch.cuda.reset_peak_memory_stats()
            
        # Append 模式写入文件
        with open(efficiency_log_path, 'a') as f:
            f.write(f'{e},{epoch_duration:.4f},{gpu_memory:.2f}\n')     

        ## best acc
        if res[0] > best_acc:
            best_acc = res[0]
            best_res = res
            best_e = e
            if args.save_model:
                torch.save(H.cpu().detach(), best_model_path+'H.pth')
        if loss < best_loss:
            best_loss = loss
            if args.save_model:
                torch.save(H.cpu().detach(), best_model_path+'best_loss_H.pth')

    print(f'best epoch: {best_e}')
    return best_res




retain_grah=True 
total_res = []
# for fold in range(5):
# final result is the mean of 5 repeated experiments
for fold in [int(x) for x in args.fold.split('-')]:
    setup_seed(43+fold)
    print("#"*60)
    print("#"*26, ' fold:%d ' % fold, "#"*26)
    print("#"*60)


    ##### compute low rank US_norm and UA_norm #####
    S_encoder = input_enc(args.input_encoder, X.shape[0], args.hidden_dim, args.emb_dim).to(args.device)
    A_encoder = input_enc(args.input_encoder, X.shape[0], args.hidden_dim, args.emb_dim).to(args.device)
    # store init US_norm / UA_norm within the model 
    # for svd, use the first mtx; for lin or mlp, use the second mtx
    if args.input_encoder == 'svd':
        pre_saved_path = './dataset/pre_saved_U/'+args.dataset+'/'
        if os.path.exists(pre_saved_path+args.svd_on_S+'.pth'):
            print('load pre-saved U')
            S_encoder.U = torch.load(pre_saved_path+args.svd_on_S+'.pth').to(args.device)
            A_encoder.U = torch.load(pre_saved_path+args.svd_on_A+'.pth').to(args.device)
        else:
            print('compute U')
            os.makedirs(pre_saved_path, exist_ok=True)
            X_cpu = X.cpu()
            S = compute_attr_simi_mtx(X_cpu, args.attr_r, args.attr_bin)
            S_norm = normalize_adj_torch(S, self_loop=False, symmetry=True)
            S_encoder.init_U(eval(args.svd_on_S), None) 
            A_encoder.init_U(eval(args.svd_on_A), None)
            S_encoder.U = S_encoder.U.to(args.device)
            A_encoder.U = A_encoder.U.to(args.device)
            print('save U')
            torch.save(S_encoder.U, pre_saved_path+args.svd_on_S+'.pth')
            torch.save(A_encoder.U, pre_saved_path+args.svd_on_A+'.pth')
    else:
        S_encoder.init_U(None, X@X.t()) 
        A_encoder.init_U(None, A)
        # S_encoder.U = S_encoder.U.to(args.device)
        # A_encoder.U = A_encoder.U.to(args.device)

    # =====================================================
    # Define the TAGSL topology branch
    #
    # Only top_agg_f is replaced in this stage.
    # S_encoder, the attribute branch, fusion, C propagation,
    # and the original dual-view losses remain unchanged.
    # =====================================================

    model = top_agg_f(
        # Raw, symmetric, non-normalized adjacency used by:
        # landmark generation, one-hop support, and A^2 support.
        A_raw=A_raw_sym,

        # S_encoder() returns [N, args.emb_dim].
        input_dim=args.emb_dim,
        embedding_dim=args.emb_dim,

        # Benchmark-level target number of clusters.
        num_clusters=cluster_num,

        # TAGSL structural objective.
        alpha=args.tagsl_alpha,
        beta=args.tagsl_beta,

        # Streaming tensor-affinity approximation.
        landmark_size=args.tagsl_landmark_size,
        structural_backend_type=args.tagsl_backend,
        affinity_mode=args.tagsl_affinity_mode,
        scale_mode=args.tagsl_scale_mode,
        sigma=args.tagsl_sigma,
        node_chunk_size=args.tagsl_node_chunk_size,
        landmark_chunk_size=args.tagsl_landmark_chunk_size,

        # Explicit alternating updates.
        q_learning_rate=args.tagsl_q_lr,
        h_learning_rate=args.tagsl_h_lr,
        q_post_update_order='center_then_dual',

        # QR / semi-orthogonal retraction.
        orthogonality_mode=args.tagsl_orthogonality_mode,

        # Keep one deterministic but fold-dependent trajectory.
        seed=43 + fold,

        # Guard for the explicit dense Q backend.
        max_dense_elements=args.tagsl_max_dense_elements,

        projection_bias=True,
        require_constant_nullspace=True,
    ).to(args.device)

    

    
    
    attr_model = attr_agg_f(half_S_norm, args.attr_alpha, args.attr_layers, args.emb_dim, args.hidden_dim, linear_prop=args.attr_prop, linear_trans=args.attr_linear_trans, norm=args.fusion_norm).to(args.device)

    fusion_attr = fusion(
        args.fusion_method,
        args.fusion_beta,
        args.emb_dim,
    ).to(args.device)
    
    C_prop_model = C_agg_f(args.cprop_alpha, args.cprop_layers, A_norm).to(args.device)


    optimizer_t = torch.optim.Adam(
        list(model.parameters())
        + list(S_encoder.parameters())
        + list(fusion_attr.parameters()),
        lr=args.t_lr,
        weight_decay=args.t_wd,
    )
    optimizer_a = torch.optim.Adam(list(attr_model.parameters())+list(A_encoder.parameters()), lr=args.a_lr, weight_decay=args.a_wd)


    ##### compute low rank X_prop #####
    X_prop = X_norm
    for _ in range(args.xprop_layers):
        X_prop = args.xprop_alpha * torch.spmm(A_norm, X_prop) + X_norm
    U, s, _ = torch.svd_lowrank(X_prop, q=args.emb_dim, niter=7)
    X_prop = U @ torch.diag(s)
    X_prop = F.normalize(X_prop, p=2, dim=1)


    ##### training and evaluation #####
    best_model_path = f'./best_model/{args.dataset}/'
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    best_model_path += f'fold{fold}_'
    res = train()
    total_res.append(res)

    print('fold: %d, acc: %.2f, nmi: %.2f, ari: %.2f, f1: %.2f' % (fold, res[0]*100, res[1]*100, res[2]*100, res[3]*100))


total_res = np.array(total_res)

print('***** final result: *****')
print('%s, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n'%(args.dataset, 
    total_res[:, 0].mean()*100, total_res[:, 0].std()*100, 
    total_res[:, 1].mean()*100, total_res[:, 1].std()*100, 
    total_res[:, 2].mean()*100, total_res[:, 2].std()*100, 
    total_res[:, 3].mean()*100, total_res[:, 3].std()*100))

print(total_res)


total_time = time.time() - start_time
print('total time:', total_time)


# save the aggregate result
with open(args.log_file, 'a+', encoding='utf-8') as f:
    if f.tell() == 0:
        f.write(
            'dataset,acc_mean,acc_std,nmi_mean,nmi_std,'
            'ari_mean,ari_std,f1_mean,f1_std,time\n'
        )
    f.write(
        f'{args.dataset},'
        f'{total_res[:, 0].mean()*100:.2f},'
        f'{total_res[:, 0].std()*100:.2f},'
        f'{total_res[:, 1].mean()*100:.2f},'
        f'{total_res[:, 1].std()*100:.2f},'
        f'{total_res[:, 2].mean()*100:.2f},'
        f'{total_res[:, 2].std()*100:.2f},'
        f'{total_res[:, 3].mean()*100:.2f},'
        f'{total_res[:, 3].std()*100:.2f},'
        f'{total_time:.2f}\n'
    )

# save the result of each fold
with open(args.log_fold_file, 'a+', encoding='utf-8') as f:
    if f.tell() == 0:
        f.write('dataset,fold,acc,nmi,ari,f1,time\n')
    for i in range(total_res.shape[0]):
        f.write(
            f'{args.dataset},{i},'
            f'{total_res[i, 0]*100:.2f},'
            f'{total_res[i, 1]*100:.2f},'
            f'{total_res[i, 2]*100:.2f},'
            f'{total_res[i, 3]*100:.2f},'
            f'{total_time:.2f}\n'
        )



print('end time:', datetime.now())


