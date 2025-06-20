import torch
import math

def reconstruct_matrix_from_graph(edge_index, edge_attr, x, num_nodes):
    """从图数据重建稀疏矩阵"""
    # 获取所有边的索引和权重
    row_indices = edge_index[0]
    col_indices = edge_index[1]
    values = edge_attr.squeeze()
    
    # 添加对角线元素
    diag_indices = torch.arange(num_nodes, device=edge_index.device)
    diag_values = x.squeeze()
    
    # 合并所有索引和值
    all_row_indices = torch.cat([row_indices, diag_indices])
    all_col_indices = torch.cat([col_indices, diag_indices])
    all_values = torch.cat([values, diag_values])
    
    # 创建稀疏矩阵
    indices = torch.stack([all_row_indices, all_col_indices])
    A = torch.sparse_coo_tensor(indices, all_values, (num_nodes, num_nodes))
    return A.coalesce()


def damping_factor(A, omega, diagonal_value, xy=None, exact=False):
    """
    Calculates the damping factor for matrix A when using weight omega
    and diagonal_value is the vector of diagonal values

    Returns max eignenvalue of I - omega * (D \ A),

    If exact = True, the return value is calculated by pytorch eigvals and solve functions.
    This cannot be used during training since there is not autograd support in pytorch for solve or eigvals.
    NOTE: DUE TO POOR SPARSE MATRIX SUPPORT IN PYTORCH, SETTING exact=True CONVERTS A TO A DENSE MATRIX

    If exact is False, uses the power method to obtain the max eigenvalue.
    This preserves the autograd support and sparsity so it can be used in training
    """
    N = A.shape[0]
    if exact:
        # Gives the "exact" damping factor using pytorch's eigvals function
        # Convert A to dense and use dense D due to lack of sparse matrix eigenvals support in pytorch
        Adense = A.to_dense()
        D = torch.diagflat(diagonal_value)
        J = torch.eye(N, device=A.device) - omega*(torch.linalg.solve(D, Adense))
        df = max(abs(torch.linalg.eigvals(J)))
    else:
        # See Taghibakhshi et al. Learning Interface Conditions in Domain Decomposition Solvers for explaination of
        # the eigval_approx method
        K, m = 3, 20
        T = build_error_matrix(A, diagonal_value, omega)
        df = eigval_approx(K, m, T, xy=xy, method='high_freq')
        # df = eigval_approx(K, m, T, method='uniform')
        # Use the power method - allows for autograd and no conversion to dense
        # x = torch.ones((N,1), device=A.device)

        # nits = 30
        # for _ in range(nits):
        #     x = jacobi_dl(diagonal_value, omega, A, x)
        #     xnrm = torch.sqrt(x.t() @ x)
        #     xnrm_torch = torch.linalg.vector_norm(x)
        #     if xnrm_torch - xnrm > 10e-6:
        #         print(f'norm difference: {xnrm-xnrm_torch}')
        #     x = x / xnrm

        # ritz_top = x.t() @ jacobi_dl(diagonal_value, omega, A, x)
        # ritz_bottom = 1

        # df = torch.abs(ritz_top / ritz_bottom)

    return df

def build_error_matrix(A, diag, omega):
    A = A.coalesce()
    A_ind = A.indices()
    A_val = A.values()
    # 确保diag是一维张量
    diag = diag.flatten()
    T_val = -omega*(1/diag[A_ind[0,:]].reshape(-1))*A_val
    eye_ind = torch.stack([torch.arange(A.shape[0], device=A.device), torch.arange(A.shape[0], device=A.device)])
    eye_val = torch.tensor([1 for _ in range(A.shape[0])], device=A.device, dtype=A.dtype)
    T_ind = torch.cat((A_ind, eye_ind), dim=1)
    T_val = torch.cat((T_val, eye_val), dim=0)
    T = torch.sparse_coo_tensor(T_ind, T_val, A.shape)
    return T.coalesce()
        

def eigval_approx(K, m, A, xy=None, method='uniform'):
    """Calculate max(norm(A^K x)**(1/K)) for m x's - an approximation to the max eigenvalue
    of A based on Gelfands formula.
    See Taghibakhshi et al. Learning Interface Conditions in Domain Decomposition Solvers
    
    If Y is given, it's assumed to be such that all it's columns are normalized and method is ignored

    Current recognized methods for generating Y are 'uniform' and 'high_freq'
    """

    # Random test vectors
    if method == 'uniform':
        Y = get_random_on_sphere(A.shape[1], m, device=A.device)
    elif method == 'high_freq':
        Y = get_random_high_freq(A.shape[1], m, xy, device=A.device)
    else:
        raise ValueError(f'method {method} not recognized')
    
    # Calculate A**K * x for all the x's
    # for _ in range(K):
    #     Y = A @ Y # origin
    for _ in range(K):
        if A.is_sparse:
            Y = torch.sparse.mm(A, Y)
        else:
            Y = A @ Y

    # Return the max (norm of A^K x)**(1/K) among all x's
    return torch.max(torch.norm(Y, dim=0))**(1/K)

def get_random_on_sphere(N, m, device=None):
    """Generates m random vectors unifromly distributed on the unit sphere in N dimensions"""

    # Generate random vectors in a gaussian distribution
    Y = torch.randn((N, m), dtype=torch.float, device=device)
    # Normalize each vector to a length of 1
    Y = normalize_vectors(Y)
    return Y

def get_random_high_freq(N, m, xy=None, device=None):
    """Generates m random vectors unifromly distributed across the high-frequency Fourier modes"""
    n = int(math.sqrt(N))
    if xy is None:
        # Generate xx and yy
        xx = torch.zeros(n*n, 1, device=device)
        yy = torch.zeros(n*n, 1, device=device)
        for j in range(n):
            for i in range(n):
                me = (j-1)*n + i
                xx[me] = (i+1)/(n+1)
                yy[me] = (j+1)/(n+1)
    else:
        xx = xy[:,0]
        yy = xy[:,1]
    
    # Chose m vectors with high-freq thetas
    Y = torch.zeros(N, m, device=device)
    nHigh = 0
    while (nHigh < m):
        thetax, thetay = (n-1)*torch.rand(2, device=device) + 1
        # Only add high-freq ones (keep looping until m of them)
        if thetax > n/2 or thetay > n/2:
            # Create the vector
            t = torch.sin(thetax*math.pi*xx)*torch.sin(thetay*math.pi*yy)
            # Add the vector to Y
            Y[:,nHigh] = t.squeeze()
            # Increment the counter so we know when we have enough vectors
            nHigh += 1
    # Normalize
    Y = normalize_vectors(Y)
    return Y

def normalize_vectors(Y):
    # col_norm = torch.linalg.vector_norm(Y, dim=0) # origin
    col_norm = torch.norm(Y, dim=0) # liang
    eplison = 1e-8
    col_norm = torch.clamp(col_norm, min=eplison)
    col_scale = torch.diagflat(1/col_norm) 
    Y = Y @ col_scale
    return Y
    

def jacobi_dl(dvals, omega, A, x):
    Ax = A @ x
    DinvA = Ax / dvals
    y = x - omega*DinvA
    return y

def loss_batch(model, batch):
    """
    Calculates the average damping factor for each graph/matrix in the batch using the output from the GNN model
    """
    vertex_attr, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, [], batch.batch)
    omega = 2./3.
    loss = 0.

    # Whether to use exact damping factor or not
    exact = not model.training
    exact = False
    for i in range(batch.num_graphs):
        # 获取当前图的数据
        node_mask = batch.batch == i
        current_nodes = torch.sum(node_mask).item()
        
        # 获取当前图的边数据
        edge_mask = (batch.edge_index[0] < current_nodes) & (batch.edge_index[1] < current_nodes)
        if i > 0:
            # 调整边索引以适应当前图
            prev_nodes = torch.sum(batch.batch < i).item()
            edge_mask = (batch.edge_index[0] >= prev_nodes) & (batch.edge_index[0] < prev_nodes + current_nodes) & \
                       (batch.edge_index[1] >= prev_nodes) & (batch.edge_index[1] < prev_nodes + current_nodes)
            current_edge_index = batch.edge_index[:, edge_mask] - prev_nodes
        else:
            current_edge_index = batch.edge_index[:, edge_mask]
        
        current_edge_attr = batch.edge_attr[edge_mask]
        current_x = vertex_attr[node_mask]
        
        # 重建矩阵
        A = reconstruct_matrix_from_graph(current_edge_index, current_edge_attr, current_x, current_nodes)
        
        xy = batch.coords[i]
        if xy.dim() == 1:
            # 如果是一维张量，重新reshape成二维
            xy = xy.view(-1, 2)
        dvals = current_x.squeeze()
        
        df = damping_factor(A, omega, dvals, xy=xy, exact=exact)
        loss += df

    return loss / batch.num_graphs

def loss_optimal_jacobi(batch):
    """
    Calculates the loss using the optimal omega value

    WARNING: converts A to dense matrix
    """
    loss = 0
    for i in range(batch.num_graphs):
        A = batch.matrix[i]
        dvals = torch.diag(A.to_dense())
        omega = optimal_omega(A, dvals)
        df = damping_factor(A, omega, dvals, exact=True)
        loss += df
    loss = loss / batch.num_graphs
    return loss

def optimal_omega(A, dvals):
    """
    Returns the optimal omega value

    WARNING: Converts A to dense matrix
    """
    D = torch.diagflat(dvals)
    DinvA = torch.linalg.solve(D, A.to_dense())
    EVals = torch.linalg.eigvals(DinvA)
    lmax = max(abs(EVals))
    lmin = min(abs(EVals))

    return 2 / (lmax + lmin)
