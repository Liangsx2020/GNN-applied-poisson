import torch
import random
import os
from torch_geometric.data import Data, Dataset
from gnn_code import MyData

# ===================================================================
# 1. Your matrix generation code and necessary auxiliary functions
# ===================================================================

def gen_2d_poisson_matrix(N):
    """Using second-order one-sided difference for Neumann boundary"""
    total_size = N * N
    row_indices = []
    col_indices = []
    values = []

    for j in range(N):
        for i in range(N):
            idx = j * N + i

            # Dirichlet boundary (top)
            if j == N - 1:
                row_indices.append(idx)
                col_indices.append(idx)
                values.append(1.0)

            # Neumann boundary handling
            elif j == 0:  # Bottom edge ∂p/∂y = 0
                if i == 0:  # Bottom left corner
                    # x direction: ∂p/∂x = 0 using (-3p[0] + 4p[1] - p[2])/(2h) = 0
                    # y direction: ∂p/∂y = 0 using (-3p[0] + 4p[1] - p[2])/(2h) = 0
                    row_indices.extend([idx, idx, idx, idx])
                    col_indices.extend([idx, 
                                      j * N + (i + 1),      # Right neighbor
                                      j * N + (i + 2),      # Neighbor's neighbor on right
                                      (j + 1) * N + i])     # Upper neighbor
                    values.extend([6.0, -4.0, 1.0, -3.0])  # 2nd order x direction + 1st order y direction mixed

                elif i == N - 1:  # Bottom right corner
                    row_indices.extend([idx, idx, idx, idx])
                    col_indices.extend([idx,
                                      j * N + (i - 1),      # Left neighbor
                                      j * N + (i - 2),      # Neighbor's neighbor on left
                                      (j + 1) * N + i])     # Upper neighbor
                    values.extend([6.0, -4.0, 1.0, -3.0])

                else:  # Interior points on bottom edge
                    # Only handle y direction Neumann: (-3p[i,0] + 4p[i,1] - p[i,2])/(2h) = 0
                    # Keep standard central difference for x direction
                    row_indices.extend([idx, idx, idx, idx, idx])
                    col_indices.extend([idx,
                                      j * N + (i - 1),      # Left neighbor
                                      j * N + (i + 1),      # Right neighbor
                                      (j + 1) * N + i,      # Upper neighbor
                                      (j + 2) * N + i])     # Upper neighbor's neighbor
                    values.extend([5.0, -1.0, -1.0, -4.0, 1.0])

            elif i == 0 and j != N - 1:  # Left edge ∂p/∂x = 0
                # x direction Neumann: (-3p[0,j] + 4p[1,j] - p[2,j])/(2h) = 0
                # Standard central difference for y direction
                row_indices.extend([idx, idx, idx, idx, idx])
                col_indices.extend([idx,
                                  j * N + (i + 1),          # Right neighbor
                                  j * N + (i + 2),          # Right neighbor's neighbor
                                  (j - 1) * N + i,          # Lower neighbor
                                  (j + 1) * N + i])         # Upper neighbor
                values.extend([5.0, -4.0, 1.0, -1.0, -1.0])

            elif i == N - 1 and j != N - 1:  # Right edge ∂p/∂x = 0  
                # x direction Neumann: (3p[N-1,j] - 4p[N-2,j] + p[N-3,j])/(2h) = 0
                row_indices.extend([idx, idx, idx, idx, idx])
                col_indices.extend([idx,
                                  j * N + (i - 1),          # Left neighbor
                                  j * N + (i - 2),          # Left neighbor's neighbor
                                  (j - 1) * N + i,          # Lower neighbor
                                  (j + 1) * N + i])         # Upper neighbor
                values.extend([5.0, -4.0, 1.0, -1.0, -1.0])

            else:  # Interior points: standard 5-point stencil
                row_indices.extend([idx, idx, idx, idx, idx])
                col_indices.extend([idx,
                                  j * N + (i - 1),
                                  j * N + (i + 1), 
                                  (j - 1) * N + i,
                                  (j + 1) * N + i])
                values.extend([4.0, -1.0, -1.0, -1.0, -1.0])

    # Convert to sparse matrix
    row_indices = torch.tensor(row_indices, dtype=torch.long)
    col_indices = torch.tensor(col_indices, dtype=torch.long) 
    values = torch.tensor(values, dtype=torch.float32)

    A = torch.sparse_coo_tensor(
        torch.stack([row_indices, col_indices]),
        values,
        (total_size, total_size)
    ).coalesce()
    return A



def convert_matrix_to_graph(A_sparse_coo):
    """Core conversion function: Convert sparse matrix A to GNN input graph format."""
    indices, values = A_sparse_coo.coalesce().indices(), A_sparse_coo.coalesce().values()
    num_nodes = A_sparse_coo.shape[0]
    vertex_attr = torch.zeros(num_nodes, 1, dtype=torch.float32)

    diag_mask = indices[0] == indices[1]
    vertex_attr[indices[0][diag_mask]] = values[diag_mask].view(-1, 1)

    off_diag_mask = ~diag_mask
    edge_index_tensor = indices[:, off_diag_mask]
    edge_attr = values[off_diag_mask].view(-1, 1)

    graph_data = MyData(x=vertex_attr, edge_index=edge_index_tensor, edge_attr=edge_attr)
    # 不直接存储稀疏矩阵，而是存储重建所需的信息
    graph_data.num_nodes = num_nodes
    N = int(num_nodes**0.5)
    graph_data.grid_size = N
    xx, yy = torch.meshgrid(torch.linspace(0, 1, N), torch.linspace(0, 1, N))
    graph_data.coords = torch.stack([xx.flatten(), yy.flatten()], dim=1).float()
    return graph_data

class MyPoissonDataset(Dataset):
    """Create a Dataset class for your own Poisson matrix problem."""
    def __init__(self, root, num_matrices, N_min=16, N_max=64, transform=None, pre_transform=None, pre_filter=None):
        self.num_matrices, self.N_min, self.N_max = num_matrices, N_min, N_max
        super().__init__(root, transform, pre_transform, pre_filter)
    @property
    def raw_file_names(self): return [f'my_matrix_{i}.pt' for i in range(self.num_matrices)]
    @property
    def processed_file_names(self): return [f'my_graph_{i}.pt' for i in range(self.num_matrices)]
    def download(self):
        print(f"Generating raw matrices in '{self.raw_dir}'...")
        os.makedirs(self.raw_dir, exist_ok=True)
        for i in range(self.num_matrices):
            N = random.randint(self.N_min, self.N_max)
            print(f"  > Generating matrix {i+1}/{self.num_matrices} (N={N})...", end='\r')
            A = gen_2d_poisson_matrix(N)
            torch.save(A, self.raw_paths[i])
        print("\nRaw matrix generation complete.")
    def process(self):
        print(f"Processing matrices into graph format in '{self.processed_dir}'...")
        os.makedirs(self.processed_dir, exist_ok=True)
        for i in range(self.num_matrices):
            print(f"  > Processing graph {i+1}/{self.num_matrices}...", end='\r')
            A = torch.load(self.raw_paths[i])
            # 确保矩阵是float32类型
            A = A.float()
            graph_data = convert_matrix_to_graph(A)
            torch.save(graph_data, self.processed_paths[i])
        print("\nGraph data processing complete.")
    def len(self): return self.num_matrices
    def get(self, idx): return torch.load(self.processed_paths[idx])

# ===================================================================
# 2. Main program entry - Actually create dataset
# ===================================================================
if __name__ == '__main__':
    dataset_root_dir = './data'
    if os.path.exists(os.path.join(dataset_root_dir, 'processed')):
        print(f"Processed data found in '{dataset_root_dir}/processed' directory, no need to recreate.")
    else:
        my_dataset = MyPoissonDataset(root=dataset_root_dir, num_matrices=200, N_min=16, N_max=64)
        print("\nDataset creation successful!")