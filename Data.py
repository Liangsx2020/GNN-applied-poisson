import torch
import random
from torch_geometric.data import Data, Dataset, DataLoader

# ===================================================================
# 步骤 1: 整合所有必要的函数定义
# ===================================================================

# 离散泊松矩阵代码
def gen_2d_poisson_matrix(N):
    """使用二阶精度单侧差分处理Neumann边界"""
    total_size = N * N
    row_indices = []
    col_indices = []
    values = []

    for j in range(N):
        for i in range(N):
            idx = j * N + i

            # Dirichlet边界 (顶部)
            if j == N - 1:
                row_indices.append(idx)
                col_indices.append(idx)
                values.append(1.0)

            # Neumann边界处理
            elif j == 0:  # 底边 ∂p/∂y = 0
                if i == 0:  # 底左角
                    # x方向: ∂p/∂x = 0 使用 (-3p[0] + 4p[1] - p[2])/(2h) = 0
                    # y方向: ∂p/∂y = 0 使用 (-3p[0] + 4p[1] - p[2])/(2h) = 0
                    row_indices.extend([idx, idx, idx, idx])
                    col_indices.extend([idx, 
                                      j * N + (i + 1),      # 右邻
                                      j * N + (i + 2),      # 右邻的邻
                                      (j + 1) * N + i])     # 上邻
                    values.extend([6.0, -4.0, 1.0, -3.0])  # x方向2阶 + y方向1阶混合

                elif i == N - 1:  # 底右角
                    row_indices.extend([idx, idx, idx, idx])
                    col_indices.extend([idx,
                                      j * N + (i - 1),      # 左邻
                                      j * N + (i - 2),      # 左邻的邻
                                      (j + 1) * N + i])     # 上邻
                    values.extend([6.0, -4.0, 1.0, -3.0])

                else:  # 底边内部点
                    # 只处理y方向Neumann: (-3p[i,0] + 4p[i,1] - p[i,2])/(2h) = 0
                    # x方向保持标准中心差分
                    row_indices.extend([idx, idx, idx, idx, idx])
                    col_indices.extend([idx,
                                      j * N + (i - 1),      # 左邻
                                      j * N + (i + 1),      # 右邻  
                                      (j + 1) * N + i,      # 上邻
                                      (j + 2) * N + i])     # 上邻的邻
                    values.extend([5.0, -1.0, -1.0, -4.0, 1.0])

            elif i == 0 and j != N - 1:  # 左边 ∂p/∂x = 0
                # x方向Neumann: (-3p[0,j] + 4p[1,j] - p[2,j])/(2h) = 0
                # y方向标准中心差分
                row_indices.extend([idx, idx, idx, idx, idx])
                col_indices.extend([idx,
                                  j * N + (i + 1),          # 右邻
                                  j * N + (i + 2),          # 右邻的邻
                                  (j - 1) * N + i,          # 下邻
                                  (j + 1) * N + i])         # 上邻
                values.extend([5.0, -4.0, 1.0, -1.0, -1.0])

            elif i == N - 1 and j != N - 1:  # 右边 ∂p/∂x = 0  
                # x方向Neumann: (3p[N-1,j] - 4p[N-2,j] + p[N-3,j])/(2h) = 0
                row_indices.extend([idx, idx, idx, idx, idx])
                col_indices.extend([idx,
                                  j * N + (i - 1),          # 左邻
                                  j * N + (i - 2),          # 左邻的邻
                                  (j - 1) * N + i,          # 下邻
                                  (j + 1) * N + i])         # 上邻
                values.extend([5.0, -4.0, 1.0, -1.0, -1.0])

            else:  # 内部点：标准5点模板
                row_indices.extend([idx, idx, idx, idx, idx])
                col_indices.extend([idx,
                                  j * N + (i - 1),
                                  j * N + (i + 1), 
                                  (j - 1) * N + i,
                                  (j + 1) * N + i])
                values.extend([4.0, -1.0, -1.0, -1.0, -1.0])

    # 转换为稀疏矩阵
    row_indices = torch.tensor(row_indices, dtype=torch.long)
    col_indices = torch.tensor(col_indices, dtype=torch.long) 
    values = torch.tensor(values, dtype=torch.float64)

    A = torch.sparse_coo_tensor(
        torch.stack([row_indices, col_indices]),
        values,
        (total_size, total_size)
    )

    return A

# 不再使用自定义的 MyData 类，直接使用标准 Data 类
# 我们会在 loss 函数中单独处理 matrix 和 coords
    
# 将矩阵转换为图数据
def convert_matrix_to_graph(A_sparse_coo):
    """Core conversion function: Converts sparse matrix A to GNN input graph format (Data object)."""
    A_sparse_coo = A_sparse_coo.coalesce()
    indices, values = A_sparse_coo.indices(), A_sparse_coo.values()
    num_nodes = A_sparse_coo.shape[0]
    vertex_attr_list = [0.] * num_nodes
    edge_indices, edge_attr_list = [], []

    for i in range(indices.shape[1]):
        row, col = indices[0, i].item(), indices[1, i].item()
        val = values[i].item()
        if row == col:
            vertex_attr_list[row] = val
        else:
            edge_indices.append([row, col])
            edge_attr_list.append(val)

    vertex_attr = torch.tensor(vertex_attr_list, dtype=torch.float).view(-1, 1)
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).view(-1, 1)

    # 使用标准的 Data 类
    graph_data = Data(x=vertex_attr, 
                      edge_index=edge_index,
                      edge_attr=edge_attr)
    
    # 将 matrix 和 coords 作为单独的属性存储，但不会参与批量化
    # 在 loss 函数中我们会单独处理这些
    graph_data.matrix = A_sparse_coo
    N = int(A_sparse_coo.shape[0]**0.5)
    xx, yy = torch.meshgrid(torch.linspace(0, 1, N), torch.linspace(0, 1, N), indexing='xy')
    graph_data.coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    return graph_data


# ===================================================================
# 步骤 2: 定义 Dataset 类，用于生成训练数据
# ===================================================================

class MyPoissonDataset(Dataset):
    """
    Create a Dataset class for your Poisson matrix problem.
    This class inherits from torch_geometric.data.Dataset and automatically handles downloading, processing and loading logic.
    """
    def __init__(self, root, num_matrices, N_min=16, N_max=64, transform=None, pre_transform=None, pre_filter=None):
        self.num_matrices = num_matrices
        self.N_min = N_min
        self.N_max = N_max
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        # Define raw data filenames format
        return [f'my_matrix_{i}.pt' for i in range(self.num_matrices)]

    @property
    def processed_file_names(self):
        # Define processed data filenames format
        return [f'my_graph_data_{i}.pt' for i in range(self.num_matrices)]

    def download(self):
        # This method is called when the "raw" folder does not exist
        # Our task is to generate and save the original matrices
        print(f"Raw data not found, starting generation in '{self.raw_dir}'...")
        for i in range(self.num_matrices):
            # Randomly select a matrix size N within the specified range
            N = random.randint(self.N_min, self.N_max)
            print(f"  Generating matrix {i+1}/{self.num_matrices} (N={N})...")
            
            # Generate your Poisson matrix
            A = gen_2d_poisson_matrix(N)
            
            # Save the original matrix
            torch.save(A, self.raw_paths[i])
        print("Raw data generation complete.")

    def process(self):
        # This method is called when the "processed" folder does not exist
        # Our task is to read raw matrices, convert to graphs, then save
        print(f"Processed data not found, starting processing in '{self.processed_dir}'...")
        for i in range(self.num_matrices):
            print(f"  Processing matrix {i+1}/{self.num_matrices}...")
            
            # Load raw matrix
            A = torch.load(self.raw_paths[i])
            
            # Convert to graph format
            graph_data = convert_matrix_to_graph(A)
            
            # Save processed graph data
            torch.save(graph_data, self.processed_paths[i])
        print("Data processing complete.")

    def len(self):
        # Return number of samples in dataset
        return self.num_matrices

    def get(self, idx):
        # Load and return a processed graph data sample
        data = torch.load(self.processed_paths[idx])
        return data


# ===================================================================
# 步骤 3: 主程序入口 - 实际创建数据集
# ===================================================================
if __name__ == "__main__":
    # --Configuration Parameters--
    dataset_root_dir = './data'
    num_matrices_to_create = 200
    min_N = 8
    max_N = 256

    print(f'About to create dataset in {dataset_root_dir} directory')
    print(f'Will generate {num_matrices_to_create} samples in total')
    print(f'Matrix size N will be randomly selected from range [{min_N}, {max_N}]')

    # Instantiate Dataset class
    my_dataset = MyPoissonDataset(
        root=dataset_root_dir,
        num_matrices=num_matrices_to_create,
        N_min=min_N,
        N_max=max_N
    )

    print("\nDataset created successfully!")
    print(f"You can check the '{dataset_root_dir}' directory.")
    
    # Let's load a sample to check
    print("\nLoading first sample for inspection:")
    first_sample = my_dataset[0]
    print(first_sample)
    print(f"Number of nodes in graph: {first_sample.num_nodes}")
    print(f"Number of edges in graph: {first_sample.num_edges}")
    print(f"Original matrix attached (matrix): {first_sample.matrix.shape}")