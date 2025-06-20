import torch
from gnn_code.TrainableJacobiGNN import get_model # Import model structure from our prepared gnn_code
from create_my_dataset import gen_2d_poisson_matrix, convert_matrix_to_graph # Import conversion function from our data creation script

# ===================================================================
# 1. Your matrix and right-hand side generation code
# ===================================================================

def build_rhs_2d(N, domain_size=1.0):
    h = domain_size / (N - 1)
    
    # Initialize right-hand side vector
    b = torch.zeros(N * N, dtype=torch.float64)
    
    # Use same indexing as matrix generation function: row-major order
    for j in range(N):  # y direction: bottom to top (j=0 bottom, j=N-1 top)
        for i in range(N):  # x direction: left to right (i=0 left, i=N-1 right)
            x = i * h
            y = j * h
            
            # Calculate index: idx = j * N + i (row-major, consistent with matrix function)
            idx = j * N + i
            
            # Only calculate source term for interior points
            if j != 0 and j != N-1 and i != 0 and i != N-1:
                # Calculate second derivatives for interior points only
                d2p_dx2 = 2 * (1 - 6 * x + 6 * x ** 2) * y ** 2 * (1 - y) ** 2
                d2p_dy2 = 2 * (1 - 6 * y + 6 * y ** 2) * x ** 2 * (1 - x) ** 2
                f_val = -(d2p_dx2 + d2p_dy2) * (h ** 2)
                b[idx] = f_val
            else:
                # Set right-hand side to 0 for all boundary points
                b[idx] = 0.0
    
    return b

# ===================================================================
# 2. GNN-accelerated solver function
# ===================================================================
def solve_with_gnn_jacobi(trained_model, A, b, device, num_iterations=500, tolerance=1e-6):
    """Solve Ax = b using Jacobi iteration accelerated by trained GNN model."""
    print("\n--- Starting GNN-accelerated solution ---")
    trained_model.eval()  # Switch to evaluation/inference mode
    A, b = A.to(device), b.to(device)

    # 1. Convert matrix A to GNN graph input format
    print("Converting matrix A to graph...")
    graph_data = convert_matrix_to_graph(A).to(device)

    # 2. GNN forward pass, predict optimal diagonal scaling factors
    print("GNN predicting optimal diagonal preconditioner...")
    with torch.no_grad(): # No gradient computation needed during inference
        # Note: model needs a batch vector, for single graph it's all zeros
        batch_vector = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
        vertex_output, _, _ = trained_model(graph_data.x, graph_data.edge_index, graph_data.edge_attr, [], batch_vector)

    # Calculate final diagonal preconditioner (inverse) based on paper and codebase logic
    learned_d_inv_flat = (2./3.) * (1 / vertex_output.flatten())

    # 3. Execute GNN-accelerated Jacobi iteration
    print("Starting iterative solution...")
    x = torch.zeros_like(b) # Initial solution vector, usually starts from 0
    for i in range(num_iterations):
        # Calculate residual r = b - Ax
        residual = b - torch.sparse.mm(A, x)
        
        # Core update step: x_new = x + D_learned_inv * r
        update = learned_d_inv_flat.view(-1, 1) * residual
        x += update
        
        # Check convergence and print progress
        residual_norm = torch.linalg.norm(residual)
        if (i + 1) % 50 == 0:
            print(f"Iteration {i+1:4d}, Residual norm: {residual_norm.item():.2e}")
        if residual_norm < tolerance:
            print(f"Successfully converged after {i+1} iterations!")
            break
            
    if i == num_iterations - 1:
        print("Maximum iterations reached.")

    return x

def generate_exact_solution(N, domain_size=1.0):
    """
    Generate the exact solution of the Poisson equation p(x,y) = x²y²(1-x)²(1-y)²

    Parameters:
    N : int
        Number of grid points in each direction (including boundary points)
    domain_size : float
        Size of computational domain (default is unit square)

    Returns:
    p_exact : torch.Tensor
        Exact solution vector of size (N*N)
    p_exact_grid : torch.Tensor
        Exact solution reconstructed on 2D grid of size (N, N)
    X, Y : torch.Tensor
        x and y coordinates of grid points
    """
    # Generate grid point coordinates
    x = torch.linspace(0, domain_size, N, dtype=torch.float64) # 创建均匀分布的网格点
    y = torch.linspace(0, domain_size, N, dtype=torch.float64) # 创建均匀分布的网格点
    X, Y = torch.meshgrid(x, y, indexing='ij') # 生成二维网格

    # Calculate exact solution p = x²y²(1-x)²(1-y)²
    p_exact_grid = X ** 2 * Y ** 2 * (1 - X) ** 2 * (1 - Y) ** 2
    p_exact = p_exact_grid.flatten()

    return p_exact, p_exact_grid, X, Y

# ===================================================================
# 3. Main program entry
# ===================================================================
if __name__ == '__main__':
    # --- Configuration ---
    N_SOLVE = 256  # Define size of new problem to solve
    MODEL_PATH = 'my_poisson_jacobi_accelerator.pth'
    device = torch.device('cpu')
    
    # --- Load trained model ---
    print(f"Loading trained model from '{MODEL_PATH}'...")
    model = get_model().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print(f"Warning: Trained model '{MODEL_PATH}' not found. Will demonstrate using an untrained random model.")
        print("Solution quality may be poor, for demonstration purposes only.")

    # --- Create and solve new problem ---
    print(f"\nCreating a new problem with N={N_SOLVE}...")
    A_test = gen_2d_poisson_matrix(N_SOLVE).float()
    b_test = build_rhs_2d(N_SOLVE).view(-1, 1).float()
    
    # Call solver
    solution = solve_with_gnn_jacobi(model, A_test, b_test, device)
    
    # --- Verify results ---
    final_residual = torch.linalg.norm(b_test.to(device) - torch.sparse.mm(A_test.to(device), solution))
    print(f"\nSolution complete! Final residual norm: {final_residual.item():.2e}")