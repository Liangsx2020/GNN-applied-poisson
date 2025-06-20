import torch
from torch_geometric.loader import DataLoader
from create_my_dataset import MyPoissonDataset # Import Dataset class from your data creation script
from gnn_code.TrainableJacobiGNN import get_model
from gnn_code.loss import loss_batch
from gnn_code import MyData  # Import MyData class to resolve pickle deserialization issue

if __name__ == '__main__':
    # --- Hyperparameters ---
    DATASET_ROOT = './data'
    # Make sure this number matches the quantity used when creating the dataset
    NUM_MATRICES = 200 
    TRAIN_VAL_SPLIT_RATIO = 0.8
    BATCH_SIZE = 10
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    MODEL_SAVE_PATH = 'my_poisson_jacobi_accelerator.pth'

    # --- Data Loading ---
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    dataset = MyPoissonDataset(root=DATASET_ROOT, num_matrices=NUM_MATRICES).shuffle()
    train_size = int(len(dataset) * TRAIN_VAL_SPLIT_RATIO)
    train_dataset, val_dataset = dataset[:train_size], dataset[train_size:]
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    print(f"Dataset loaded successfully: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # --- Model and Optimizer ---
    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Training Loop ---
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_batch(model, batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.num_graphs
        avg_train_loss = total_train_loss / len(train_dataset)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                loss = loss_batch(model, batch)
                total_val_loss += loss.item() * batch.num_graphs
        avg_val_loss = total_val_loss / len(val_dataset)
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Training Loss: {avg_train_loss:.6f} | Validation Loss: {avg_val_loss:.6f}")

    print("\nTraining completed!")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")