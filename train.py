import os
import torch
import random
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from Data import *
from TrainableJacobiGNN import get_model
from loss import *

# 1. Instantiate GNN model
model = get_model()
print("GNN Model Structure:")
print(model)

# 2. Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



if __name__ == '__main__':
    # --- Training Hyperparameters ---
    NUM_MATRICES_IN_DATASET = 200  # Must match the number used when creating dataset
    TRAIN_VAL_SPLIT_RATIO = 0.8  # 80% for training, 20% for validation
    BATCH_SIZE = 10
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 50  # Set to a small number for demonstration, can be increased in practice

    # --- Load Dataset ---
    print("Loading created dataset...")
    # root='./data' points to your data folder
    dataset = MyPoissonDataset(root='./data', num_matrices=NUM_MATRICES_IN_DATASET).shuffle()
    
    # Split into training and validation sets
    train_size = int(len(dataset) * TRAIN_VAL_SPLIT_RATIO)
    val_size = len(dataset) - train_size
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    
    print(f"Dataset loaded successfully: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)  # Validation set typically doesn't need shuffling

    # --- Prepare Model and Optimizer ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Training Loop ---
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # loss_batch function in loss.py needs the entire batch object
            loss = loss_batch(model, batch)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.num_graphs
        
        avg_train_loss = total_train_loss / len(train_dataset)

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss = loss_batch(model, batch)
                total_val_loss += loss.item() * batch.num_graphs
        
        avg_val_loss = total_val_loss / len(val_dataset)

        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | Training Loss: {avg_train_loss:.6f} | Validation Loss: {avg_val_loss:.6f}")

    print("\nTraining complete!")
    
    # --- Save Trained Model ---
    model_save_path = 'my_jacobi_accelerator.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")