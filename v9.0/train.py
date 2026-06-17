# Chess AI Model Trainer
import os
import datetime
import json
import ijson
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import warnings
import numpy as np

# Suppress PyTorch Conv2d padding warning
warnings.filterwarnings("ignore", message=".*Using padding='same' with even kernel lengths.*")

# Training configuration and setup
HEADLESS_MODE = True  
CHECKPOINT_PATH = "models/chess_model_checkpoint_latest.pth"
# Updated target tracking your newly generated jsonl data splits
TARGET_TRAINING_DATASET = "data/encoded/split/chess_puzzle_training_dataset_202606171141_train.json"  
LEARNING_RATE = 0.001
ESTOP_PATIENCE = 10
TRAINING_EPOCHS = 100
SCHEDULER_FACTOR = 0.1  
SCHEDULER_PATIENCE = 5  

class ParallelLineConv(nn.Module):
    """
    Scans a 6x6 spatial region on the 8x8 board but zeros out the center cells.
    Maintains parallel line scanning context adapted for piece relations.
    """
    def __init__(self, in_channels, out_channels):
        super(ParallelLineConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=6, padding='same')
        mask = torch.zeros(6, 6)
        mask[:, 0] = 1.0  
        mask[:, 2] = 1.0  
        self.register_buffer('mask', mask.unsqueeze(0).unsqueeze(0)) 

    def forward(self, x):
        with torch.no_grad():
            self.conv.weight.mul_(self.mask)
        return self.conv(x)

class SnakePathConv(nn.Module):
    """
    Scans an asymmetric 6x4 bounding box following a coordinate path matrix.
    Useful for tracking winding piece attacks or knight maneuvers.
    """
    def __init__(self, in_channels, out_channels):
        super(SnakePathConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(6, 4), padding='same')
        mask = torch.zeros(6, 4)
        path = [(0,0), (1,0), (2,0), (2,1), (2,2), (2,3), (3,3), (4,3), (5,3)]
        for r, c in path:
            mask[r, c] = 1.0
        self.register_buffer('mask', mask.unsqueeze(0).unsqueeze(0)) 

    def forward(self, x):
        with torch.no_grad():
            self.conv.weight.mul_(self.mask)
        return self.conv(x)
    
class ChessDataset(Dataset):
    def __init__(self, puzzles, solutions_from, solutions_to):
        self.puzzles = puzzles
        self.solutions_from = solutions_from
        self.solutions_to = solutions_to

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        # Unpack uint8 layers to standard working floats/longs for the batch
        puzzle = self.puzzles[idx].float()
        from_square = self.solutions_from[idx].long()
        to_square = self.solutions_to[idx].long()
         
        return puzzle, from_square, to_square

def load_and_pack_dataset(file_path):
    """
    Streams the JSON lines format directly and compresses the 12-channel bitboard
    and 1D coordinates into efficient matrices.
    """
    p_temp, f_temp, t_temp = [], [], []
    all_p, all_f, all_t = [], [], []
    
    print(f"Streaming {file_path} into memory...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            
            # Compress inputs (8x8x12)
            p_temp.append(torch.tensor(item['puzzle_tensor'], dtype=torch.uint8).unsqueeze(0))
            
            # Extract 1D cross-entropy targets (0-63) out of the sparse 2x8x8 solution matrices
            sol_arr = np.array(item['solution_tensor'], dtype=np.int8)
            from_idx = int(np.argmax(sol_arr[0]))
            to_idx = int(np.argmax(sol_arr[1]))
            
            f_temp.append(torch.tensor(from_idx, dtype=torch.uint8).unsqueeze(0))
            t_temp.append(torch.tensor(to_idx, dtype=torch.uint8).unsqueeze(0))
            
            if (i + 1) % 10000 == 0:
                all_p.append(torch.cat(p_temp, dim=0))
                all_f.append(torch.cat(f_temp, dim=0))
                all_t.append(torch.cat(t_temp, dim=0))
                p_temp, f_temp, t_temp = [], [], []
                
        if p_temp:
            all_p.append(torch.cat(p_temp, dim=0))
            all_f.append(torch.cat(f_temp, dim=0))
            all_t.append(torch.cat(t_temp, dim=0))
            
    return torch.cat(all_p, dim=0), torch.cat(all_f, dim=0), torch.cat(all_t, dim=0)

class V7P3RChessCNN(nn.Module):
    """
    Refactored version of your multi-lens architecture optimized for an 8x8 board.
    Implements a dual classification output head for piece origins and targets.
    """
    def __init__(self):
        super(V7P3RChessCNN, self).__init__()
        
        # --- LAYER 1: MULTI-SCALE GEOMETRIC CHESS LENSES ---
        # Input: 12 channels (Bitboard piece configurations)
        # Output: 16 filters * 8 lenses = 128 channels
        self.conv2x2 = nn.Conv2d(12, 16, kernel_size=2, padding='same')
        self.conv3x3 = nn.Conv2d(12, 16, kernel_size=3, padding='same')
        self.conv4x4 = nn.Conv2d(12, 16, kernel_size=4, padding='same')
        self.conv2x6 = nn.Conv2d(12, 16, kernel_size=(2, 6), padding='same')
        self.conv1x5 = nn.Conv2d(12, 16, kernel_size=(1, 5), padding='same')
        self.conv5x1 = nn.Conv2d(12, 16, kernel_size=(5, 1), padding='same')
        self.parallel_lens = ParallelLineConv(12, 16)  
        self.snake_lens = SnakePathConv(12, 16)  

        self.dropout = nn.Dropout2d(p=0.1)
        
        # --- LAYER 2: THE SYNTHESIZER ---
        self.layer2 = nn.Conv2d(128, 64, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        
        # --- LAYER 3: DUAL POLICY HEADS ---
        # Splitting predictions into departure logits (64) and arrival logits (64)
        self.from_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )
        
        self.to_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

    def forward(self, x):
        out2x2 = F.relu(self.conv2x2(x))
        out3x3 = F.relu(self.conv3x3(x))
        out4x4 = F.relu(self.conv4x4(x))
        out2x6 = F.relu(self.conv2x6(x))
        out1x5 = F.relu(self.conv1x5(x))
        out5x1 = F.relu(self.conv5x1(x))
        out_parallel = F.relu(self.parallel_lens(x))  
        out_snake = F.relu(self.snake_lens(x))  
        
        x1 = torch.cat([
            out2x2, out3x3, out4x4, out2x6, out1x5, out5x1, 
            out_parallel, out_snake        
        ], dim=1)
        
        x1_regularized = self.dropout(x1)
        x2 = F.relu(self.bn2(self.layer2(x1_regularized)))
        
        # Branch decisions across separate heads
        logits_from = self.from_head(x2)
        logits_to = self.to_head(x2)
        
        return logits_from, logits_to

def setup_logging(target_training_dataset):
    log_dir = "analysis/logs"
    os.makedirs(log_dir, exist_ok=True)
    dataset_name = os.path.basename(target_training_dataset).replace("_train.json", "")
    return os.path.join(log_dir, f"results_{dataset_name}_train.log")

def log_message(log_file, message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + "\n")

def main():
    log_file = setup_logging(TARGET_TRAINING_DATASET)
    os.makedirs("models", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(log_file, f"Running chess engine training on: {device}")

    # Load and pack inputs
    train_p, train_f, train_t = load_and_pack_dataset(TARGET_TRAINING_DATASET)
    log_message(log_file, f"Loaded training dataset with {len(train_p)} positions.")
    
    val_p, val_f, val_t = load_and_pack_dataset(TARGET_TRAINING_DATASET.replace("_train.json", "_val.json"))
    log_message(log_file, f"Loaded validation dataset with {len(val_p)} positions.")

    train_dataset = ChessDataset(train_p, train_f, train_t)
    val_dataset = ChessDataset(val_p, val_f, val_t)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = V7P3RChessCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    start_epoch = 0
    epochs_without_improvement = 0

    if os.path.exists(CHECKPOINT_PATH):
        log_message(log_file, f"Restoring checkpoint session from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']

    for epoch in range(start_epoch, TRAINING_EPOCHS):
        model.train()
        running_train_loss = 0.0
        correct_moves = 0
        total_samples = 0

        for puzzles, from_squares, to_squares in train_loader:
            puzzles = puzzles.to(device)
            from_squares = from_squares.to(device)
            to_squares = to_squares.to(device)
            
            optimizer.zero_grad()
            pred_from, pred_to = model(puzzles)
            
            # Combined Loss function tracking both heads
            loss = criterion(pred_from, from_squares) + criterion(pred_to, to_squares)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * puzzles.size(0)
            
            # Accurate metrics verification loop
            p_from_class = torch.argmax(pred_from, dim=1)
            p_to_class = torch.argmax(pred_to, dim=1)
            correct_moves += ((p_from_class == from_squares) & (p_to_class == to_squares)).sum().item()
            total_samples += puzzles.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_acc = (correct_moves / total_samples) * 100

        # Evaluation Track
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for v_puzzles, v_from, v_to in val_loader:
                v_puzzles, v_from, v_to = v_puzzles.to(device), v_from.to(device), v_to.to(device)
                vp_from, vp_to = model(v_puzzles)
                v_loss = criterion(vp_from, v_from) + criterion(vp_to, v_to)
                running_val_loss += v_loss.item() * v_puzzles.size(0)
                
                vp_from_c = torch.argmax(vp_from, dim=1)
                vp_to_c = torch.argmax(vp_to, dim=1)
                val_correct += ((vp_from_c == v_from) & (vp_to_c == v_to)).sum().item()
                val_total += v_puzzles.size(0)
            
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_acc = (val_correct / val_total) * 100
        scheduler.step(epoch_val_loss)

        epoch_log = f"Epoch {epoch+1}/{TRAINING_EPOCHS} - Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f} | Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%"
        log_message(log_file, epoch_log)

        # Checkpoint serialization
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        torch.save(checkpoint, CHECKPOINT_PATH)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "models/chess_model_best.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= ESTOP_PATIENCE:
                log_message(log_file, f"Early stopping triggered at epoch {epoch+1}.")
                break

    # Save validation reporting graphics
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', color='blue')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss', color='orange')
    plt.title('V7P3R Chess Policy Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    os.makedirs("analysis/training_runs", exist_ok=True)
    plt.savefig(f"analysis/training_runs/chess_training_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png")
    log_message(log_file, "Training completed successfully!")

if __name__ == "__main__":
    main()