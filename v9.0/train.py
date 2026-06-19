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
TARGET_TRAINING_DATASET = "data/encoded/split/chess_puzzle_training_dataset_202606171718_train.json"  
LEARNING_RATE = 0.001
ESTOP_PATIENCE = 10
TRAINING_EPOCHS = 100
SCHEDULER_FACTOR = 0.1  
SCHEDULER_PATIENCE = 5  

class BoardForceAttention(nn.Module):
    """
    Self-Attention engine that maps long-range piece dependencies.
    Treats the 64 squares as interconnected force points, calculating
    pairwise tactical relationships regardless of physical board distance.
    """
    def __init__(self, in_channels, embedding_dim=64):
        super(BoardForceAttention, self).__init__()
        self.embedding_dim = embedding_dim
        # Compress spatial channels into an interaction embedding space
        self.proj = nn.Conv2d(in_channels, embedding_dim, kernel_size=1)
        
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key   = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        
        self.out_conv = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(embedding_dim)

    def forward(self, x):
        batch, c, h, w = x.size()
        
        # 1. Project channels and flatten to 64 spatial tokens (8x8 grid)
        feat = self.proj(x).view(batch, self.embedding_dim, h * w).permute(0, 2, 1) # (B, 64, Dim)
        
        # 2. Linear projection for Attention matrices
        Q = self.query(feat)
        K = self.key(feat)
        V = self.value(feat)
        
        # 3. Calculate force-field interaction scores (64 x 64 matrix)
        scores = torch.bmm(Q, K.permute(0, 2, 1)) / (self.embedding_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        # 4. Contextualize values and reshape back to standard 8x8 grid
        context = torch.bmm(attn_weights, V) # (B, 64, Dim)
        context = context.permute(0, 2, 1).view(batch, self.embedding_dim, h, w)
        
        return F.relu(self.bn(self.out_conv(context)))
    
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
    Refactored Chess Engine Policy Network.
    Combines your multi-scale local spatial lenses with a long-range
    global self-attention force-field layer for high-fidelity move prediction.
    """
    def __init__(self):
        super(V7P3RChessCNN, self).__init__()
        
        # --- PHASE 1: MICRO-SPATIAL REGIONAL LENSES ---
        # Input: 12 standard bitboard channels
        # Retaining your structural layout for pawns, blocks, and local clusters
        self.conv2x2 = nn.Conv2d(12, 16, kernel_size=2, padding='same')
        self.conv3x3 = nn.Conv2d(12, 16, kernel_size=3, padding='same')
        self.conv4x4 = nn.Conv2d(12, 16, kernel_size=4, padding='same')
        
        # --- PHASE 2: LONG-RANGE FORCE-FIELD OBSERVATION ---
        # Captures active lines of sight, pins, and King defenses globally
        self.global_force_lens = BoardForceAttention(in_channels=12, embedding_dim=48)
        
        # Combine channels: 16*3 (local) + 48 (global) = 96 feature layers
        self.dropout = nn.Dropout2d(p=0.3)
        
        # --- PHASE 3: THE SYNTHESIZER ---
        self.layer2 = nn.Conv2d(96, 64, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        
        # --- PHASE 4: DUAL POLICY DENSE HEADS ---
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
        # 1. Run local geometric lenses
        out2x2 = F.relu(self.conv2x2(x))
        out3x3 = F.relu(self.conv3x3(x))
        out4x4 = F.relu(self.conv4x4(x))
        
        # 2. Run long-range global relationship lens
        out_forces = self.global_force_lens(x)
        
        # 3. Concatenate local shapes and global attention views seamlessly
        x1 = torch.cat([out2x2, out3x3, out4x4, out_forces], dim=1)
        x1_regularized = self.dropout(x1)
        
        # 4. Synthesize down to core features
        x2 = F.relu(self.bn2(self.layer2(x1_regularized)))
        
        # 5. Output raw move action scores
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