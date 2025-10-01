"""
Training script for Transformer model on Tiny Shakespeare dataset.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import requests
import os
import csv
from datetime import datetime
from model import Transformer, count_parameters


# Hyperparameters
BATCH_SIZE = 64
SEQ_LENGTH = 128
N_LAYERS = 4
N_HEADS = 4
HIDDEN_SIZE = 256
DROPOUT = 0.2
LEARNING_RATE = 3e-4
EPOCHS = 20
EVAL_INTERVAL = 500
EVAL_ITERS = 200

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CharTokenizer:
    """Character-level tokenizer."""
    
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        """Convert text to list of token indices."""
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        """Convert list of token indices to text."""
        return ''.join([self.idx_to_char[i] for i in indices])


class ShakespeareDataset(Dataset):
    """Dataset for character-level language modeling."""
    
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def download_data():
    """Download Tiny Shakespeare dataset."""
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    
    if not os.path.exists('input.txt'):
        print("Downloading Tiny Shakespeare dataset...")
        response = requests.get(url)
        with open('input.txt', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("Download complete!")
    
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    return text


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters):
    """Estimate loss on train and validation sets."""
    model.eval()
    out = {}
    
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        for i, (x, y) in enumerate(loader):
            if i >= eval_iters:
                break
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses) if losses else 0.0
    
    model.train()
    return out


def save_checkpoint(model, optimizer, tokenizer, epoch, step, val_loss, filename='checkpoint.pt'):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'tokenizer': tokenizer,
        'config': {
            'vocab_size': tokenizer.vocab_size,
            'hidden_size': HIDDEN_SIZE,
            'n_layers': N_LAYERS,
            'n_heads': N_HEADS,
            'seq_length': SEQ_LENGTH,
            'dropout': DROPOUT
        }
    }, filename)


def train():
    """Main training function."""
    
    # Setup
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Download and prepare data
    text = download_data()
    print(f"\nDataset size: {len(text):,} characters")
    
    # Create tokenizer
    tokenizer = CharTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Characters: {''.join(tokenizer.chars[:50])}...")
    
    # Encode dataset
    data = tokenizer.encode(text)
    
    # Train/validation split (90/10)
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Create datasets and dataloaders
    train_dataset = ShakespeareDataset(train_data, SEQ_LENGTH)
    val_dataset = ShakespeareDataset(val_data, SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\nTraining samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Training batches per epoch: {len(train_loader)}")
    
    # Create model
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        hidden_size=HIDDEN_SIZE,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        seq_length=SEQ_LENGTH,
        dropout=DROPOUT
    ).to(device)
    
    total_params = count_parameters(model)
    print(f"\nModel parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Prepare CSV log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f'training_log_{timestamp}.csv'
    
    with open(log_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'step', 'train_loss', 'val_loss', 'learning_rate'])
    
    # Training loop
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")
    
    print(f"{'Epoch':<8} {'Step':<8} {'Train Loss':<12} {'Val Loss':<12}")
    print("-" * 70)
    
    step = 0
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, loss = model(x, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            step += 1
            
            # Evaluate periodically
            if step % EVAL_INTERVAL == 0 or (epoch == 0 and batch_idx == 0):
                losses = estimate_loss(model, train_loader, val_loader, EVAL_ITERS)
                
                print(f"{epoch+1:<8} {step:<8} {losses['train']:<12.4f} {losses['val']:<12.4f}")
                
                # Log to CSV
                with open(log_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, step, losses['train'], losses['val'], LEARNING_RATE])
                
                # Save best model
                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    save_checkpoint(model, optimizer, tokenizer, epoch, step, 
                                  best_val_loss, 'best_model.pt')
        
        # End of epoch summary
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses = estimate_loss(model, train_loader, val_loader, EVAL_ITERS)
        
        print(f"{epoch+1:<8} (end)  {losses['train']:<12.4f} {losses['val']:<12.4f}")
        
        # Log epoch summary
        with open(log_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, 'epoch_end', losses['train'], losses['val'], LEARNING_RATE])
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, tokenizer, epoch, step, 
                          losses['val'], f'checkpoint_epoch_{epoch+1}.pt')
    
    # Final save
    save_checkpoint(model, optimizer, tokenizer, EPOCHS, step, 
                   best_val_loss, 'final_model.pt')
    
    print("\n" + "="*70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Training log saved to: {log_filename}")
    print(f"Best model saved to: best_model.pt")
    print("="*70)


if __name__ == "__main__":
    train()