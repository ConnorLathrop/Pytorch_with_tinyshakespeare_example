import os, time, json, random, urllib.request
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import CharTransformer

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def download_dataset(path="data/input.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

class CharDataset(Dataset):
    def __init__(self, data_str, seq_len, split="train", split_ratio=0.9):
        chars = sorted(set(data_str))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

        data = np.array([self.stoi[c] for c in data_str], dtype=np.int64)
        N = len(data)
        split_at = int(N * split_ratio)
        self.data = data[:split_at] if split == "train" else data[split_at:]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        start = random.randint(0, len(self.data) - self.seq_len - 1)
        chunk = self.data[start:start+self.seq_len+1]
        x = torch.from_numpy(chunk[:-1]).long()
        y = torch.from_numpy(chunk[1:]).long()
        return x, y

def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len, batch_size, layers, hidden, heads = 128, 64, 4, 256, 4
    epochs, lr, wd = 10, 3e-4, 0.01
    save_dir = "outputs"

    raw = download_dataset()
    train_ds = CharDataset(raw, seq_len, split="train")
    val_ds = CharDataset(raw, seq_len, split="val")
    val_ds.stoi, val_ds.itos, val_ds.vocab_size = train_ds.stoi, train_ds.itos, train_ds.vocab_size

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = CharTransformer(train_ds.vocab_size, seq_len, layers, hidden, heads).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    history = {"train": [], "val": []}
    best_val = float("inf")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs+1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, opt, device)
        val_loss = eval_epoch(model, val_loader, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"Epoch {epoch}: train {train_loss:.4f}, val {val_loss:.4f}, time {time.time()-t0:.1f}s")

        ckpt = {
            "model_state": model.state_dict(),
            "config": {"seq_len": seq_len, "layers": layers, "hidden": hidden, "heads": heads},
            "stoi": train_ds.stoi,
            "itos": train_ds.itos,
            "history": history,
        }
        torch.save(ckpt, os.path.join(save_dir, f"ckpt_epoch{epoch}.pth"))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(save_dir, "best.pth"))
        with open(os.path.join(save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
