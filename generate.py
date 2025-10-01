import argparse, torch, os, re
from model import CharTransformer

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = CharTransformer(
        vocab_size=len(ckpt["stoi"]),
        seq_len=ckpt["config"]["seq_len"],
        layers=ckpt["config"]["layers"],
        hidden=ckpt["config"]["hidden"],
        heads=ckpt["config"]["heads"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["stoi"], ckpt["itos"]

def generate_text(model, stoi, itos, prompt="ROMEO:", max_new_tokens=200, sample=True, top_k=50):
    ids = [stoi.get(ch, 0) for ch in prompt]
    x = torch.tensor([ids], dtype=torch.long)
    gen_ids = model.generate(x, max_new_tokens=max_new_tokens, sample=sample, top_k=top_k)
    return "".join([itos[i] for i in gen_ids])

def save_output(text, prompt, outdir="outputs"):
    os.makedirs(outdir, exist_ok=True)
    # Clean prompt for filename
    prefix = re.sub(r'\W+', '', prompt.lower())
    # Find next available index
    i = 1
    while os.path.exists(os.path.join(outdir, f"{prefix}_{i}.txt")):
        i += 1
    filename = os.path.join(outdir, f"{prefix}_{i}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved output to {filename}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="outputs/best.pth")
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stoi, itos = load_checkpoint(args.ckpt, device)
    text = generate_text(model, stoi, itos, prompt=args.prompt)
    print(f"\nPrompt: {args.prompt}\n{text}\n")
    save_output(text,args.prompt)

if __name__ == "__main__":
    main()
