"""
Text generation script using trained Transformer model.
"""

import torch
import argparse
import os
from datetime import datetime
from model import Transformer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path='best_model.pt'):
    """Load trained model from checkpoint."""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract configuration and tokenizer
    config = checkpoint['config']
    tokenizer = checkpoint['tokenizer']
    
    # Create model
    model = Transformer(
        vocab_size=config['vocab_size'],
        hidden_size=config['hidden_size'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        seq_length=config['seq_length'],
        dropout=0.0  # No dropout during inference
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {config['vocab_size']}")
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_tokens=200, temperature=0.8, top_k=40):
    """
    Generate text from a prompt.
    
    Args:
        model: Trained Transformer model
        tokenizer: Character tokenizer
        prompt: Starting text prompt
        max_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Sample from top k tokens only
    
    Returns:
        Generated text
    """
    model.eval()
    
    # Encode prompt
    context = tokenizer.encode(prompt)
    idx = torch.tensor([context], dtype=torch.long, device=device)
    
    # Generate tokens
    with torch.no_grad():
        generated_idx = model.generate(
            idx, 
            max_new_tokens=max_tokens, 
            temperature=temperature, 
            top_k=top_k
        )
    
    # Decode to text
    generated_text = tokenizer.decode(generated_idx[0].tolist())
    
    return generated_text


def save_generated_text(text, filename, prompt, temperature, top_k):
    """Save generated text with metadata."""
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Prompt: {prompt}\n")
        f.write(f"Temperature: {temperature}\n")
        f.write(f"Top-k: {top_k}\n")
        f.write(f"\n{'='*70}\n\n")
        f.write(text)
        f.write(f"\n\n{'='*70}\n")
    
    print(f"Saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Generate text using trained Transformer model')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='ROMEO:',
                       help='Text prompt to start generation')
    parser.add_argument('--tokens', type=int, default=200,
                       help='Number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Sample from top k tokens')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename (default: auto-generated)')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode for multiple generations')
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.checkpoint)
    
    print(f"\nUsing device: {device}")
    print("="*70)
    
    if args.interactive:
        # Interactive mode
        print("\nInteractive Generation Mode")
        print("Enter prompts (or 'quit' to exit)")
        print("="*70)
        
        while True:
            prompt = input("\nPrompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            if not prompt:
                continue
            
            print(f"\nGenerating with prompt: '{prompt}'")
            print("-"*70)
            
            text = generate_text(
                model, tokenizer, prompt, 
                max_tokens=args.tokens,
                temperature=args.temperature,
                top_k=args.top_k
            )
            
            print(text)
            print("-"*70)
            
            save_choice = input("\nSave this output? (y/n): ").strip().lower()
            if save_choice == 'y':
                filename = f"generated_{prompt.replace(':', '').replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                save_generated_text(text, filename, prompt, args.temperature, args.top_k)
    
    else:
        # Single generation mode
        print(f"\nGenerating text with prompt: '{args.prompt}'")
        print(f"Tokens: {args.tokens}, Temperature: {args.temperature}, Top-k: {args.top_k}")
        print("="*70)
        
        text = generate_text(
            model, tokenizer, args.prompt,
            max_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        print(f"\n{text}\n")
        print("="*70)
        
        # Save output
        if args.output:
            filename = args.output
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_prompt = args.prompt.replace(':', '').replace(' ', '_').lower()[:20]
            filename = f"generated_{safe_prompt}_{timestamp}.txt"
        
        save_generated_text(text, filename, args.prompt, args.temperature, args.top_k)


def generate_default_samples():
    """Generate default samples (ROMEO and JULIET) for deliverables."""
    
    print("\n" + "="*70)
    print("Generating default samples for deliverables...")
    print("="*70)
    
    # Load model
    model, tokenizer = load_model('best_model.pt')
    
    prompts = ["ROMEO:", "JULIET:"]
    
    for prompt in prompts:
        print(f"\n{'='*70}")
        print(f"Generating with prompt: {prompt}")
        print('='*70)
        
        text = generate_text(
            model, tokenizer, prompt,
            max_tokens=200,
            temperature=0.8,
            top_k=40
        )
        
        print(text)
        
        # Save to file
        filename = f"sample_{prompt.replace(':', '').lower()}.txt"
        save_generated_text(text, filename, prompt, 0.8, 40)
    
    print("\n" + "="*70)
    print("Sample generation complete!")
    print("="*70)


if __name__ == "__main__":
    import sys
    
    # If run with --samples flag, generate default samples
    if len(sys.argv) > 1 and sys.argv[1] == '--samples':
        generate_default_samples()
    else:
        main()