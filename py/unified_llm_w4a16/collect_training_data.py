#!/usr/bin/env python3
"""
Script to collect training data (post-attention norm embeddings and router logits)
for training an MLP predictor.

This script runs inference on text samples and collects:
- Post-attention norm embeddings (input to MoE router) per layer
- Router logits (output of router before softmax) per layer

The collected data can be used to train an MLP to predict expert selection.
"""

import torch
import numpy as np
import json
from pathlib import Path
from mixtral_8x7B_w4a16_model import Mixtral8x7BW4A16Model

# Try importing datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def collect_training_data_from_dataset(
    model,
    dataset_name="fineweb",
    num_samples=100,
    output_dir="training_data",
    min_tokens=100,
    max_tokens=512,
):
    """
    Collect training data from a streaming dataset.
    
    Args:
        model: Mixtral model instance
        dataset_name: Dataset to use ('fineweb', 'fineweb-edu', or 'orca')
        num_samples: Number of samples to collect
        output_dir: Directory to save collected data
        min_tokens: Minimum tokens per sample
        max_tokens: Maximum tokens per sample
    
    Returns:
        Dictionary with collected data statistics
    """
    if not HAS_DATASETS:
        raise ImportError("'datasets' library needed. Install with: pip install datasets")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TRAINING DATA COLLECTION FROM DATASET")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Target samples: {num_samples}")
    print(f"Token range: {min_tokens}-{max_tokens}")
    print(f"Output directory: {output_path}")
    print()
    
    # Load dataset
    print(f"Loading {dataset_name} dataset (streaming)...")
    
    if dataset_name == "fineweb":
        dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
    elif dataset_name == "fineweb-edu":
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    elif dataset_name == "orca":
        # Open-Orca dataset
        dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print("Dataset loaded!\n")
    
    # Enable training data collection
    model.model.enable_training_data_collection()
    
    # Storage for all collected data
    all_data = []
    processed_count = 0
    skipped_count = 0
    
    # Process dataset
    for example_idx, example in enumerate(dataset):
        # Extract text based on dataset format
        if dataset_name in ["fineweb", "fineweb-edu"]:
            text = example.get("text", "").strip()
        elif dataset_name == "orca":
            # Orca has system, question, and response fields
            system = example.get("system_prompt", "")
            question = example.get("question", "")
            response = example.get("response", "")
            text = f"{system}\n{question}\n{response}".strip()
        else:
            text = str(example).strip()
        
        if not text:
            skipped_count += 1
            continue
        
        # Tokenize to check length
        try:
            input_ids = model.tokenize(text)
            token_count = input_ids.shape[1]
        except Exception as e:
            print(f"  Skipping sample {example_idx}: tokenization error - {e}")
            skipped_count += 1
            continue
        
        # Filter by token count
        if token_count < min_tokens:
            skipped_count += 1
            continue
        
        # Truncate if too long
        if token_count > max_tokens:
            input_ids = input_ids[:, :max_tokens]
            token_count = max_tokens
        
        print(f"\nProcessing sample {processed_count + 1}/{num_samples} (example {example_idx})...")
        print(f"  Token count: {token_count}")
        
        # Clear previous data
        model.model.clear_training_data()
        
        # Run forward pass (this collects data internally)
        try:
            with torch.no_grad():
                _ = model(input_ids, start_pos=0)
        except Exception as e:
            print(f"  Error during forward pass: {e}")
            skipped_count += 1
            continue
        
        # Retrieve collected data
        training_data = model.model.get_training_data()
        
        if not training_data:
            print(f"  Warning: No data collected for this sample")
            skipped_count += 1
            continue
        
        print(f"  Collected {len(training_data)} layer samples")
        
        # Store each layer's data with metadata
        for layer_idx, (embeddings, router_logits) in enumerate(training_data):
            # embeddings: [batch, seq_len, hidden_size]
            # router_logits: [batch * seq_len, num_experts]
            
            sample_data = {
                'sample_idx': processed_count,
                'layer_idx': layer_idx,
                'embeddings': embeddings.cpu(),
                'router_logits': router_logits.cpu(),
                'token_count': token_count,
                'dataset': dataset_name
            }
            
            all_data.append(sample_data)
        
        processed_count += 1
        
        # Save periodically to avoid memory issues
        if processed_count % 10 == 0:
            print(f"\n  Saving checkpoint at {processed_count} samples...")
            _save_checkpoint(all_data, output_path, processed_count)
        
        if processed_count >= num_samples:
            break
    
    # Disable collection
    model.model.disable_training_data_collection()
    
    print("\n" + "=" * 60)
    print("SAVING FINAL DATA")
    print("=" * 60)
    print(f"Processed: {processed_count} samples")
    print(f"Skipped: {skipped_count} samples")
    print(f"Total layer samples: {len(all_data)}")
    
    # Save final data
    _save_final_data(all_data, output_path, dataset_name, processed_count, skipped_count, min_tokens, max_tokens)
    
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    
    return {
        'processed_samples': processed_count,
        'skipped_samples': skipped_count,
        'total_layer_samples': len(all_data),
        'dataset': dataset_name
    }


def _save_checkpoint(all_data, output_path, checkpoint_num):
    """Save a checkpoint of collected data"""
    checkpoint_file = output_path / f"checkpoint_{checkpoint_num}.pt"
    torch.save(all_data, checkpoint_file)


def _save_final_data(all_data, output_path, dataset_name, processed_count, skipped_count, min_tokens, max_tokens):
    """Save final collected data and metadata"""
    
    # Save as single PyTorch file
    data_file = output_path / "training_data.pt"
    torch.save(all_data, data_file)
    print(f"Saved training data to: {data_file}")
    
    # Also save as separate embeddings and logits files for convenience
    embeddings_list = [d['embeddings'] for d in all_data]
    logits_list = [d['router_logits'] for d in all_data]
    
    embeddings_file = output_path / "embeddings.pt"
    logits_file = output_path / "router_logits.pt"
    
    torch.save(embeddings_list, embeddings_file)
    torch.save(logits_list, logits_file)
    
    print(f"Saved embeddings to: {embeddings_file}")
    print(f"Saved router logits to: {logits_file}")
    
    # Save metadata
    metadata = {
        "dataset": dataset_name,
        "processed_samples": processed_count,
        "skipped_samples": skipped_count,
        "total_layer_samples": len(all_data),
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "embedding_shape_example": str(all_data[0]['embeddings'].shape) if all_data else "N/A",
        "logits_shape_example": str(all_data[0]['router_logits'].shape) if all_data else "N/A",
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to: {metadata_file}")


def collect_training_data_from_texts(
    model,
    texts,
    output_dir="training_data",
    max_tokens_per_sample=512
):
    """
    Collect training data from a list of text strings.
    
    Args:
        model: Mixtral model instance
        texts: List of text strings to process
        output_dir: Directory to save collected data
        max_tokens_per_sample: Maximum tokens to process per sample
    
    Returns:
        Dictionary with collected data statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TRAINING DATA COLLECTION FROM TEXT LIST")
    print("=" * 60)
    print(f"Number of samples: {len(texts)}")
    print(f"Output directory: {output_path}")
    print()
    
    # Enable training data collection
    model.model.enable_training_data_collection()
    
    all_data = []
    
    for idx, text in enumerate(texts):
        print(f"\nProcessing sample {idx + 1}/{len(texts)}...")
        
        # Tokenize
        input_ids = model.tokenize(text)
        
        # Truncate if needed
        if input_ids.shape[1] > max_tokens_per_sample:
            input_ids = input_ids[:, :max_tokens_per_sample]
            print(f"  Truncated to {max_tokens_per_sample} tokens")
        else:
            print(f"  Token count: {input_ids.shape[1]}")
        
        # Clear previous data
        model.model.clear_training_data()
        
        # Run forward pass
        with torch.no_grad():
            _ = model(input_ids, start_pos=0)
        
        # Retrieve collected data
        training_data = model.model.get_training_data()
        
        print(f"  Collected {len(training_data)} layer samples")
        
        # Store each layer's data
        for layer_idx, (embeddings, router_logits) in enumerate(training_data):
            sample_data = {
                'sample_idx': idx,
                'layer_idx': layer_idx,
                'embeddings': embeddings.cpu(),
                'router_logits': router_logits.cpu(),
                'token_count': input_ids.shape[1],
                'dataset': 'custom_texts'
            }
            all_data.append(sample_data)
    
    # Disable collection
    model.model.disable_training_data_collection()
    
    print("\n" + "=" * 60)
    print("SAVING DATA")
    print("=" * 60)
    
    _save_final_data(all_data, output_path, 'custom_texts', len(texts), 0, 0, max_tokens_per_sample)
    
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    
    return {
        'processed_samples': len(texts),
        'total_layer_samples': len(all_data),
        'dataset': 'custom_texts'
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect training data for MLP predictor")
    parser.add_argument("--model-path", type=str, default="TheBloke/mixtral-8x7b-v0.1-AWQ",
                        help="Path to model")
    parser.add_argument("--cache-size", type=int, default=2,
                        help="Expert cache size")
    parser.add_argument("--output-dir", type=str, default="training_data",
                        help="Output directory for collected data")
    parser.add_argument("--min-tokens", type=int, default=100,
                        help="Minimum tokens per sample")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens per sample")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples to collect")
    parser.add_argument("--dataset", type=str, choices=["default", "fineweb", "fineweb-edu", "orca"], 
                        default="fineweb",
                        help="Dataset to use")
    parser.add_argument("--config-path", type=str, default=None,
                        help="Path to config file")
    
    args = parser.parse_args()
    
    print("Initializing model...")
    model = Mixtral8x7BW4A16Model(
        model_path=args.model_path,
        backend="cached",
        max_cached_experts_per_layer=args.cache_size,
        device="cuda",
        config_path=args.config_path
    )
    print("Model initialized!\n")
    
    # Collect data based on dataset choice
    if args.dataset == "default":
        # Use default example texts
        example_texts = [
            "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.",
            "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet at least once.",
            "Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models that enable computers to improve their performance on a specific task through experience.",
            "Climate change is one of the most pressing issues of our time, affecting ecosystems, weather patterns, and human societies around the world.",
            "The history of computing dates back to ancient times with devices like the abacus, but modern computers emerged in the 20th century with the development of electronic circuits."
        ]
        
        # Repeat to get desired number of samples
        texts = []
        while len(texts) < args.num_samples:
            texts.extend(example_texts)
        texts = texts[:args.num_samples]
        
        metadata = collect_training_data_from_texts(
            model,
            texts,
            output_dir=args.output_dir,
            max_tokens_per_sample=args.max_tokens
        )
    else:
        # Use streaming dataset
        metadata = collect_training_data_from_dataset(
            model,
            dataset_name=args.dataset,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens
        )
    
    print("\nCollection Summary:")
    print(f"  Dataset: {metadata['dataset']}")
    print(f"  Processed samples: {metadata['processed_samples']}")
    print(f"  Total layer samples: {metadata['total_layer_samples']}")
    
    print("\nNext steps:")
    print("1. Use the collected embeddings and router_logits to train an MLP")
    print("2. The MLP should predict expert selection from embeddings")
    print("3. Target labels can be derived from router_logits (e.g., top-k experts)")
    
    return 0


if __name__ == "__main__":
    exit(main())
