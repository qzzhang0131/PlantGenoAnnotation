import os
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from .utils.file_utils import get_valid_chromosomes

class SequenceTokenizer:
    """Handles sequence tokenization using Caduceus tokenizer"""
    
    def __init__(self, config):
        self.config = config
    
    def tokenize_function(self, examples, tokenizer, max_len):
        """Tokenization function for dataset mapping"""
        tokenized_sequence = tokenizer(
            examples["sequence"],
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )["input_ids"]
        return {"input_ids": tokenized_sequence}
    
    def tokenize_chromosome(self, n: int, tsv_path: str, output_dir: str):
        """Tokenize sequences for a chunk"""
        try:
            # Load dataset from TSV
            raw_dataset = load_dataset("csv", data_files={"data": tsv_path}, split="data")
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
            
            # Determine optimal batch size and number of processes
            effective_batch_size = self.config.chunk_size
            effective_num_proc = min(len(raw_dataset), self.config.num_tokenize_proc)
            
            # Tokenize sequences
            tokenized_chunk = raw_dataset.map(
                lambda x: self.tokenize_function(x, tokenizer, self.config.sequence_length+2),
                batched=True,
                batch_size=effective_batch_size,
                num_proc=effective_num_proc,
                desc=f"Tokenizing chunk {n}"
            )
            
            # Save tokenized dataset
            chr_save_path = os.path.join(output_dir, f"chunk_{n}")
            tokenized_chunk.save_to_disk(chr_save_path)
            
        except Exception as e:
            raise RuntimeError(f"Failed to tokenize chunk {n}: {e}")
    
    def process(self, num_chunks: int):
        """Execute tokenization pipeline for all chunks"""
        for i in range(num_chunks):
            tsv_path = os.path.join(self.config.cache_path, f"chunk_{i + 1}.tsv")
            if os.path.exists(tsv_path):
                self.tokenize_chromosome(i + 1 ,tsv_path, self.config.cache_path)
            else:
                print(f"Warning: TSV file not found for chunks {i + 1}")
        
        print(f"Tokenization completed.")
        return 