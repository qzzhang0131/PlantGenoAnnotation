import argparse
import os
import gc
import h5py
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_from_disk
from src.caduceus_wrapper import CaduceusModelWrapper

class GenomeAnnotator:
    """Main class for genome annotation using Caduceus model"""
    
    def __init__(self, model_path: str, cache_path: str, num_chunks: int, batch_size: int, num_workers: int):
        self.model_path = model_path
        self.cache_path = cache_path
        self.num_chunks = num_chunks
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.accelerator = Accelerator(mixed_precision="bf16")
        self.device = self.accelerator.device
        self.num_processes = self.accelerator.state.num_processes
        self.model_wrapper = CaduceusModelWrapper(self.model_path, self.device)
    
    def evaluate(self, model, test_dataloader) -> np.ndarray:
        """Perform model evaluation on the given dataloader"""
        model, dataloader = self.accelerator.prepare(model, test_dataloader)
        model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Model Inference", leave=False, disable=not self.accelerator.is_local_main_process):
                input_ids = batch["input_ids"]
                outputs = model(input_ids=input_ids)
                logits = torch.sigmoid(outputs.logits)
                
                # Gather predictions from all processes and select specific channels
                gathered_logits = self.accelerator.gather(logits)
                selected_logits = gathered_logits.cpu().numpy()[:, [0, 1, 6, 7, 8, 9], :].astype(np.float16)
                all_predictions.extend(selected_logits)
                
                # Clean up to save memory
                del input_ids, outputs, logits, gathered_logits, selected_logits
            torch.cuda.empty_cache()
            if len(all_predictions) == 0:
                return np.empty((0, 6, 0), dtype=np.float16)
        
        return np.array(all_predictions)

    def process_chromosome(self, n: int, datasets_dir: str):
        """Process a single chromosome for annotation"""
        dataset_path = os.path.join(datasets_dir, f"chunk_{n}")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Cache file for chunk {n} not found")

        datasets = load_from_disk(dataset_path)
        datasets.set_format(type="torch", columns=["input_ids"])
        dataloader = DataLoader(
            datasets,
            batch_size=min(len(datasets), self.batch_size),
            num_workers=min(len(datasets), self.num_workers),
            pin_memory=True,
            prefetch_factor=min(len(datasets), 2 * self.num_workers),
            shuffle=False,
        )
        
        # Perform model inference
        all_predictions = self.evaluate(self.model_wrapper.model, dataloader)[:len(datasets), :, :]
        
        if self.accelerator.is_main_process:
            with h5py.File(os.path.join(self.cache_path, "model_predictions.h5"), "a") as f:
                dset = f.create_dataset(
                    f"chunk_{n}",
                    data=all_predictions,
                    compression="gzip",
                    compression_opts=4,
                    chunks=True
                )
            print(f"Successfully saved predictions for chunk {n}")
        self.accelerator.wait_for_everyone()
        
        # torch.cuda.empty_cache()
        del all_predictions
        gc.collect()
        
        return

    def process(self):
        """Execute the complete genome annotation pipeline"""
        # Process each chunks
        for i in range(self.num_chunks):
            self.process_chromosome(i + 1, self.cache_path)

        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Annotator")
    parser.add_argument("--model_path", required=True, help="Specify the path to the prediction model.")
    parser.add_argument("--cache_path", required=True, help="Path to cache.")
    parser.add_argument("--num_chunks", type=int, required=True, help="Numbers of chunks.")
    parser.add_argument("--batch_size", type=int, default=8, help="The number of samples in a batch.")
    parser.add_argument("--num_workers", type=int, default=8, help="The number of CPU cores to load data in parallel.")
    args = parser.parse_args()

    annotator = GenomeAnnotator(
        model_path=args.model_path,
        cache_path=args.cache_path,
        num_chunks=args.num_chunks,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    annotator.process()