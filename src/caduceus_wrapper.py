import torch
import warnings
from transformers import AutoModel

class CaduceusModelWrapper:
    """Wrapper for Caduceus model with proper resource management"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained Caduceus model"""
        warnings.filterwarnings("ignore")
        try:
            self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")
    
    def predict(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Perform inference on input sequences"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Move input to correct device
        input_ids = input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits
        return logits
    
    def cleanup(self):
        """Explicitly clean up model resources"""
        if self.model is not None:
            # Move model to CPU first to ensure GPU memory is released
            self.model.to('cpu')
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            print("Model resources cleaned up")
    
    def __enter__(self):
        self._load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()