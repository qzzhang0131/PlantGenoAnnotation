# PlantGenoAnn: Plant Genome Annotation Model

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/qzzhang/PlantGenoANN)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 📖 Introduction
**PlantGenoAnn** is a plant genomic segmentation model designed for predicting genomic elements at single-nucleotide resolution. It is available in two variants: **PlantGenoAnn-model-plants** (trained on 9 model plants) and **PlantGenoAnn-multi-species** (trained on 42 plant species). We recommend using PlantGenoAnn-model-plants for genomic predictions.

Built upon the **[PlantBiMoE](https://github.com/HUST-Keep-Lin/PlantBiMoE)** architecture with a 1D U-Net segmentation head, it automates the annotation of gene structures—including genes, CDSs, and exons—on both forward and reverse strands. Beyond standard annotation, PlantGenoAnn serves as a **long-context plant genomic foundation model** (up to 49,152 bp), adaptable via fine-tuning to predict diverse omic signal tracks such as RNA-seq and ATAC-seq.

## 🤗 Model Access
The pre-trained weights for **PlantGenoAnn-model-plants** and **PlantGenoAnn-multi-species** are hosted on Hugging Face:
* **model-plants:** [qzzhang/PlantGenoAnn-model-plants](https://huggingface.co/qzzhang/PlantGenoAnn-model-plants) (recommanded)
* **multi-species:** [qzzhang/PlantGenoAnn-multi-species](https://huggingface.co/qzzhang/PlantGenoAnn-multi-species)

---

## 📁 Repository Structure
* `run_annotator.py`: Main entry point (extraction, tokenization, inference dispatch).
* `annotator.py`: Core inference script utilizing `accelerate` library (bf16 precision).
* `src/`: Functional modules for sequence processing, model wrapping, and GFF generation.

## ⚙️ Installation & Environment
```bash
# 1. Create and activate conda environment
conda create -n plantgenoann python=3.8
conda activate plantgenoann

# 2. Clone the repository
git clone https://github.com/qzzhang0131/PlantGenoAnn.git
cd PlantGenoAnn

# 3. Install dependencies
pip install -r requirements.txt
```
The model requires the `mamba-ssm` and `causal-conv1d` libraries for the core backbone.

## 🚀 Quick Start (Usage)

You can use PlantGenoAnn in two ways: by directly interacting with the model in Python for sequence analysis, or by running the complete pipeline script to generate standard annotation files.

### 1. Direct Model Inference (Python)
You can retrieve both genomic feature probabilities and sequence embeddings using the following snippet:

```python
import torch
from transformers import AutoTokenizer, AutoModel

# Load model and tokenizer
repo_id = "qzzhang/PlantGenoAnn-model-plants"
tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# The number of DNA tokens (excluding the [CLS] and [SEP] token) needs to be divisible by 8 
# as required by the U-Net downsampling blocks. 
sequences = ["ACTAGAGCGAGAGAAA","TTTGAGAGCGCGCGGA"] 

# Tokenize
tokenized_sequences = tokenizer(
    sequences, 
    return_tensors="pt", 
    padding="longest"
)["input_ids"]

# Infer
model.to("cuda")
model.eval()
with torch.no_grad():
    outs = model(input_ids=tokenized_sequences.to("cuda"))

# Obtain the logits over the genomic features
# Shape: [batch, sequence_length, num_features]
logits = outs.logits

# Get probabilities associated with CDS on the forward strand (+)
pos_strand_cds_probs = model.get_feature_logits(feature="CDS", strand="+", logtis=logits).detach()
print(f"CDS probabilities on the forward strand: {pos_strand_cds_probs}")

# Get the sequence embeddings
# Shape: [batch, sequence_length, 1024]
hidden_states = outs.hidden_states.detach()
print(f"Sequence embeddings shape is: {hidden_states.shape}")
```

### 2. Full Annotation Pipeline (CLI)
To run the full annotation pipeline, use the `run_annotator.py` script. The pipeline will automatically handle sliding windows, multi-process model inference, and standard GFF3 assembly.

**Basic Command:**
```bash
python run_annotator.py \
    -i /examples/Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa \
    -m /PlantGenoAnn-model-plants \
    -o output_annotation.gff3
```


## 🛠️ Advanced Configuration (Optional)
PlantGenoANN is highly customizable. You can adjust sliding windows, confidence thresholds, and hardware utilization to fit your specific needs:

**Hardware & Processing:**
* `--batch_size`: The number of samples in a batch (default: 8).
* `--num_workers`: The number of CPU cores to load data in parallel (default: 8).
* `--num_tokenize_threads`: Number of CPU cores used to tokenize the sequence (default: 16).
* `--cache_path`: Specify the path to cache intermediate datasets (default: "auto").

**Sequence & Window Settings:**
* `--sliding_window_size`: Length of the sliding window used to segment the chromosome (default: 49152).
* `--overlap_window_size`: Overlap length between consecutive windows (default: 6144).
* `--min_chromosome_size`: Minimum chromosome size for annotating (default: 1000000).

**Filtering & Thresholds:**
* `--threshold`: Minimum probability threshold for valid nucleotides (default: 0.50).
* `--min_gene_conf_score`: The lowest gene confidence score (default: 0.50).
* `--min_intron_conf_score`: The lowest intron confidence score (default: 0.50).
* `--min_cds_conf_score`: The lowest CDS confidence score (default: 0.50).
* `--min_gene_length`, `--min_intron_length`, `--min_cds_length`: Filter out predicted elements shorter than these values.

*For a full list of parameters, simply run `python run_annotator.py --help`.*
## 📜 License
See the LICENSE file for details.
## 📧 Contact
Feel free to contact qzzhang@webmail.hzau.edu.cn if you have any questions or suggestions regarding the code and models.

