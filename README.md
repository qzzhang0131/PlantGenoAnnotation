# PlantGenoAnn: Plant Genome Annotation Model

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/qzzhang/PlantGenoANN)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 📖 Introduction
**PlantGenoAnn** is a plant genomic segmentation model designed for predicting genomic elements at single-nucleotide resolution. It is available in two variants: **PlantGenoAnn-model-plants** (trained on 9 model plants) and **PlantGenoAnn-multi-species** (trained on 42 plant species). We recommend using PlantGenoAnn-model-plants for genomic predictions.

Built upon the **[PlantBiMoE](https://github.com/HUST-Keep-Lin/PlantBiMoE)** architecture with a 1D U-Net segmentation head, it automates the prediction of gene structures—including genes, CDSs, and exons—on both forward and reverse strands. Beyond standard annotation, PlantGenoAnn serves as a **long-context plant genomic foundation model** (up to 49,152 bp), adaptable via fine-tuning to predict diverse omic signal tracks such as RNA-seq and ATAC-seq.


## 🤗 Model Access
The pre-trained weights for **PlantGenoAnn-model-plants** and **PlantGenoAnn-multi-species** are hosted on Hugging Face:
|Model Name|Access Link|Species|Training Tokens|
| :--- | :--- | :--- | :--- |
| model-plants |https://huggingface.co/qzzhang/PlantGenoAnn-model-plants| 9 | 18B |
| multi-species | https://huggingface.co/qzzhang/PlantGenoAnn-multi-species| 42 | 72B |

---

## 📁 Repository Structure
* `run_annotator.py`: Main entry point (extraction, tokenization, inference dispatch).
* `annotator.py`: Core inference script utilizing [accelerate](https://github.com/huggingface/accelerate) library (bf16 precision).
* `src/`: Functional modules for sequence processing, model wrapping, and output files generation.

## ⚙️ Installation & Environment
The model requires the [mamba-ssm](https://github.com/state-spaces/mamba) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) libraries for the core backbone.
```
# 1. Clone repository & create environment
git clone https://github.com/qzzhang0131/PlantGenoAnn.git
cd PlantGenoAnn
conda create -n plantgenoann python=3.8 -y
conda activate plantgenoann

# 2. Install dependencies (Crucial for Triton JIT & CUDA extensions)
conda install -c nvidia -c conda-forge cuda-toolkit=12.1.0 libxcrypt -y
pip install -r requirements.txt

# 3. Compile core CUDA libraries (May take 10-20 minutes)
export CUDA_HOME=$CONDA_PREFIX PATH=$CONDA_PREFIX/bin:$PATH
MAX_JOBS=4 pip install causal-conv1d==1.2.0.post2 mamba-ssm==1.2.0.post1 flash-attn==2.5.6 --no-build-isolation
```
---

## 🚀 Quick Start (Usage)

You can use PlantGenoAnn in two ways: directly using the [transformers](https://github.com/huggingface/transformers) library for model inference and obtaining embeddings, or running the complete pipeline script to generate prediction tracks or standard GFF annotation files.

### 1. Direct Model Inference
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

### 2. Full Prediction Pipeline
To run the full prediction pipeline, use the `run_annotator.py` script. The pipeline will automatically handle sliding windows, multi-GPU model inference, and standard output format.

**🛠️ 2.1 Basic Configuration:**
* `-i`: The genome FA/FNA file to be predicted.
* `-s`: The species name to be predicted.
* `-m`: Specify the path to the prediction model (downloaded weights from HuggingFace above).
* `-o`: Specify the output path.
* `-f`: Choose to write predictions to BigWig files or a standard GFF3 file (default: "bigwig").

**Save Full Prediction Tracks to BigWig Files (Recommanded):**
```bash
python run_annotator.py \
    -i ./example/Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa \
    -s Arabidopsis_lyrata \
    -m ./PlantGenoAnn-model-plants \
    -o ./example \
    -f bigwig
```
**Write Prediction Tracks to a Standard GFF3 File (Early Beta):**
```bash
python run_annotator.py \
    -i ./example/Arabidopsis_lyrata.v.1.0.dna.chromosome.8.fa \
    -s Arabidopsis_lyrata \
    -m ./PlantGenoAnn-model-plants \
    -o ./example \
    -f gff
```

**🛠️ 2.2 Advanced Pipeline Configuration (Optional):**

PlantGenoAnn prediction pipeline is highly customizable. You can adjust sliding windows, confidence thresholds, and hardware utilization to fit your specific needs:

**Hardware & Processing:**
* `--chunk_size`: The size of the chunks processed by annotator model (default: 3,200).
* `--batch_size`: The number of samples in a batch (default: 8).
* `--num_workers`: The number of CPU cores to load data in parallel (default: 8).
* `--num_tokenize_threads`: Number of CPU cores used to tokenize the sequence (default: 16).
* `--cache_path`: Specify the path to cache intermediate datasets (default: "auto").

**Sequence & Window Settings:**
* `--sliding_window_size`: Length of the sliding window used to segment the chromosome (default: 49,152).
* `--overlap_window_size`: Overlap length between consecutive windows (default: 6,144).
* `--min_chromosome_size`: Minimum chromosome size for annotating (default: 1,000,000).

**Filtering & Thresholds (only with gff output format):**
* `--threshold`: Minimum probability threshold for valid nucleotides (default: 0.50).
* `--min_gene_conf_score`: The lowest gene confidence score (default: 0.50).
* `--min_intron_conf_score`: The lowest intron confidence score (default: 0.50).
* `--min_cds_conf_score`: The lowest CDS confidence score (default: 0.50).
* `--min_gene_length`, `--min_intron_length`, `--min_cds_length`: Filter out predicted elements shorter than these values.

*For a full list of parameters, simply run `python run_annotator.py --help`.*

---

## 📜 License
See the LICENSE file for details.
## 📧 Contact
Feel free to contact qzzhang@webmail.hzau.edu.cn if you have any questions or suggestions regarding the code and models.

