# PlantGenoANN: Plant Genome Annotation Pipeline

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/qzzhang/PlantGenoANN)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## 📖 Introduction
**PlantGenoANN** is a high-performance plant genomic segmentation model designed for predicting genomic elements at single-nucleotide resolution. Built upon the **PlantBiMoE** architecture with a 1D U-Net segmentation head, it automates the annotation of gene structures—including genes, CDSs, and exons—on both forward and reverse strands. 

Beyond standard annotation, PlantGenoANN serves as a **foundation model**, adaptable via fine-tuning to predict diverse omic signal tracks such as RNA-seq and ATAC-seq.

## 🤗 Model Access
The pre-trained weights for PlantGenoANN are hosted on Hugging Face:
* **Model Hub:** [qzzhang/PlantGenoANN](https://huggingface.co/qzzhang/PlantGenoANN)

---

## 📁 Repository Structure
* `run_annotator.py`: Main entry point (extraction, tokenization, inference dispatch).
* `annotator.py`: Core inference script utilizing `accelerate` (bf16 precision).
* `src/`: Functional modules for sequence processing, model wrapping, and GFF generation.

## ⚙️ Installation & Environment
```bash
# 1. Create and activate conda environment
conda create -n plantgenoann python=3.8
conda activate plantgenoann

# 2. Clone the repository
git clone [https://github.com/Your-Org/PlantGenoANN.git](https://github.com/Your-Org/PlantGenoANN.git)
cd PlantGenoANN

# 3. Install dependencies
pip install -r requirements.txt
```


## 🚀 Quick Start (Usage)
To run the full annotation pipeline, use the `run_annotator.py` script. The pipeline will automatically handle sliding windows, multi-process model inference, and standard GFF3 assembly.

**Basic Command:**
```bash
python run_annotator.py \
    -i path/to/your/genome.fasta \
    -m path/to/your/pretrained_model_directory \
    -o output_annotation.gff
```

## 🛠️ Advanced Configuration (Optional)
PlantGenoANN is highly customizable. You can adjust sliding windows, confidence thresholds, and hardware utilization to fit your specific needs:

**Hardware & Processing:**
* `--batch_size`: The number of samples in a batch (default: 8).
* `--num_workers`: The number of CPU cores to load data in parallel (default: 8).
* `--num_tokenize_threads`: Number of CPU cores used to tokenize the sequence (default: 24).
* `--cache_path`: Specify the path to cache intermediate datasets (default: "auto").

**Sequence & Window Settings:**
* `--sliding_window_size`: Length of the sliding window used to segment the chromosome (default: 49152).
* `--overlap_window_size`: Overlap length between consecutive windows (default: 6144).
* `--min_chromosome_size`: Minimum chromosome size for annotating (default: 1000000).

**Filtering & Thresholds:**
* `--threshold`: Minimum probability threshold for valid nucleotides (default: 0.5).
* `--min_gene_conf_score`: The lowest gene confidence score (default: 0.6).
* `--min_intron_conf_score`: The lowest intron confidence score (default: 0.70).
* `--min_cds_conf_score`: The lowest CDS confidence score (default: 0.70).
* `--min_gene_length`, `--min_intron_length`, `--min_cds_length`: Filter out predicted elements shorter than these values.

*For a full list of parameters, simply run `python run_annotator.py --help`.*
## 📜 License
See the LICENSE file for details.
