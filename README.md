# PlantGenoANN: Plant Genome Annotation Pipeline

## 📖 Introduction
PlantGenoANN is an automated, deep-learning-based pipeline for plant genome annotation. Fine-tuned from our foundation model PlantBiMoE (released in Dec 2025), this tool leverages a powerful language model (via Caduceus architecture) to directly process raw genome sequences and accurately predict gene structures, including CDS and introns.

With PlantGenoANN, you can perform end-to-end genome annotation with a single command: going directly from a raw FASTA/FNA sequence file to a standard GFF3 annotation file.

## 📁 Repository Structure
* `run_annotator.py`: The main entry point for the pipeline. It handles sequence extraction, tokenization, model inference dispatch, and GFF generation.
* `annotator.py`: The core model inference script, utilizing `accelerate` for efficient, multi-GPU processing (bf16 precision).
* `src/`: Core functional modules:
  * `sequence_extractor.py` & `sequence_tokenizer.py`: Handles sliding-window sequence chunking and tokenization.
  * `caduceus_wrapper.py`: Wraps the loaded model for distributed inference.
  * `gff_writer.py`: Formats model predictions into standard GFF3 output.
  * `configuration.py`: Manages pipeline hyperparameters.

## ⚙️ Installation & Environment
We recommend creating a fresh Conda environment for PlantGenoANN (Python 3.8+).

```bash
# 1. Create and activate conda environment
conda create -n plantgenoann python=3.8
conda activate plantgenoann

# 2. Clone the repository
git clone [https://github.com/Your-Org/PlantGenoANN.git](https://github.com/Your-Org/PlantGenoANN.git)
cd PlantGenoANN

# 3. Install dependencies
pip install -r requirements.txt

## 🚀 Quick Start (Usage)
To run the full annotation pipeline, use the run_annotator.py script. The pipeline will automatically handle sliding windows, multi-process model inference, and standard GFF3 assembly.

Basic Command:

Bash
python run_annotator.py \
    -i path/to/your/genome.fasta \
    -m path/to/your/finetuned_model_directory \
    -o output_annotation.gff

🛠️ Advanced Configuration (Optional)
PlantGenoANN is highly customizable. You can adjust sliding windows, confidence thresholds, and hardware utilization to fit your specific needs:

Hardware & Processing:

--batch_size: The number of samples in a batch (default: 8).

--num_workers: The number of CPU cores to load data in parallel (default: 8).

--num_tokenize_threads: Number of CPU cores used to tokenize the sequence (default: 24).

--cache_path: Specify the path to cache intermediate datasets (default: "auto").

Sequence & Window Settings:

--sliding_window_size: Length of the sliding window used to segment the chromosome (default: 49152).

--overlap_window_size: Overlap length between consecutive windows (default: 6144).

--min_chromosome_size: Minimum chromosome size for annotating (default: 1000000).

Filtering & Thresholds:

--threshold: Minimum probability threshold for valid nucleotides (default: 0.5).

--min_gene_conf_score: The lowest gene confidence score (default: 0.6).

--min_intron_conf_score: The lowest intron confidence score (default: 0.70).

--min_cds_conf_score: The lowest CDS confidence score (default: 0.70).

--min_gene_length, --min_intron_length, --min_cds_length: Filter out predicted elements shorter than these values.

For a full list of parameters, simply run python run_annotator.py --help.

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
