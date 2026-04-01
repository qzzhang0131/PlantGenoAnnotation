# PlantGenoANN: Plant Genome Annotation Pipeline

## 📖 Introduction
PlantGenoANN is a plant genomic segmentation model that enables the prediction of various plant genomic elements at single-nucleotide resolution. The model is built upon the PlantBiMoE architecture with a 1D U-Net segmentation head, specifically designed for automated plant genome annotation. It predicts gene structures—including genes, CDSs, and exons—on both the forward and reverse strands. In addition, the model can serve as a foundation model, adaptable through fine-tuning to predict omic signal tracks such as plant RNA-seq and ATAC-seq.

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
```


## 🚀 Quick Start (Usage)
To run the full annotation pipeline, use the `run_annotator.py` script. The pipeline will automatically handle sliding windows, multi-process model inference, and standard GFF3 assembly.

**Basic Command:**
```bash
python run_annotator.py \
    -i path/to/your/genome.fasta \
    -m path/to/your/finetuned_model_directory \
    -o output_annotation.gff
```

## 🛠️ Advanced Configuration (Optional)
PlantGenoANN is customizable. You can adjust sliding windows, confidence thresholds, and hardware utilization to fit your specific needs:

**Hardware & Processing:**
* `--batch_size`: The number of samples in a batch (default: 8).
* `--num_workers`: The number of CPU cores to load data in parallel (default: 8).
* `--num_tokenize_threads`: Number of CPU cores used to tokenize the sequence (default: 24).
* `--cache_path`: Specify the path to cache intermediate datasets (default: "auto").

**Sequence & Window Settings:**
* `--sliding_window_size`: Length of the sliding window used to segment the chromosome (default: 32768).

*For a full list of parameters, simply run `python run_annotator.py --help`.*

## 📜 License
See the LICENSE file for details.
