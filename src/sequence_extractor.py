import os
from typing import List, Tuple, Dict
from .utils.file_utils import get_valid_chromosomes, get_chromosome_sequences, save_sequences_to_tsv

class SequenceExtractor:
    """Handles genome sequence extraction and overlapping segmentation using pyfaidx"""
    
    def __init__(self, config):
        self.config = config
    
    def read_chromosomes(self, fasta_file: str, valid_ids: List[str]) -> Dict[str, str]:
        """Read sequences for specified chromosomes from FASTA file using pyfaidx"""
        return get_chromosome_sequences(fasta_file, valid_ids)
    
    def _slice_single_chromosome(self, chrom_id: str, chrom_seq: str) -> List[Tuple]:
        """Extract overlapping sliding windows from a single chromosome"""
        valid_sequences = []
        seq_length = self.config.sequence_length
        overlap_offset = self.config.overlap_offset
        
        # Slide through chromosome with overlap
        start_pos = 0
        while start_pos + seq_length <= len(chrom_seq):
            end_pos = start_pos + seq_length
            sequence = chrom_seq[start_pos:end_pos]
            valid_sequences.append((chrom_id, start_pos, end_pos, sequence))
            start_pos += (seq_length - overlap_offset)
        
        # Ensure the last segment is included
        if len(chrom_seq) > seq_length:
            last_start = len(chrom_seq) - seq_length
            last_sequence = chrom_seq[last_start:len(chrom_seq)]
            valid_sequences.append((chrom_id, last_start, len(chrom_seq), last_sequence))
        
        return valid_sequences
    
    def overlap_slice(self, chromosomes: Dict[str, str], save_dir: str):
        """Perform overlapping slice on all chromosomes and save results"""
        all_sequences = []
        chrom_sequence_info = {}
        
        for chrom_id, chrom_seq in chromosomes.items():
            chrom_length = len(chrom_seq)
            
            if chrom_length < self.config.sequence_length:
                print(f"Warning: Chromosome {chrom_id} is too short ({chrom_length} bp), skipping")
                continue
                
            valid_sequences = self._slice_single_chromosome(chrom_id, chrom_seq)
            chrom_sequence_info[chrom_id] = (chrom_length ,len(valid_sequences))
            all_sequences.extend(valid_sequences)
        
        num_chunks = len(all_sequences) // self.config.chunk_size
        for i in range(num_chunks):
            index = i * self.config.chunk_size
            output_file = os.path.join(save_dir, f"chunk_{i + 1}.tsv")
            save_sequences_to_tsv(all_sequences[index : index + self.config.chunk_size], output_file)
            
        output_file = os.path.join(save_dir, f"chunk_{num_chunks + 1}.tsv")
        save_sequences_to_tsv(all_sequences[num_chunks * self.config.chunk_size: ], output_file)
        
        return chrom_sequence_info, num_chunks + 1
    
    def process(self):
        """Execute the complete sequence extraction pipeline"""
        # Get valid chromosome IDs from FASTA file
        valid_ids = get_valid_chromosomes(
            self.config.input_fasta,
            self.config.min_chrom_length,
            self.config.exclude_patterns,
            getattr(self.config, 'include_patterns', None)
        )
        
        if not valid_ids:
            raise ValueError("No valid chromosomes found meeting the criteria")
        print(f"Found {len(valid_ids)} valid chromosomes")
        
        chromosomes = self.read_chromosomes(self.config.input_fasta, valid_ids)
        chrom_sequence_info, num_chunks = self.overlap_slice(chromosomes, self.config.cache_path)
        print(f"Successfully divide all sliding windows into {num_chunks} chunks")
        print(f"Sliding windows extraction completed.")
        
        return chrom_sequence_info, num_chunks