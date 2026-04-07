import os
import logging
from typing import List, Tuple, Dict, Optional
import pyfaidx

# Set up logging
logger = logging.getLogger(__name__)

class FastaManager:
    """Manage FASTA file access using pyfaidx for efficient random access"""
    
    def __init__(self, fasta_file: str):
        """
        Initialize FASTA manager
        
        Args:
            fasta_file: Path to FASTA file
        """
        if not os.path.exists(fasta_file):
            raise FileNotFoundError(f"FASTA file not found: {fasta_file}")
        
        self.fasta_file = fasta_file
        self._faidx = None
    
    @property
    def faidx(self) -> pyfaidx.Fasta:
        """Lazy loading of FASTA index"""
        if self._faidx is None:
            try:
                self._faidx = pyfaidx.Fasta(self.fasta_file, one_based_attributes=False)
                logger.info(f"Loaded FASTA index for {self.fasta_file}")
            except Exception as e:
                raise IOError(f"Failed to load FASTA file {self.fasta_file}: {e}")
        return self._faidx
    
    def get_chromosomes(self) -> List[Tuple[str, int]]:
        """
        Get all chromosome names and lengths
        
        Returns:
            List of tuples (chromosome_name, length)
        """
        chromosomes = []
        for chrom_name in self.faidx.keys():
            chrom_length = len(self.faidx[chrom_name])
            chromosomes.append((chrom_name, chrom_length))
        
        logger.info(f"Found {len(chromosomes)} sequences in FASTA file")
        return chromosomes
    
    def get_sequence(self, chrom_name: str, start: int = 0, end: Optional[int] = None) -> str:
        """
        Get sequence for a chromosome or region
        
        Args:
            chrom_name: Chromosome name
            start: Start position (0-based)
            end: End position (0-based, exclusive)
            
        Returns:
            DNA sequence as string
        """
        if chrom_name not in self.faidx:
            raise ValueError(f"Chromosome {chrom_name} not found in FASTA file")
        
        try:
            if end is None:
                end = len(self.faidx[chrom_name])
            
            # pyfaidx uses 0-based coordinates, same as our indexing
            sequence = str(self.faidx[chrom_name][start:end])
            return sequence.upper()
            
        except Exception as e:
            raise IOError(f"Failed to get sequence for {chrom_name}[{start}:{end}]: {e}")
    
    def close(self):
        """Close the FASTA index"""
        if self._faidx is not None:
            self._faidx.close()
            self._faidx = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def get_valid_chromosomes(
    fasta_file: str, 
    min_length: int = 1000000,
    exclude_patterns: Optional[List[str]] = None,
    include_patterns: Optional[List[str]] = None
) -> List[str]:
    """
    Get valid chromosome IDs from FASTA file using pyfaidx
    
    Args:
        fasta_file: Path to input FASTA file
        min_length: Minimum chromosome length in base pairs
        exclude_patterns: Patterns to exclude from chromosome names
        include_patterns: Patterns to specifically include
        
    Returns:
        List of valid chromosome names that meet criteria
    """
    if exclude_patterns is None:
        exclude_patterns = ["random", "Un", "alt", "hap", "scaffold"]
    
    logger.info(f"Processing FASTA file: {fasta_file}")
    logger.info(f"Minimum chromosome length: {min_length:,} bp")
    logger.info(f"Exclusion patterns: {exclude_patterns}")
    if include_patterns:
        logger.info(f"Inclusion patterns: {include_patterns}")
    
    with FastaManager(fasta_file) as fasta:
        all_chromosomes = fasta.get_chromosomes()
    
    # Filter chromosomes
    valid_chromosomes = filter_chromosomes_by_length(
        all_chromosomes, min_length, exclude_patterns, include_patterns
    )
    
    logger.info(f"Selected {len(valid_chromosomes)} chromosomes after filtering")
    
    # Log detailed statistics
    if valid_chromosomes:
        valid_chromosomes_with_lengths = [
            (name, length) for name, length in all_chromosomes 
            if name in valid_chromosomes
        ]
        valid_lengths = [length for _, length in valid_chromosomes_with_lengths]
        total_length = sum(valid_lengths)
        avg_length = total_length / len(valid_lengths)
        max_length = max(valid_lengths)
        min_valid_length = min(valid_lengths)
        
        logger.info(f"Total length of selected chromosomes: {total_length:,} bp")
        logger.info(f"Average chromosome length: {avg_length:,.0f} bp")
        logger.info(f"Largest chromosome: {max_length:,} bp")
        logger.info(f"Smallest selected chromosome: {min_valid_length:,} bp")
        
        # Log the selected chromosomes with their lengths
        logger.info("Selected chromosomes:")
        for chrom_name, chrom_length in valid_chromosomes_with_lengths[:15]:
            logger.info(f"  {chrom_name}: {chrom_length:,} bp")
        
        if len(valid_chromosomes_with_lengths) > 15:
            logger.info(f"  ... and {len(valid_chromosomes_with_lengths) - 15} more chromosomes")
    
    return valid_chromosomes

def filter_chromosomes_by_length(
    chromosomes: List[Tuple[str, int]], 
    min_length: int = 1000000,
    exclude_patterns: Optional[List[str]] = None,
    include_patterns: Optional[List[str]] = None
) -> List[str]:
    """
    Filter chromosomes based on length and name patterns
    
    Args:
        chromosomes: List of (chromosome_name, length) tuples
        min_length: Minimum chromosome length in base pairs
        exclude_patterns: Patterns to exclude from chromosome names
        include_patterns: Patterns to specifically include
        
    Returns:
        List of filtered chromosome names
    """
    if exclude_patterns is None:
        exclude_patterns = ["random", "Un", "alt", "hap", "scaffold"]
    
    if include_patterns is None:
        include_patterns = []
    
    filtered_chromosomes = []
    
    for chrom_name, chrom_length in chromosomes:
        # Check length threshold
        if chrom_length < min_length:
            logger.debug(f"Excluding {chrom_name} (length: {chrom_length} < {min_length})")
            continue
        
        # Check if chromosome matches any include patterns (if specified)
        if include_patterns:
            included = any(pattern in chrom_name for pattern in include_patterns)
            if not included:
                logger.debug(f"Excluding {chrom_name} (does not match include patterns)")
                continue
        
        # Check exclusion patterns
        excluded = any(pattern in chrom_name for pattern in exclude_patterns)
        if excluded:
            logger.debug(f"Excluding {chrom_name} (matches exclusion pattern)")
            continue
        
        filtered_chromosomes.append(chrom_name)
    
    return filtered_chromosomes

def get_chromosome_regions(
    fasta_file: str,
    min_length: int = 1000000,
    exclude_patterns: Optional[List[str]] = None,
    include_patterns: Optional[List[str]] = None
) -> List[Tuple[str, int, int]]:
    """
    Get chromosome regions (start=0, end=length) for valid chromosomes
    
    Args:
        fasta_file: Path to input FASTA file
        min_length: Minimum chromosome length
        exclude_patterns: Patterns to exclude
        include_patterns: Patterns to specifically include
        
    Returns:
        List of tuples (chromosome_name, start, end)
    """
    with FastaManager(fasta_file) as fasta:
        all_chromosomes = fasta.get_chromosomes()
    
    if exclude_patterns is None:
        exclude_patterns = ["random", "Un", "alt", "hap", "scaffold"]
    
    regions = []
    for chrom_name, chrom_length in all_chromosomes:
        # Check length threshold
        if chrom_length < min_length:
            continue
        
        # Check inclusion patterns if specified
        if include_patterns:
            included = any(pattern in chrom_name for pattern in include_patterns)
            if not included:
                continue
        
        # Check exclusion patterns
        excluded = any(pattern in chrom_name for pattern in exclude_patterns)
        if not excluded:
            regions.append((chrom_name, 0, chrom_length))
    
    return regions

def get_chromosome_sequences(
    fasta_file: str,
    chromosome_ids: List[str]
) -> Dict[str, str]:
    """
    Extract full sequences for specific chromosomes from FASTA file
    
    Args:
        fasta_file: Path to FASTA file
        chromosome_ids: List of chromosome IDs to extract
        
    Returns:
        Dictionary mapping chromosome ID to sequence
    """
    chromosomes = {}
    with FastaManager(fasta_file) as fasta:
        for chrom_name in chromosome_ids:
            if chrom_name in fasta.faidx:
                sequence = fasta.get_sequence(chrom_name)
                chromosomes[chrom_name] = sequence
            else:
                logger.warning(f"Chromosome {chrom_name} not found in FASTA file")
    
    # Verify we found all requested chromosomes
    missing_chromosomes = set(chromosome_ids) - set(chromosomes.keys())
    if missing_chromosomes:
        logger.warning(f"Could not find chromosomes: {missing_chromosomes}")
    
    logger.info(f"Extracted sequences for {len(chromosomes)} chromosomes")
    return chromosomes

def get_one_chromosome(
    fasta_file: str,
    chromosome_id: str
) -> str:
    """
    Extract full sequences for specific chromosomes from FASTA file
    
    Args:
        fasta_file: Path to FASTA file
        chromosome_ids: List of chromosome IDs to extract
        
    Returns:
        Dictionary mapping chromosome ID to sequence
    """
    with FastaManager(fasta_file) as fasta:
        if chromosome_id in fasta.faidx:
            sequence = fasta.get_sequence(chromosome_id)
            return sequence
        else:
            logger.warning(f"Chromosome {chrom_name} not found in FASTA file")
            return None

def save_sequences_to_tsv(sequences: List[Tuple], output_file: str):
    """
    Save extracted sequences to TSV file with header
    
    Args:
        sequences: List of sequence tuples (chrom_id, start, end, sequence)
        output_file: Path to output TSV file
    """
    try:
        with open(output_file, "w") as f:
            f.write("sequence\n")  # Write header
            for _, _, _, sequence in sequences:
                # Ensure sequence is string and remove unwanted characters
                clean_sequence = str(sequence).replace('\n', '').replace('\t', '')
                f.write(f"{clean_sequence}\n")
    except Exception as e:
        raise IOError(f"Failed to save sequences to {output_file}: {e}")