import os
import h5py
import numpy as np
from typing import List, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor

def _extract_regions(
    prob_track: np.ndarray,
    threshold: float,
    min_len: int,
    min_conf: float,
) -> List[Tuple[int, int, float]]:
    """Extract continuous regions above threshold and filter by length/confidence.

    Returns (start, end, conf) with 1-based coordinates (no start_offset applied).
    """
    if prob_track.ndim != 1:
        raise ValueError("prob_track must be 1D")

    mask = prob_track >= threshold
    if not np.any(mask):
        return []

    idx = np.where(mask)[0]
    split_points = np.where(np.diff(idx) > 1)[0] + 1
    chunks = np.split(idx, split_points)

    regions: List[Tuple[int, int, float]] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        # 1-based coordinates relative to prob_track
        start = chunk[0] + 1
        end = chunk[-1] + 1
        length = end - start + 1
        if length < min_len:
            continue
        conf = float(prob_track[chunk].mean())
        if conf < min_conf:
            continue
        regions.append((start, end, conf))
    return regions

def _find_nearby_codon(
    sequence: str,
    center: int,
    window: int,
    patterns: List[str],
    min_pos: int,
    max_pos: int,
):
    """Find the nearest specified codon (triplet) around a given center coordinate.

    Search is performed by expanding symmetrically around `center` from distance 0
    up to `window`, and stops immediately when a matching codon is found.

    All coordinates are 1-based and consistent with _extract_regions.
    Returns (codon_start, codon_end) in genomic coordinates (start <= end),
    or None if nothing is found.
    """
    seq = sequence.upper()
    n = len(seq)
    if n < 3:
        return None

    if window < 0:
        return None

    # Clamp search bounds to the sequence length
    min_pos = max(1, min_pos)
    max_pos = min(max_pos, n)
    if min_pos > max_pos:
        return None

    # Codon start must be between [min_pos, max_pos - 2]
    codon_min_start = min_pos
    codon_max_start = max_pos - 2
    if codon_min_start > codon_max_start:
        return None

    patterns_set = {p.upper() for p in patterns}

    # Expand from center outwards; at each distance d, check left then right.
    for d in range(window + 1):
        # Left side (including center when d == 0)
        p_left = center - d
        if codon_min_start <= p_left <= codon_max_start:
            codon = seq[p_left - 1 : p_left + 2]  # p_left is 1-based
            if codon in patterns_set:
                return p_left, p_left + 2

        # Right side (avoid double-checking center when d == 0)
        if d > 0:
            p_right = center + d
            if codon_min_start <= p_right <= codon_max_start:
                codon = seq[p_right - 1 : p_right + 2]
                if codon in patterns_set:
                    return p_right, p_right + 2

    return None

def _process_strand(
    strand: str,
    gene_track: np.ndarray,
    intron_track: np.ndarray,
    cds_track: np.ndarray,
    seqid: str,
    sequence: str,
    threshold: float,
    min_gene_len: int,
    min_intron_len: int,
    min_cds_len: int,
    min_gene_conf: float,
    min_intron_conf: float,
    min_cds_conf: float,
    source: str,
    start_offset: int,
) -> List[Dict[str, Any]]:
    """Process one strand (+ or -) and return gene units (for later merge and sort).

    Each gene unit is a dict:
        {
            "start": genomic_start_coordinate_with_offset,
            "lines": [list of GFF3 lines including ### separator]
        }
    """
    strand_prefix = "P" if strand == "+" else "N"

    # Local ID counters for this strand only
    id_counters = {
        ("gene", strand): 0,
        ("intron", strand): 0,
        ("CDS", strand): 0,
    }

    gene_regions = _extract_regions(
        gene_track, threshold, min_gene_len, min_gene_conf
    )
    intron_regions = _extract_regions(
        intron_track, threshold, min_intron_len, min_intron_conf
    )
    cds_regions = _extract_regions(
        cds_track, threshold, min_cds_len, min_cds_conf
    )

    gene_units: List[Dict[str, Any]] = []

    for g_start, g_end, g_conf in gene_regions:
        id_counters[("gene", strand)] += 1
        gene_id = (
            f"{seqid}-{strand_prefix}.strand.gene."
            f"{id_counters[('gene', strand)]}"
        )
        gff_start, gff_end = g_start + start_offset, g_end + start_offset

        # Introns/CDSs belonging to this gene (coordinates without start_offset)
        intron_for_gene = [
            (s, e, c)
            for (s, e, c) in intron_regions
            if g_start <= s and e <= g_end
        ]
        cds_for_gene_regions = [
            (s, e, c)
            for (s, e, c) in cds_regions
            if g_start <= s and e <= g_end
        ]

        # === start/stop codon search + CDS boundary alignment ===
        start_codon_range = None
        stop_codon_range = None

        if cds_for_gene_regions:
            cds_sorted = sorted(cds_for_gene_regions, key=lambda x: x[0])

            if strand == "+":
                # 5' most CDS
                first_cds = cds_sorted[0]
                last_cds = cds_sorted[-1]

                # Start codon: pivot at first CDS start, within [gene_start, first_cds.end]
                start_center = first_cds[0]
                start_left_bound = g_start
                start_right_bound = first_cds[1]
                # radius = min distance to the two bounds
                start_window = min(
                    start_center - start_left_bound,
                    start_right_bound - start_center,
                )
                if start_window < 0:
                    start_window = 0
                if start_left_bound <= start_right_bound:
                    start_codon_range = _find_nearby_codon(
                        sequence=sequence,
                        center=start_center,
                        window=start_window,
                        patterns=["ATG"],
                        min_pos=start_left_bound,
                        max_pos=start_right_bound,
                    )

                # Stop codon: pivot at last CDS end, within [last_cds.start, gene_end]
                stop_center = last_cds[1]
                stop_left_bound = last_cds[0]
                stop_right_bound = g_end
                stop_window = min(
                    stop_center - stop_left_bound,
                    stop_right_bound - stop_center,
                )
                if stop_window < 0:
                    stop_window = 0
                if stop_left_bound <= stop_right_bound:
                    stop_codon_range = _find_nearby_codon(
                        sequence=sequence,
                        center=stop_center,
                        window=stop_window,
                        patterns=["TAA", "TAG", "TGA"],
                        min_pos=stop_left_bound,
                        max_pos=stop_right_bound,
                    )

            else:
                # Negative strand: cds_sorted is in ascending coordinate order
                # 5' CDS = the region with the largest coordinates
                # 3' CDS = the region with the smallest coordinates
                first_cds = cds_sorted[-1]  # 5' CDS (larger coordinates)
                last_cds = cds_sorted[0]    # 3' CDS (smaller coordinates)

                # Start codon on - strand: pivot at first_cds end (5' end),
                # search within [first_cds.start, gene_end]
                start_center = first_cds[1]
                start_left_bound = first_cds[0]
                start_right_bound = g_end
                start_window = min(
                    start_center - start_left_bound,
                    start_right_bound - start_center,
                )
                if start_window < 0:
                    start_window = 0
                if start_left_bound <= start_right_bound:
                    # On the negative strand, ATG <-> CAT on the positive strand
                    start_codon_range = _find_nearby_codon(
                        sequence=sequence,
                        center=start_center,
                        window=start_window,
                        patterns=["CAT"],
                        min_pos=start_left_bound,
                        max_pos=start_right_bound,
                    )

                # Stop codon on - strand: pivot at last_cds start (3' end),
                # search within [gene_start, last_cds.end]
                stop_center = last_cds[0]
                stop_left_bound = g_start
                stop_right_bound = last_cds[1]
                stop_window = min(
                    stop_center - stop_left_bound,
                    stop_right_bound - stop_center,
                )
                if stop_window < 0:
                    stop_window = 0
                if stop_left_bound <= stop_right_bound:
                    # On the negative strand, TAA/TAG/TGA <-> TTA/CTA/TCA on the positive strand
                    stop_codon_range = _find_nearby_codon(
                        sequence=sequence,
                        center=stop_center,
                        window=stop_window,
                        patterns=["TTA", "CTA", "TCA"],
                        min_pos=stop_left_bound,
                        max_pos=stop_right_bound,
                    )

            # Use the found codons to align the 5'/3' boundaries of the first/last CDS
            cds_sorted = sorted(cds_for_gene_regions, key=lambda x: x[0])

            if start_codon_range is not None:
                sc_start, sc_end = start_codon_range
                if strand == "+":
                    # Align the start of the first CDS to the start of the start_codon
                    s, e, c = cds_sorted[0]
                    cds_sorted[0] = (sc_start, e, c)
                else:
                    # Negative strand: align the end of the 5' CDS to the end of the start_codon
                    idx = len(cds_sorted) - 1
                    s, e, c = cds_sorted[idx]
                    cds_sorted[idx] = (s, sc_end, c)

            if stop_codon_range is not None:
                tc_start, tc_end = stop_codon_range
                if strand == "+":
                    # Align the end of the last CDS to the end of the stop_codon
                    idx = len(cds_sorted) - 1
                    s, e, c = cds_sorted[idx]
                    cds_sorted[idx] = (s, tc_end, c)
                else:
                    # Negative strand: align the start of the 3' CDS to the start of the stop_codon
                    s, e, c = cds_sorted[0]
                    cds_sorted[0] = (tc_start, e, c)

            cds_for_gene_regions = cds_sorted

        # === gene line (no codon coordinates here, only the gene ID) ===
        gene_rec = (
            f"{seqid}\t{source}\tgene\t{gff_start}\t{gff_end}"
            f"\t{g_conf:.3f}\t{strand}\t.\tID={gene_id}"
        )

        # intron lines
        intron_lines: List[str] = []
        for s, e, c in intron_for_gene:
            id_counters[("intron", strand)] += 1
            intron_id = (
                f"{seqid}-{strand_prefix}.strand.intron."
                f"{id_counters[('intron', strand)]}"
            )
            s_gff, e_gff = s + start_offset, e + start_offset
            intron_lines.append(
                f"{seqid}\t{source}\tintron\t{s_gff}\t{e_gff}"
                f"\t{c:.3f}\t{strand}\t.\tID={intron_id};Parent={gene_id}"
            )

        # CDS lines (coordinates already adjusted according to codon alignment)
        cds_lines: List[str] = []
        for s, e, c in cds_for_gene_regions:
            id_counters[("CDS", strand)] += 1
            cds_id = (
                f"{seqid}-{strand_prefix}.strand.CDS."
                f"{id_counters[('CDS', strand)]}"
            )
            s_gff, e_gff = s + start_offset, e + start_offset
            cds_lines.append(
                f"{seqid}\t{source}\tCDS\t{s_gff}\t{e_gff}"
                f"\t{c:.3f}\t{strand}\t.\tID={cds_id};Parent={gene_id}"
            )

        codon_lines: List[str] = []

        if start_codon_range is not None:
            sc_start, sc_end = start_codon_range
            sc_start_gff, sc_end_gff = sc_start + start_offset, sc_end + start_offset
            codon_lines.append(
                f"{seqid}\t{source}\tstart_codon\t{sc_start_gff}\t{sc_end_gff}"
                f"\t.\t{strand}\t.\tParent={gene_id}"
            )

        if stop_codon_range is not None:
            tc_start, tc_end = stop_codon_range
            tc_start_gff, tc_end_gff = tc_start + start_offset, tc_end + start_offset
            codon_lines.append(
                f"{seqid}\t{source}\tstop_codon\t{tc_start_gff}\t{tc_end_gff}"
                f"\t.\t{strand}\t.\tParent={gene_id}"
            )

        gene_units.append(
            {
                "start": gff_start,
                "lines": [gene_rec] + intron_lines + cds_lines + codon_lines + ["###"],
            }
        )

    return gene_units


def genoann_to_gff(
    probs: np.ndarray,
    seqid: str,
    sequence: str,
    threshold: float = 0.5,
    min_gene_len: int = 100,
    min_intron_len: int = 30,
    min_cds_len: int = 30,
    min_gene_conf: float = 0.0,
    min_intron_conf: float = 0.0,
    min_cds_conf: float = 0.0,
    source: str = "PlantGenoANN",
    start_offset: int = 0,
):
    """Convert GenoANN output (6, L) matrix to a coordinate-sorted GFF3-like
    structure, grouping each gene with its introns/CDSs and separating units
    by '###'.
    Preserved original behavior:
      - Only output three feature types: gene / intron / CDS
      - Filter by threshold, minimum length, and minimum confidence
      - Return a list of GFF lines for each gene unit
    Parallelization:
      - The positive and negative strands are processed in parallel using two
        worker processes (one for '+' and one for '-'), then merged and sorted
        by genomic coordinate before writing to the output GFF3 file.
    """
    if probs.ndim != 2 or probs.shape[0] != 6:
        raise ValueError("probs must have shape (6, L)")

    L = probs.shape[1]
    if len(sequence) < L:
        raise ValueError("sequence length must be at least probs.shape[1]")

    gene_plus, gene_minus = probs[0], probs[1]
    exon_plus, exon_minus = probs[2], probs[3]
    cds_plus, cds_minus = probs[4], probs[5]

    intron_plus = gene_plus * (1.0 - exon_plus)
    intron_minus = gene_minus * (1.0 - exon_minus)

    # Prepare arguments for each strand
    strand_args = [
        (
            "+",
            gene_plus,
            intron_plus,
            cds_plus,
        ),
        (
            "-",
            gene_minus,
            intron_minus,
            cds_minus,
        ),
    ]

    gene_units: List[Dict[str, Any]] = []

    # Use two processes: one for '+' strand and one for '-' strand
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = []
        for strand, g_track, i_track, c_track in strand_args:
            futures.append(
                executor.submit(
                    _process_strand,
                    strand,
                    g_track,
                    i_track,
                    c_track,
                    seqid,
                    sequence,
                    threshold,
                    min_gene_len,
                    min_intron_len,
                    min_cds_len,
                    min_gene_conf,
                    min_intron_conf,
                    min_cds_conf,
                    source,
                    start_offset,
                )
            )

        for fut in futures:
            # Each future returns a list of gene_units for that strand
            gene_units.extend(fut.result())

    # Sort all gene units by genomic start coordinate (with offset)
    gene_units.sort(key=lambda x: x["start"])

    return gene_units