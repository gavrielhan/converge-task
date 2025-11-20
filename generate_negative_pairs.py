#!/usr/bin/env python3
"""
Generate a negative (non-interacting) protein–protein dataset in FASTA format.

The negative samples are produced by randomly pairing protein IDs while
avoiding any known positive (interacting) pairs and self-pairs. The output FASTA
matches the original format: each entry has a header with two IDs and a single
sequence line containing the two protein sequences separated by a hyphen.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple


def parse_positive_fasta(path: Path) -> Tuple[Dict[str, str], Set[Tuple[str, str]]]:
    """Parse the positive FASTA file, returning sequences per ID and the set of pairs."""
    sequences: Dict[str, str] = {}
    positive_pairs: Set[Tuple[str, str]] = set()

    header: str | None = None
    seq_lines: List[str] = []

    def commit_entry(entry_header: str, entry_seq_lines: Sequence[str]) -> None:
        ids = entry_header.split()
        if len(ids) != 2:
            raise ValueError(f"Expected two IDs per header, got: '{entry_header}'")
        seq = "".join(entry_seq_lines).replace(" ", "")
        if "-" not in seq:
            raise ValueError(f"No '-' separator found for header '{entry_header}'")
        seq_a, seq_b = seq.split("-", 1)

        for pid, pseq in zip(ids, (seq_a, seq_b)):
            existing = sequences.get(pid)
            if existing and existing != pseq:
                raise ValueError(f"Conflicting sequences encountered for protein '{pid}'")
            sequences[pid] = pseq

        canonical_pair = tuple(sorted(ids))
        positive_pairs.add(canonical_pair)

    with path.open("r") as fasta_file:
        for raw_line in fasta_file:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    commit_entry(header, seq_lines)
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)

    if header is not None:
        commit_entry(header, seq_lines)

    return sequences, positive_pairs


def generate_negative_pairs(
    ids: Sequence[str],
    positive_pairs: Set[Tuple[str, str]],
    count: int,
    rng: random.Random,
) -> List[Tuple[str, str]]:
    """Randomly sample negative PPI pairs."""
    negatives: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    while len(negatives) < count:
        a = rng.choice(ids)
        b = rng.choice(ids)
        if a == b:
            continue
        canonical = tuple(sorted((a, b)))
        if canonical in positive_pairs or canonical in seen:
            continue
        seen.add(canonical)
        negatives.append((a, b))

    return negatives


def write_fasta(
    output_path: Path,
    pairs: Sequence[Tuple[str, str]],
    sequences: Dict[str, str],
) -> None:
    """Write the negative pairs to a FASTA file in the required format."""
    with output_path.open("w") as out_fasta:
        for pid_a, pid_b in pairs:
            seq_a = sequences[pid_a]
            seq_b = sequences[pid_b]
            out_fasta.write(f">{pid_a} {pid_b}\n")
            out_fasta.write(f"{seq_a}-{seq_b}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ppi_human_interactions.fasta"),
        help="Path to the positive PPI FASTA file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ppi_negative_interactions.fasta"),
        help="Destination FASTA for the generated negatives.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    args = parser.parse_args()

    sequences, positive_pairs = parse_positive_fasta(args.input)
    rng = random.Random(args.seed)
    protein_ids = list(sequences.keys())
    negatives = generate_negative_pairs(protein_ids, positive_pairs, len(positive_pairs), rng)
    write_fasta(args.output, negatives, sequences)

    print(
        f"Generated {len(negatives)} negatives covering {len(protein_ids)} proteins "
        f"from {args.input.name} -> {args.output}"
    )


if __name__ == "__main__":
    main()

