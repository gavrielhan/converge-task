#!/usr/bin/env python3
"""Count FASTA entries and report distinct amino acids present."""

from pathlib import Path

FASTA_PATH = Path("ppi_human_interactions.fasta")

sequence_count = 0
distinct_amino_acids: set[str] = set()

with FASTA_PATH.open("r") as handle:
    for raw_line in handle:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            sequence_count += 1
            continue

        for char in line.replace(" ", "").upper():
            if char.isalpha():
                distinct_amino_acids.add(char)

sorted_aas = "".join(sorted(distinct_amino_acids))

print(f"Total number of sequences: {sequence_count}")
print(f"Distinct amino acids detected ({len(distinct_amino_acids)}): {sorted_aas}")

