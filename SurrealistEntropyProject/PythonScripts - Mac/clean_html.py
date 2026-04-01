#!/usr/bin/env python3
from bs4 import BeautifulSoup
import os

def clean_html(input_path, output_path):
    """Convert an HTML file to clean plain text and save as a .txt file."""
    print(f"Reading {input_path}...")

    with open(input_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Extract plain text
    text = soup.get_text()

    # Remove excessive blank lines and strip whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    size_kb = os.path.getsize(output_path) // 1024
    print(f"Saved to {output_path} ({size_kb} KB)")

# --- Euclid's Elements ---
clean_html(
    input_path="../../Euclid_Elements/Euclid_Elements.html",
    output_path="euclid_elements.txt"
)

# --- Baudelaire (merge both works into one file) ---
def clean_and_merge_html(input_paths, output_path):
    """Convert multiple HTML files and merge into a single .txt file."""
    combined_lines = []
    for input_path in input_paths:
        print(f"Reading {input_path}...")
        with open(input_path, encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        text = soup.get_text()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        combined_lines.extend(lines)

    clean_text = "\n".join(combined_lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    size_kb = os.path.getsize(output_path) // 1024
    print(f"Saved to {output_path} ({size_kb} KB)")

clean_and_merge_html(
    input_paths=[
        "../../Baudelaire/Les Fleurs du Mal_Baudelaire/Les Fleurs du Mal_Baudelaire.html",
        "../../Baudelaire/Poems in Prose_Baudelaire/Poems in Prose_Baudelaire.html"
    ],
    output_path="baudelaire.txt"
)

# --- Rimbaud ---
clean_html(
    input_path="../../poetry_sources/Rimbaud, Arthur (1854–1891) - Une Saison en Enfer.html",
    output_path="rimbaud.txt"
)

print("\nAll files converted successfully!")
