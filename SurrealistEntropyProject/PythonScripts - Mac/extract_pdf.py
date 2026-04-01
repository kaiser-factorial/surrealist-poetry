#!/usr/bin/env python3
# Extracts text from PDF files and saves as clean .txt
# Usage: run once to convert PDFs in poetry_sources/ to .txt files

from pdfminer.high_level import extract_text
import os

def extract_pdf(input_path, output_path):
    """Extract plain text from a PDF and save as .txt"""
    print(f"Reading {input_path}...")
    try:
        text = extract_text(input_path)

        # Clean up excessive whitespace and blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(clean_text)

        size_kb = os.path.getsize(output_path) // 1024
        print(f"Saved to {output_path} ({size_kb} KB)")

    except Exception as e:
        print(f"Error extracting {input_path}: {e}")

# --- Breton Manifesto ---
extract_pdf(
    input_path="../../poetry_sources/MANIFESTO OF SURREALISM.pdf",
    output_path="breton_manifesto.txt"
)

print("\nDone!")
