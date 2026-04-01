#!/usr/bin/env python3
import requests
import os
import time
from transformers import AutoTokenizer

# --- Articles to fetch ---

TOPOLOGY_ARTICLES = [
    "Topological space",
    "Topology",
    "Homeomorphism",
    "Compactness",
    "Continuous function",
    "Open set",
    "Closed set",
    "Metric space",
    "Neighbourhood (mathematics)",
    "Connectedness",
    "Boundary (topology)",
    "Hausdorff space",
]

SET_THEORY_ARTICLES = [
    "Set theory",
    "Zermelo–Fraenkel set theory",
    "Set (mathematics)",
    "Subset",
    "Power set",
    "Union (set theory)",
    "Intersection (set theory)",
    "Complement (set theory)",
    "Cardinality",
    "Countable set",
    "Infinite set",
    "Axiom of choice",
]

# --- Fetch from Wikipedia API ---

def fetch_article(title):
    """Fetch plain text of a Wikipedia article via the API."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
        "redirects": 1,
    }
    headers = {
        "User-Agent": "SurrealistEntropyProject/1.0 (academic research; python requests)"
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        pages = data["query"]["pages"]
        page = next(iter(pages.values()))
        if "extract" in page:
            return page["extract"]
        else:
            print(f"  Warning: no content found for '{title}'")
            return ""
    except Exception as e:
        print(f"  Error fetching '{title}': {e}")
        return ""

def clean_text(text):
    """Basic cleanup of Wikipedia plain text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Remove lines that are just section headers with no content (== Header ==)
    lines = [l for l in lines if not (l.startswith("==") and l.endswith("=="))]
    return "\n".join(lines)

def scrape_and_save(articles, output_path, label):
    """Fetch a list of articles, combine, clean, and save to a file."""
    print(f"\nFetching {label} articles...")
    combined = []
    for title in articles:
        print(f"  Fetching: {title}")
        text = fetch_article(title)
        if text:
            combined.append(f"--- {title} ---\n{clean_text(text)}")
        time.sleep(0.5)  # Be polite to Wikipedia's servers

    full_text = "\n\n".join(combined)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"  Saved to {output_path} ({os.path.getsize(output_path) // 1024} KB)")
    return full_text

# --- Run ---

topology_text = scrape_and_save(
    TOPOLOGY_ARTICLES,
    "topology.txt",
    "Topology"
)

set_theory_text = scrape_and_save(
    SET_THEORY_ARTICLES,
    "set_theory.txt",
    "Set Theory"
)

# --- Token counts ---

print("\nChecking token counts...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

for name, text in [("Topology", topology_text), ("Set Theory", set_theory_text)]:
    tokens = tokenizer(text, truncation=False)
    print(f"  {name}: {len(tokens['input_ids']):,} tokens")

print("\nDone! Files saved: topology.txt, set_theory.txt")
