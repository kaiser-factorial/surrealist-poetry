#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup, NavigableString
import os
import re

# Configuration
URLS = [
    "https://bashforever.com/?top",
    "https://bashforever.com/?top2"
]
# Path relative to this script's location
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "corpus", "bash_logs.txt")

def scrape_page(url):
    print(f"Scraping {url}...")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        # BashForever is an older site using Latin-1/ISO-8859-1
        if response.encoding == 'ISO-8859-1' or response.encoding is None:
            response.encoding = response.apparent_encoding
    except Exception as e:
        print(f"  Error fetching page: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    quotes_div = soup.find('div', class_='quotes')
    if not quotes_div:
        print("  Could not find <div class='quotes'> - attempting alternate selector...")
        # Fallback if structure is slightly different
        quotes_div = soup.find('div', class_='pageContent')
        
    if not quotes_div:
        return []

    extracted_quotes = []
    
    # Each quote is inside a <p> tag
    for p in quotes_div.find_all('p'):
        # The first div inside <p> is the header (#ID and score)
        header = p.find('div')
        if header:
            # We want everything AFTER the header div
            # A simple way is to take the text but remove the header text
            full_text = p.get_text(separator="\n").strip()
            header_text = header.get_text(separator="\n").strip()
            
            # Remove header from the start
            if full_text.startswith(header_text):
                quote_body = full_text[len(header_text):].strip()
                if quote_body:
                    extracted_quotes.append(quote_body)
        else:
            # Fallback for quotes without standard headers
            text = p.get_text(separator="\n").strip()
            if text:
                extracted_quotes.append(text)
                
    print(f"  Extracted {len(extracted_quotes)} quotes.")
    return extracted_quotes

def main():
    all_quotes = []
    for url in URLS:
        all_quotes.extend(scrape_page(url))

    # De-duplicate
    unique_quotes = list(dict.fromkeys(all_quotes))
    print(f"Total unique quotes extracted: {len(unique_quotes)}")

    # Ensure corpus directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for q in unique_quotes:
            # Use the EOS token to mark the end of the conversational fragment
            f.write(q + "<|endoftext|>\n")

    print(f"Successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
