import os

# Paths
CORPUS_FILE = os.path.join(os.path.dirname(__file__), "..", "shared", "corpus", "bash_logs.txt")
EXPORT_FILE = os.path.join(os.path.dirname(__file__), "claude_upload_slice.txt")

# Configuration
SLICE_SIZE = 50  # How many quotes to give Claude at a time for best quality

def prepare_slice():
    if not os.path.exists(CORPUS_FILE):
        print(f"Error: {CORPUS_FILE} not found.")
        return

    with open(CORPUS_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by the EOS token
    quotes = content.split("<|endoftext|>")
    quotes = [q.strip() for q in quotes if q.strip()]

    print(f"Total quotes available: {len(quotes)}")
    
    # Take the first slice
    slice_data = quotes[:SLICE_SIZE]
    
    with open(EXPORT_FILE, "w", encoding="utf-8") as f:
        for i, q in enumerate(slice_data):
            f.write(f"--- LOG ITEM {i+1} ---\n")
            f.write(q + "\n\n")

    print(f"Successfully exported {len(slice_data)} quotes to {EXPORT_FILE}")
    print("You can now upload this file to Claude along with the instructions.")

if __name__ == "__main__":
    prepare_slice()
