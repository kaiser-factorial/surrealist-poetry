import re
import random
import os
from collections import Counter

# File paths
INPUT_FILE = "training/shared/corpus/bash_logs.txt"
OUTPUT_FILE = "training/shared/corpus/bash_logs_id.txt"

# Token handling
EOS_PATTERN = r"<\|endoftext\|>"
EOS_LITERAL = "<|endoftext|>" 

def swap_identities(content):
    # Split using the regex
    blocks = re.split(EOS_PATTERN, content)
    new_blocks = []

    for block in blocks:
        raw_block = block
        block = block.strip()
        if not block:
            continue

        # Find all name mentions 
        # We capture the RAW match including internal whitespace for exact replacement
        names_found = set()
        
        # 1. Speaking: Capture the content inside < > exactly
        names_found.update(re.findall(r"<([^>]+)>", block))
        # 2. Speaking: Name: at start of line
        names_found.update(re.findall(r"(?:^|(?<=\n))([a-zA-Z0-9_\[\]\|\\`-{}\^ ]+): ", block))
        # 3. Action: * Name
        names_found.update(re.findall(r"(?:^|(?<=\n))\*\s+([a-zA-Z0-9_\[\]\|\\`-{}\^ ]+)", block))
        # 4. Markers
        names_found.update(re.findall(r"(?:^|(?<=\n))(?:-->|<--|\*\*\*)\s+([a-zA-Z0-9_\[\]\|\\`-{}\^ ]+)", block))
        # 5. join/quit
        names_found.update(re.findall(r"(?:join|quit):\s+\(([^ \)]+)\)", block))
        
        # Determine the "canonical" version of each name for counting
        name_map_raw_to_canonical = {}
        canonical_counts = Counter()
        
        for raw in names_found:
            canonical = raw.strip()
            if not canonical or canonical.lower() in ["topic", "none", "server", "nick"]: continue
            if canonical.startswith("#"): continue
            if len(canonical) > 25: continue
            
            name_map_raw_to_canonical[raw] = canonical
            canonical_counts[canonical] += 1

        if not canonical_counts:
            new_blocks.append(raw_block)
            continue

        # Rank canonical names
        ranked_canonical = [name for name, _ in canonical_counts.most_common()]
        
        # Identity assignment
        bot_ids = ["MAUK", "ABACI"]
        random.shuffle(bot_ids)
        
        ident_map = {}
        if len(ranked_canonical) >= 1:
            ident_map[ranked_canonical[0]] = bot_ids[0]
        if len(ranked_canonical) >= 2:
            ident_map[ranked_canonical[1]] = bot_ids[1]
        for i in range(2, len(ranked_canonical)):
            ident_map[ranked_canonical[i]] = "USER"

        # Apply replacements
        new_block = raw_block
        
        # We iterate over the RAW names to ensure we hit < Alkivar> exactly as found
        for raw_name in sorted(names_found, key=len, reverse=True):
            if raw_name in name_map_raw_to_canonical:
                canonical = name_map_raw_to_canonical[raw_name]
                new_identity = ident_map[canonical]
                
                esc = re.escape(raw_name)
                # Replace <RawName>
                new_block = re.sub(rf"<{esc}>", f"<{new_identity}>", new_block)
                # Replace RawName: 
                new_block = re.sub(rf"(^|(?<=\n)){esc}: ", f"\\1{new_identity}: ", new_block)
                # Replace * RawName
                new_block = re.sub(rf"(^|(?<=\n))\*\s+{esc}", f"\\1* {new_identity}", new_block)
                # Replace Join/Quit
                new_block = re.sub(rf"(-->|<--|\*\*\*)\s+{esc}", f"\\1 {new_identity}", new_block)
                new_block = re.sub(rf"(join|quit):\s+\({esc}\)", f"\\1: ({new_identity})", new_block)
        
        new_blocks.append(new_block)

    token = EOS_LITERAL if EOS_LITERAL != "[[REPLACE_ME_GURL]]" else "<|endoftext|>"
    return (token + "\n").join(new_blocks) + token

if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            data = f.read()
        modified_data = swap_identities(data)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(modified_data)
        print(f"Successfully processed {INPUT_FILE} -> {OUTPUT_FILE}")
