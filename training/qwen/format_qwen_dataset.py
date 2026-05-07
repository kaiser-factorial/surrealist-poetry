import re
import json
import os

# The same system prompt used in server_2_director.py
# SYSTEM_PROMPT = "You are an introspective forum-goer. You observe the disjoint dialogues of your peers and reflect on their topology and form, occasionally steering the conversation toward deeper resonance."
SYSTEM_PROMPT = "You are an introspective forum-goer. You analyze the conversations around you, reflecting on their form and topology."

def parse_to_chatml(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern to find <other> and <me> blocks
    pattern = re.compile(r"<other>(.*?)</other>\s*<me>(.*?)</me>", re.DOTALL)
    matches = pattern.findall(content)
    
    chatml_data = []
    for other_text, me_text in matches:
        # We add a generic speaker tag since the training data doesn't specify who said it.
        # This matches the "[SPEAKER]: text" format used in server_2_director.py
        user_content = f"[OTHER]: {other_text.strip()}"
        
        entry = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": me_text.strip()}
            ]
        }
        chatml_data.append(entry)
        
    return chatml_data

def main():
    input_file = "training/qwen/synthetic_thinking_data.txt"
    output_file = "training/qwen/prepared_chatml_dataset.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Formatting {input_file} for ChatML (with speaker tags)...")
    data = parse_to_chatml(input_file)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Successfully prepared {len(data)} ChatML entries in {output_file}")

if __name__ == "__main__":
    main()
