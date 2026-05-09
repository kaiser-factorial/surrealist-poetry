import re
import random
import os

input_file = '/Users/corinakaiser/Desktop/VAT/brain-vat-training/training/qwen/synthetic_thinking_data.txt'
output_file = '/Users/corinakaiser/Desktop/VAT/brain-vat-training/training/gpt2/bashforever_mixed_train_no_closers.txt'

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

pattern = re.compile(r'(<other>.*?</other>.*?<me>.*?</me>)', re.DOTALL)
entries = pattern.findall(content)

random.seed(42)
choices = [0, 1, 2] * (len(entries) // 3 + 1)
choices = choices[:len(entries)]
random.shuffle(choices)

processed_entries = []
for idx, entry in enumerate(entries):
    choice = choices[idx]
    
    if choice == 0:
        entry = entry.replace('<other>', '[ABACI]')
        entry = entry.replace('</other>', '')
        entry = entry.replace('<me>', '[MAUK]')
        entry = entry.replace('</me>', '')
    elif choice == 1:
        entry = entry.replace('<other>', '[MAUK]')
        entry = entry.replace('</other>', '')
        entry = entry.replace('<me>', '[ABACI]')
        entry = entry.replace('</me>', '')
    else:
        entry = entry.replace('<other>', '[OTHER]')
        entry = entry.replace('</other>', '')
        entry = entry.replace('<me>', '[ME]')
        entry = entry.replace('</me>', '')
        
    entry = entry + '\n<|endoftext|>'
    processed_entries.append(entry)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(processed_entries))

print(f"Processed {len(entries)} entries and saved to {output_file}")
