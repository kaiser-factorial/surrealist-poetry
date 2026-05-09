import os

input_file = "data/bashforever_mixed_train_no_closers.txt"
output_file = "data/bashforever_mixed_train_think_only.txt"

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    stripped = line.strip()
    if stripped == "<think-in>" or stripped == "</think-in>":
        continue # Skip these entirely to flatten the structure
    elif stripped == "<think-out>":
        new_lines.append("<think>\n")
    elif stripped == "</think-out>":
        new_lines.append("</think>\n")
    else:
        new_lines.append(line)

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"Created {output_file} without nested tags!")
