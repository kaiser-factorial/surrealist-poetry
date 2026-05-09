import os

input_file = "data/bashforever_mixed_train_no_closers.txt"
output_file = "data/bashforever_mixed_train_think_only.txt"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("<think-in>", "<think>")
text = text.replace("<think-out>", "<think>")
text = text.replace("</think-in>", "</think>")
text = text.replace("</think-out>", "</think>")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Created {output_file}")
