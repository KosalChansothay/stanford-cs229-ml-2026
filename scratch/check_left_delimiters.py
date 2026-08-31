import re

for doc in [r"d:\File of ML\Machine-Learning-Note\notes\cs231n\cs231n-lecture-7-notes.md", r"d:\File of ML\Machine-Learning-Note\notes\cs231n\cs231n-lecture-17-notes.md"]:
    with open(doc, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print("="*40, doc, "="*40)
    for idx, line in enumerate(lines, 1):
        if r"\left" in line:
            print(f"Line {idx}: {line.strip()}")
