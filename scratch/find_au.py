import re
with open(r"d:\File of ML\Machine-Learning-Note\notes\cs231n\cs231n-lecture-1-notes.md", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, 1):
        m = re.findall(r'(?<=\s)au\b', line)
        if m:
            print(f"Line {idx}: {line.strip()}")
