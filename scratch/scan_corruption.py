import os
import glob
import re

base_dir = r"d:\File of ML\Machine-Learning-Note"
notes_dir = os.path.join(base_dir, "notes", "cs231n")

note_files = sorted(glob.glob(os.path.join(notes_dir, "cs231n-lecture-*-notes.md")),
                    key=lambda x: int(re.search(r'lecture-(\d+)', x).group(1)))

print("=== CS231N CORRUPTION / VECTOR-EATEN SCAN ===")
print("Looking for telltale artifacts of the timestamp-stripping regex eating vector contents.\n")

patterns = {
    "=^T / mangled transpose (vector contents eaten)": r'=\s*\^?\s*T\s*\$',
    "empty brackets []": r'\[\s*\]',
    "one-hot-like eaten (digit immediately after '=' then space then '^')": r'=\s*\d*\s*\^',
    "stray '^T' without content": r'=\s*\^T',
    "colon-comma vector truncation ': 0, 0' or ', 0, 0]'": r'\[\s*0\s*\]|\[\s*0\s*,',
    "unterminated bracket [1, 0 or [0,": r'\[\s*[01]\s*,\s*[01]\s*$',
}

for nf in note_files:
    fname = os.path.basename(nf)
    with open(nf, "r", encoding="utf-8") as f:
        content = f.read()
    for label, pat in patterns.items():
        for m in re.finditer(pat, content):
            line = content.count("\n", 0, m.start()) + 1
            s = max(0, m.start() - 25)
            e = min(len(content), m.end() + 25)
            ctx = content[s:e].replace("\n", " ")
            print(f"[{fname} L{line}] ({label}): ...{ctx}...")
