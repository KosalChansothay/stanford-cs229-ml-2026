import os
import glob
import re

base_dir = r"d:\File of ML\Machine-Learning-Note"
cs231n_notes = os.path.join(base_dir, "notes", "cs231n")

note_files = sorted(glob.glob(os.path.join(cs231n_notes, "*.md")))

print("=== DETAILED CORRUPTION DIAGNOSTIC ===")

for nf in note_files:
    fname = os.path.basename(nf)
    with open(nf, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    
    corruptions = []
    for idx, line in enumerate(lines, 1):
        # find any control characters except newline
        for ch_idx, ch in enumerate(line):
            code = ord(ch)
            if code in [7, 8, 11, 12] or (code == 9 and re.search(r'\t[a-zA-Z]', line)):
                corruptions.append((idx, code, line.strip()))
                break
            # check for corrupted math keywords
            if re.search(r'(\x0crac|rac|\x08egin|\x07pprox|\x07lpha|\x08eta|\th|	ext|	anh|	heta|	au|	imes|	ilde|	op|	o\b)', line):
                corruptions.append((idx, "regex", line.strip()))
                break
                
    if corruptions:
        print(f"\n--- {fname} ({len(corruptions)} corrupted lines) ---")
        for line_no, code, text in corruptions[:8]:
            # sanitize display
            sanitized = repr(text)
            print(f"  Line {line_no} [{code}]: {sanitized}")
        if len(corruptions) > 8:
            print(f"  ... and {len(corruptions)-8} more lines")
