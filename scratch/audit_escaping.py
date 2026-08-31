import os
import glob
import re

base_dir = r"d:\File of ML\Machine-Learning-Note"
notes_dir = os.path.join(base_dir, "notes", "cs231n")

note_files = sorted(glob.glob(os.path.join(notes_dir, "cs231n-lecture-*-notes.md")),
                    key=lambda x: int(re.search(r'lecture-(\d+)', x).group(1)))

print("=== UNESCAPED BRACED SUBSCRIPT (_{) AUDIT FOR CS231N ===")
print("Pattern: a '_' immediately followed by '{' that is NOT preceded by a backslash.")
print("Per AI_CONTEXT convention, braced subscripts inside math should be \\_{ ... }.\n")

total = 0
for nf in note_files:
    fname = os.path.basename(nf)
    with open(nf, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    hits = []
    for idx, line in enumerate(lines, 1):
        # find every '_' that is followed by '{' and not preceded by backslash
        for m in re.finditer(r'(?<!\\)_{', line):
            # get surrounding context
            s = max(0, m.start() - 12)
            e = min(len(line), m.end() + 12)
            ctx = line[s:e]
            hits.append((idx, ctx))
    if hits:
        total += len(hits)
        print("\n[%s] %d unescaped _{" % (fname, len(hits)))
        for idx, ctx in hits[:20]:
            print(f"    L{idx}: ...{ctx}...")

print("\nTOTAL unescaped _{" + ": %d" % total)

print("\n\n=== SIMPLE UNBRACED SUBSCRIPTS (e.g. N_c) ===")
print("These are acceptable per reference convention; listing count only per file.\n")
for nf in note_files:
    fname = os.path.basename(nf)
    with open(nf, "r", encoding="utf-8") as f:
        content = f.read()
    # count _<single char or digit> that is not followed by {
    unbr = len(re.findall(r'(?<!\\)_([A-Za-z0-9])(?![\w{])', content))
    print(f"  {fname}: {unbr} unbraced subscripts")
