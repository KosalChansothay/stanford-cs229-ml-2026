import glob
import os
import re

base = r"d:\File of ML\Machine-Learning-Note"
notes_dir = os.path.join(base, "notes", "cs231n")

print("=== MATH DELIMITER BALANCE CHECK (all 18 notes) ===")
bs = chr(92)
for f in sorted(glob.glob(os.path.join(notes_dir, "*.md")), key=lambda x: int(re.search(r'lecture-(\d+)', x).group(1))):
    n = os.path.basename(f)
    c = open(f, encoding="utf-8").read()
    lines = c.split("\n")
    issues = []
    # display math $$ balance (per-file count should be even)
    dd = c.count("$$")
    if dd % 2 != 0:
        issues.append("odd $$ count: %d" % dd)
    # per-line \left / \right balance in math
    for i, line in enumerate(lines, 1):
        if "$$" in line:
            lf = line.count(bs + "left[") + line.count(bs + "left(") + line.count(bs + "left\\{")
            rf = line.count(bs + "right]") + line.count(bs + "right)") + line.count(bs + "right\\}")
            if lf != rf:
                issues.append("L%d: left[=%d right]=%d left(=%d right)=%d" % (
                    i,
                    line.count(bs + "left["), line.count(bs + "right]"),
                    line.count(bs + "left("), line.count(bs + "right)")))
    # unescaped braced subscript
    unbr = len(re.findall(r"(?<!\\)_{", c))
    if unbr:
        issues.append("unescaped _{: %d" % unbr)
    print("[%s]: %s" % (n, "OK" if not issues else "; ".join(issues)))

print("\n=== JS SYNTAX CHECK via node ===")
