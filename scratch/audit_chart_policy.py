import glob
import re
import os

base = r"d:\File of ML\Machine-Learning-Note\js\cs231n"
files = sorted(glob.glob(os.path.join(base, "*.js")), key=lambda x: int(re.search(r'lecture(\d+)', x).group(1)))

print("=== CS231N CHART POLICY AUDIT ===")
for f in files:
    n = os.path.basename(f)
    c = open(f, encoding="utf-8").read()
    issues = []
    # 1. Transparent backgrounds
    if "rgba(0,0,0,0)" not in c:
        issues.append("no transparent bgcolor")
    # 2. Log axes must have tickmode array
    for m in re.finditer(r"type:\s*'log'", c):
        if "tickmode: 'array'" not in c:
            issues.append("log axis without tickmode array")
            break
    # 3. Margin check
    m = re.search(r"margin:\s*\{([^}]*)\}", c)
    if m:
        mt = re.search(r"t:\s*(\d+)", m.group(1))
        mr = re.search(r"r:\s*(\d+)", m.group(1))
        ml = re.search(r"l:\s*(\d+)", m.group(1))
        mb = re.search(r"b:\s*(\d+)", m.group(1))
        if mt and int(mt.group(1)) < 80:
            issues.append(f"top margin {mt.group(1)} < 80")
        if mr and int(mr.group(1)) < 45:
            issues.append(f"right margin {mr.group(1)} < 45")
        if not mb:
            issues.append("no bottom margin")
    else:
        issues.append("no margin found")
    # 4. Slider robustness: sliderupdate backup listener
    if "slider" in c.lower():
        if "plotly_sliderupdate" not in c:
            issues.append("slider present but no plotly_sliderupdate backup listener")
    # 5. Trace-count stability for sliders (count traces at step 0 - hard to verify statically)
    print(f"[{n}]: {'OK' if not issues else '; '.join(issues)}")

print("\n=== JS SYNTAX CHECK (node) ===")
