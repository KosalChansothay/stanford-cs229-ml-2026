import os
import glob
import re

base_dir = r"d:\File of ML\Machine-Learning-Note"
js_dir = os.path.join(base_dir, "js", "cs231n")
notes_dir = os.path.join(base_dir, "notes", "cs231n")

chart_files = sorted(glob.glob(os.path.join(js_dir, "*.js")), key=lambda x: int(re.search(r'lecture(\d+)', x).group(1)))

print(f"=== DETAILED CS231N JS CHART AUDIT ({len(chart_files)} files) ===")

all_divs = {}

for cf in chart_files:
    fname = os.path.basename(cf)
    lec_num = int(re.search(r'lecture(\d+)', fname).group(1))
    with open(cf, "r", encoding="utf-8") as f:
        code = f.read()
        
    div_ids = re.findall(r'draw[a-zA-Z0-9_]*\([\'"]([^\'"]+)[\'"]\)', code)
    div_ids += re.findall(r'render[a-zA-Z0-9_]*\([\'"]([^\'"]+)[\'"]\)', code)
    div_ids += re.findall(r'document\.getElementById\([\'"]([^\'"]+)[\'"]\)', code)
    div_ids = sorted(list(set(div_ids)))
    
    # Check matching markdown
    note_path = os.path.join(notes_dir, f"cs231n-lecture-{lec_num}-notes.md")
    with open(note_path, "r", encoding="utf-8") as nf:
        note_content = nf.read()
    
    # Find all plotly- divs in note
    md_divs = re.findall(r'<div id="([^"]+)"\s+class="plotly-chart"', note_content)
    
    print(f"\n[Lecture {lec_num}] {fname}:")
    print(f"  JS Targets: {div_ids}")
    print(f"  MD Containers: {md_divs}")
    
    missing_in_md = [d for d in div_ids if d not in md_divs]
    missing_in_js = [d for d in md_divs if d not in div_ids]
    if missing_in_md:
        print(f"  [!] MISMATCH: In JS but missing in MD: {missing_in_md}")
    if missing_in_js:
        print(f"  [!] MISMATCH: In MD but missing in JS: {missing_in_js}")
        
    # Check bgcolor
    if "rgba(0,0,0,0)" not in code and "transparent" not in code:
        print("  [!] Missing transparent paper_bgcolor / plot_bgcolor")

print("\nAudit finished.")
