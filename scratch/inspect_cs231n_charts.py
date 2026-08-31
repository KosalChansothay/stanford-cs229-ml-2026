import os
import glob
import re

base_dir = r"d:\File of ML\Machine-Learning-Note"
js_dir = os.path.join(base_dir, "js", "cs231n")
notes_dir = os.path.join(base_dir, "notes", "cs231n")

chart_files = sorted(glob.glob(os.path.join(js_dir, "*.js")), key=lambda x: int(re.search(r'lecture(\d+)', x).group(1)))

print(f"=== INSPECTING {len(chart_files)} CS231N PLOTLY CHART SCRIPTS ===")

for cf in chart_files:
    fname = os.path.basename(cf)
    lec_num = int(re.search(r'lecture(\d+)', fname).group(1))
    with open(cf, "r", encoding="utf-8") as f:
        code = f.read()
    
    print(f"\n--- {fname} ---")
    
    # 1. Check IIFE
    if not (code.strip().startswith("(function") and code.strip().endswith("})();")):
        print("  [!] Warning: Not strictly enclosed in (function() { ... })();")
        
    # 2. Check markdown:rendered listener
    if "markdown:rendered" not in code:
        print("  [!] Warning: Missing 'markdown:rendered' event listener")
        
    # 3. Check window export
    exports = re.findall(r'window\.(render[a-zA-Z0-9_]+)\s*=', code)
    print(f"  Exports: {exports}")
    
    # 4. Find all Plotly.newPlot calls
    plots = re.findall(r'Plotly\.newPlot\(\s*[\'"]([^\'"]+)[\'"]', code)
    print(f"  Chart Containers: {plots}")
    
    # Check matching note file
    note_path = os.path.join(notes_dir, f"cs231n-lecture-{lec_num}-notes.md")
    if os.path.isfile(note_path):
        with open(note_path, "r", encoding="utf-8") as nf:
            note_content = nf.read()
        for p in plots:
            if p not in note_content:
                print(f"  [ERROR] Container '{p}' NOT FOUND in {os.path.basename(note_path)}!")
            else:
                print(f"  [OK] Container '{p}' present in {os.path.basename(note_path)}")
    else:
        print(f"  [ERROR] Matching note file not found: {note_path}")
        
    # 5. Check layout properties: paper_bgcolor, plot_bgcolor, margins, title
    layouts = re.findall(r'var\s+layout\w*\s*=\s*\{([\s\S]*?)\};', code)
    for idx, l in enumerate(layouts, 1):
        if "paper_bgcolor" not in l or "plot_bgcolor" not in l:
            print(f"  [!] Layout {idx}: missing explicit paper_bgcolor / plot_bgcolor (rgba(0,0,0,0))")
        if "margin" not in l:
            print(f"  [!] Layout {idx}: missing margin object")
        else:
            t = re.search(r't:\s*(\d+)', l)
            r = re.search(r'r:\s*(\d+)', l)
            b = re.search(r'b:\s*(\d+)', l)
            l_val = re.search(r'l:\s*(\d+)', l)
            t_num = int(t.group(1)) if t else 0
            r_num = int(r.group(1)) if r else 0
            b_num = int(b.group(1)) if b else 0
            l_num = int(l_val.group(1)) if l_val else 0
            if t_num < 85:
                print(f"  [!] Layout {idx}: top margin t={t_num} < 85 (title might be tight)")
            if r_num < 45:
                print(f"  [!] Layout {idx}: right margin r={r_num} < 45")
                
    # 6. Check for log axes without explicit ticks
    if "type: 'log'" in code or 'type: "log"' in code:
        if "tickvals" not in code:
            print("  [!] Warning: Log-scale axis without explicit tickvals array")

print("\nChart inspection complete.")
