import os
import glob
import re

base_dir = r"d:\File of ML\Machine-Learning-Note"
js_dir = os.path.join(base_dir, "js", "cs231n")

chart_files = sorted(glob.glob(os.path.join(js_dir, "*.js")), key=lambda x: int(re.search(r'lecture(\d+)', x).group(1)))

print(f"=== UPGRADING {len(chart_files)} CS231N CHART SCRIPTS ===")

for cf in chart_files:
    fname = os.path.basename(cf)
    with open(cf, "r", encoding="utf-8") as f:
        code = f.read()
        
    orig = code
    
    # 1. Add paper_bgcolor and plot_bgcolor if missing in layout
    if "paper_bgcolor" not in code:
        # insert after title: { ... }, or at start of layout object
        code = re.sub(
            r'(var\s+layout\w*\s*=\s*\{)',
            r"\1\n            paper_bgcolor: 'rgba(0,0,0,0)',\n            plot_bgcolor: 'rgba(0,0,0,0)',",
            code
        )
        
    # 2. Add gridcolor: '#e5e5e5' to xaxis and yaxis if missing
    def fix_axis(match):
        axis_block = match.group(0)
        if "gridcolor" not in axis_block:
            axis_block = axis_block.rstrip(" }") + ", gridcolor: '#e5e5e5' }"
        return axis_block

    code = re.sub(r'xaxis\d*:\s*\{[^}]+\}', fix_axis, code)
    code = re.sub(r'yaxis\d*:\s*\{[^}]+\}', fix_axis, code)
    
    # 3. Standardize margins if needed
    if "margin:" in code:
        code = re.sub(r'margin:\s*\{[^}]+\}', "margin: { t: 90, b: 55, l: 70, r: 50 }", code)
    else:
        code = re.sub(
            r'(var\s+layout\w*\s*=\s*\{[\s\S]*?)(height:\s*\d+,)',
            r"\1\2\n            margin: { t: 90, b: 55, l: 70, r: 50 },",
            code
        )

    if code != orig:
        with open(cf, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"Upgraded chart layout: {fname}")

print("All CS231N chart scripts upgraded!")
