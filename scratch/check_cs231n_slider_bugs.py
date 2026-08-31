import os
import glob
import re

base_dir = r"d:\File of ML\Machine-Learning-Note"
js_dir = os.path.join(base_dir, "js", "cs231n")

chart_files = sorted(glob.glob(os.path.join(js_dir, "*.js")), key=lambda x: int(re.search(r'lecture(\d+)', x).group(1)))

print(f"=== CHECKING PLOTLY SLIDERS & MENUS ACROSS {len(chart_files)} CS231N CHARTS ===")

for cf in chart_files:
    fname = os.path.basename(cf)
    with open(cf, "r", encoding="utf-8") as f:
        code = f.read()
        
    print(f"\n--- {fname} ---")
    
    # Check sliders
    has_slider = "sliders" in code
    has_menu = "updatemenus" in code
    print(f"  Slider: {has_slider}, Menu: {has_menu}")
    
    if has_slider or has_menu:
        # print the slider/menu code
        steps_match = re.search(r'steps\s*=\s*([\s\S]*?)(?:layout\.sliders|Plotly\.react)', code)
        if steps_match:
            print("  Steps snippet:\n", steps_match.group(0)[:300])
        menu_match = re.search(r'updatemenus\s*=\s*\[([\s\S]*?)\];', code)
        if menu_match:
            print("  Menu snippet:\n", menu_match.group(0)[:300])
