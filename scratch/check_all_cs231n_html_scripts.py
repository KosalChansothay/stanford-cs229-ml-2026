import os
import glob
import re

courses_dir = r"d:\File of ML\Machine-Learning-Note\courses\cs231n"
html_files = sorted(glob.glob(os.path.join(courses_dir, "lecture-*.html")), key=lambda x: int(re.search(r'lecture-(\d+)', x).group(1)))

print(f"=== CHECKING SCRIPT INCLUSIONS ACROSS {len(html_files)} CS231N HTML FILES ===")

updated = 0
for hf in html_files:
    fname = os.path.basename(hf)
    lec_num = int(re.search(r'lecture-(\d+)', fname).group(1))
    with open(hf, "r", encoding="utf-8") as f:
        content = f.read()
        
    orig = content
    
    # Ensure highlight.js CDN stylesheet in <head>
    if "highlight.js" not in content[:content.find("</head>")]:
        content = content.replace("</head>", '    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">\n</head>')
        
    # Ensure highlight.js script before script.js
    if "highlight.min.js" not in content:
        content = content.replace('<script src="../../js/script.js', '<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>\n    <script src="../../js/script.js')
        
    if content != orig:
        with open(hf, "w", encoding="utf-8") as f:
            f.write(content)
        updated += 1
        print(f"Updated scripts in {fname}")

print(f"Verified {len(html_files)} files. Updated {updated} files.")
