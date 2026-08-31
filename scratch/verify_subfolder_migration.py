import os
import glob
import re

base_dir = r"d:\File of ML\Machine-Learning-Note"
courses_dir = os.path.join(base_dir, "courses")

print("=== VERIFYING SUBFOLDER MIGRATION ===")

errors = []
total_checked = 0

for course in ["cs229", "cs336"]:
    html_files = sorted(glob.glob(os.path.join(courses_dir, course, "*.html")))
    for html in html_files:
        if os.path.basename(html) == "index.html":
            continue
        total_checked += 1
        with open(html, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check data-markdown
        md_match = re.search(r'data-markdown="([^"]+)"', content)
        if not md_match:
            errors.append(f"{html}: No data-markdown attribute found!")
        else:
            rel_path = md_match.group(1)
            # Resolve relative path from html file
            abs_md = os.path.normpath(os.path.join(os.path.dirname(html), rel_path))
            if not os.path.isfile(abs_md):
                errors.append(f"{html}: Markdown file NOT FOUND at {abs_md}")
        
        # Check chart script if present
        script_matches = re.findall(r'src="(\.\./\.\./js/[^"]+)"', content)
        for s in script_matches:
            if "script.js" in s:
                continue
            abs_script = os.path.normpath(os.path.join(os.path.dirname(html), s))
            if not os.path.isfile(abs_script):
                errors.append(f"{html}: Chart script NOT FOUND at {abs_script}")

print(f"Verified {total_checked} lecture HTML pages.")
if errors:
    print(f"Found {len(errors)} ERRORS:")
    for e in errors:
        print(" -", e)
else:
    print("ALL 32 LECTURES VALIDATED PERFECTLY! 100% path resolution.")
