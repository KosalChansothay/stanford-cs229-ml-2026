import os
import glob
import re

base_dir = r"d:\File of ML\Machine-Learning-Note"
html_files = glob.glob(os.path.join(base_dir, "**", "*.html"), recursive=True)

print(f"Updating navbar header across {len(html_files)} HTML files...")

count = 0
for html_file in html_files:
    with open(html_file, "r", encoding="utf-8") as fp:
        content = fp.read()
    
    orig = content
    
    # Replace sidebar-name
    content = re.sub(r'<div class="sidebar-name">[^<]+</div>', '<div class="sidebar-name">ML Notes</div>', content)
    
    # Replace sidebar-role
    content = re.sub(r'<div class="sidebar-role">[^<]+</div>', '<div class="sidebar-role">Stanford AI &amp; ML Series</div>', content)
    
    # Replace sidebar-bio
    content = re.sub(r'<div class="sidebar-bio">[\s\S]*?</div>', '<div class="sidebar-bio">Lecture notes and study materials for Stanford machine learning and artificial intelligence courses.</div>', content)
    
    if content != orig:
        with open(html_file, "w", encoding="utf-8") as fp:
            fp.write(content)
        count += 1
        print(f"Updated: {os.path.relpath(html_file, base_dir)}")

print(f"Successfully updated navbar header across {count} files!")
