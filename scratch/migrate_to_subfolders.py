import os
import shutil
import glob
import re

base_dir = r"d:\File of ML\Machine-Learning-Note"
notes_dir = os.path.join(base_dir, "notes")
js_dir = os.path.join(base_dir, "js")
courses_dir = os.path.join(base_dir, "courses")

# Create subfolders
os.makedirs(os.path.join(notes_dir, "cs229"), exist_ok=True)
os.makedirs(os.path.join(notes_dir, "cs336"), exist_ok=True)
os.makedirs(os.path.join(js_dir, "cs229"), exist_ok=True)
os.makedirs(os.path.join(js_dir, "cs336"), exist_ok=True)

# 1. Move notes
for f in glob.glob(os.path.join(notes_dir, "cs229-*.md")):
    dest = os.path.join(notes_dir, "cs229", os.path.basename(f))
    shutil.move(f, dest)
    print(f"Moved note: {os.path.basename(f)} -> notes/cs229/")

for f in glob.glob(os.path.join(notes_dir, "cs336-*.md")):
    dest = os.path.join(notes_dir, "cs336", os.path.basename(f))
    shutil.move(f, dest)
    print(f"Moved note: {os.path.basename(f)} -> notes/cs336/")

# 2. Move js chart files
for f in glob.glob(os.path.join(js_dir, "cs229-*.js")):
    dest = os.path.join(js_dir, "cs229", os.path.basename(f))
    shutil.move(f, dest)
    print(f"Moved js: {os.path.basename(f)} -> js/cs229/")

for f in glob.glob(os.path.join(js_dir, "cs336-*.js")):
    dest = os.path.join(js_dir, "cs336", os.path.basename(f))
    shutil.move(f, dest)
    print(f"Moved js: {os.path.basename(f)} -> js/cs336/")

# 3. Update CS229 HTML files
for html_file in glob.glob(os.path.join(courses_dir, "cs229", "*.html")):
    if os.path.basename(html_file) == "index.html":
        continue
    with open(html_file, "r", encoding="utf-8") as fp:
        content = fp.read()
    
    # Update data-markdown
    content = re.sub(r'data-markdown="\.\./\.\./notes/(cs229-[^"]+)"', r'data-markdown="../../notes/cs229/\1"', content)
    # Update chart script src
    content = re.sub(r'src="\.\./\.\./js/(cs229-[^"]+)"', r'src="../../js/cs229/\1"', content)
    
    with open(html_file, "w", encoding="utf-8") as fp:
        fp.write(content)
    print(f"Updated HTML: courses/cs229/{os.path.basename(html_file)}")

# 4. Update CS336 HTML files
for html_file in glob.glob(os.path.join(courses_dir, "cs336", "*.html")):
    if os.path.basename(html_file) == "index.html":
        continue
    with open(html_file, "r", encoding="utf-8") as fp:
        content = fp.read()
    
    # Update data-markdown
    content = re.sub(r'data-markdown="\.\./\.\./notes/(cs336-[^"]+)"', r'data-markdown="../../notes/cs336/\1"', content)
    # Update chart script src
    content = re.sub(r'src="\.\./\.\./js/(cs336-[^"]+)"', r'src="../../js/cs336/\1"', content)
    
    with open(html_file, "w", encoding="utf-8") as fp:
        fp.write(content)
    print(f"Updated HTML: courses/cs336/{os.path.basename(html_file)}")

print("Migration completed successfully!")
