import os
import glob
import re
import json

base_dir = r"d:\File of ML\Machine-Learning-Note"
cs231n_courses = os.path.join(base_dir, "courses", "cs231n")
cs231n_notes = os.path.join(base_dir, "notes", "cs231n")
cs231n_js = os.path.join(base_dir, "js", "cs231n")

print("=" * 60)
print("COMPREHENSIVE AUDIT OF CS231N")
print("=" * 60)

issues = []

# 1. Check HTML files
html_files = sorted(glob.glob(os.path.join(cs231n_courses, "lecture-*.html")), key=lambda x: int(re.search(r'lecture-(\d+)', x).group(1)))
print(f"\n[1] Checking {len(html_files)} Lecture HTML files...")

for hf in html_files:
    fname = os.path.basename(hf)
    lec_num = int(re.search(r'lecture-(\d+)', fname).group(1))
    with open(hf, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check data-markdown
    md_match = re.search(r'data-markdown="([^"]+)"', content)
    if not md_match:
        issues.append(f"HTML: {fname} missing data-markdown attribute")
    else:
        md_rel = md_match.group(1)
        abs_md = os.path.normpath(os.path.join(os.path.dirname(hf), md_rel))
        if not os.path.isfile(abs_md):
            issues.append(f"HTML: {fname} data-markdown target not found: {md_rel}")
    
    # Check chart script
    js_match = re.search(r'src="(\.\./\.\./js/cs231n/[^"]+)"', content)
    if not js_match:
        # Check if lecture has a chart script
        expected_js = os.path.join(cs231n_js, f"cs231n-lecture{lec_num}-charts.js")
        if os.path.isfile(expected_js):
            issues.append(f"HTML: {fname} has chart script on disk ({os.path.basename(expected_js)}) but not referenced in HTML")
    else:
        js_rel = js_match.group(1)
        abs_js = os.path.normpath(os.path.join(os.path.dirname(hf), js_rel))
        if not os.path.isfile(abs_js):
            issues.append(f"HTML: {fname} chart script target not found: {js_rel}")

    # Check script.js and marked.js
    if "marked.min.js" not in content:
        issues.append(f"HTML: {fname} missing marked.min.js CDN")
    if "../../js/script.js" not in content:
        issues.append(f"HTML: {fname} missing ../../js/script.js")
    
    # Check pagination
    if "lecture-pagination" not in content:
        issues.append(f"HTML: {fname} missing lecture-pagination block")
    
    # Check brand navbar
    if "ML Notes" not in content:
        issues.append(f"HTML: {fname} navbar does not use 'ML Notes'")

# 2. Check Markdown files
note_files = sorted(glob.glob(os.path.join(cs231n_notes, "cs231n-lecture-*-notes.md")), key=lambda x: int(re.search(r'lecture-(\d+)', x).group(1)))
print(f"\n[2] Checking {len(note_files)} Markdown Note files...")

for nf in note_files:
    fname = os.path.basename(nf)
    with open(nf, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Strip math blocks and code blocks before checking timestamps
    clean_text = re.sub(r'\$\$[\s\S]*?\$\$', '', content)
    clean_text = re.sub(r'\$[^$\n]+\$', '', clean_text)
    clean_text = re.sub(r'```[\s\S]*?```', '', clean_text)
    
    # Check transcript timestamps like [358, 429]
    ts_matches = re.findall(r'\[\d{2,4}(?:,\s*\d{2,4})+\]', clean_text)
    if ts_matches:
        issues.append(f"MARKDOWN: {fname} contains unstripped transcript timestamps: {ts_matches[:5]}")
    
    # Check single timestamps like [358] outside of markdown links
    non_link_single = re.findall(r'(?<!\]\()(?:(?<=\s)|(?<=^))\[(\d{2,4})\](?!\()', clean_text)
    if non_link_single:
        issues.append(f"MARKDOWN: {fname} contains single numeric timestamps: {non_link_single[:5]}")
    
    # Check for placeholder text
    if "[Insert diagram:" in content:
        issues.append(f"MARKDOWN: {fname} contains unreplaced '[Insert diagram:' placeholder")

    # Check for unescaped multiple underscores in paragraphs (causing accidental italicization)
    lines = content.splitlines()
    for i, l in enumerate(lines, 1):
        if not l.startswith("$$") and not l.startswith("```"):
            # Check for unescaped _{
            if re.search(r'(?<!\\)_\{', l):
                # Count occurrences outside $...$
                # simple check
                pass

    # Check delimiter balance
    dollar_count = content.count("$$")
    if dollar_count % 2 != 0:
        issues.append(f"MATH: {fname} has unbalanced display math $$ delimiters (count: {dollar_count})")

# 3. Check Chart Scripts
js_files = sorted(glob.glob(os.path.join(cs231n_js, "cs231n-lecture*-charts.js")), key=lambda x: int(re.search(r'lecture(\d+)', x).group(1)))
print(f"\n[3] Checking {len(js_files)} JS Chart files...")

for jf in js_files:
    fname = os.path.basename(jf)
    lec_num = int(re.search(r'lecture(\d+)', fname).group(1))
    with open(jf, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find all document.getElementById or plotly target ids
    chart_ids = re.findall(r'document\.getElementById\([\'"]([^\'"]+)[\'"]\)', content)
    chart_ids += re.findall(r'Plotly\.newPlot\([\'"]([^\'"]+)[\'"]', content)
    chart_ids = list(set(chart_ids))
    
    # Check if matching markdown file contains these chart IDs
    matching_md = os.path.join(cs231n_notes, f"cs231n-lecture-{lec_num}-notes.md")
    if os.path.isfile(matching_md):
        with open(matching_md, "r", encoding="utf-8") as mf:
            md_content = mf.read()
        for cid in chart_ids:
            if cid not in md_content:
                issues.append(f"JS-MD MISMATCH: {fname} renders chart id '{cid}', but id is NOT in {os.path.basename(matching_md)}")
    
    # Check transparency
    if "plot_bgcolor: 'white'" in content or 'plot_bgcolor: "white"' in content:
        issues.append(f"CHART THEME: {fname} has hardcoded plot_bgcolor: 'white' (should be transparent rgba(0,0,0,0))")
    if "paper_bgcolor: 'white'" in content or 'paper_bgcolor: "white"' in content:
        issues.append(f"CHART THEME: {fname} has hardcoded paper_bgcolor: 'white'")
        
    # Check margins
    margin_match = re.search(r'margin:\s*\{([^}]+)\}', content)
    if margin_match:
        m_str = margin_match.group(1)
        t_val = re.search(r't:\s*(\d+)', m_str)
        r_val = re.search(r'r:\s*(\d+)', m_str)
        l_val = re.search(r'l:\s*(\d+)', m_str)
        b_val = re.search(r'b:\s*(\d+)', m_str)
        if t_val and int(t_val.group(1)) < 80:
            issues.append(f"CHART MARGIN: {fname} top margin too small ({t_val.group(1)} < 80)")
        if r_val and int(r_val.group(1)) < 45:
            issues.append(f"CHART MARGIN: {fname} right margin too small ({r_val.group(1)} < 45)")
            
    # Check log axes for clean array ticks
    if "type: 'log'" in content or 'type: "log"' in content:
        if "tickmode: 'array'" not in content and 'tickmode: "array"' not in content:
            issues.append(f"CHART LOG AXIS: {fname} has log-scale axis without explicit tickmode: 'array'")

print("\n" + "=" * 60)
print(f"AUDIT SUMMARY: {len(issues)} ISSUES FOUND")
print("=" * 60)
for iss in issues:
    print(" [!]", iss)

