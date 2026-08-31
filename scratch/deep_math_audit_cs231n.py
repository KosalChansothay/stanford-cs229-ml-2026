import os
import glob
import re

base_dir = r"d:\File of ML\Machine-Learning-Note"
cs231n_notes = os.path.join(base_dir, "notes", "cs231n")

note_files = sorted(glob.glob(os.path.join(cs231n_notes, "cs231n-lecture-*-notes.md")), key=lambda x: int(re.search(r'lecture-(\d+)', x).group(1)))

print("=== DEEP MATH & LATEX AUDIT FOR CS231N ===")

for nf in note_files:
    fname = os.path.basename(nf)
    with open(nf, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 1. Check unbalanced $$
    lines = content.splitlines()
    display_math_blocks = []
    in_block = False
    block_start_line = 0
    current_block = []
    
    for idx, line in enumerate(lines, 1):
        # count $$ in line
        c = line.count("$$")
        if c == 1:
            if not in_block:
                in_block = True
                block_start_line = idx
                current_block = [line]
            else:
                in_block = False
                current_block.append(line)
                display_math_blocks.append((block_start_line, idx, "\n".join(current_block)))
                current_block = []
        elif c == 2:
            # single line $$ ... $$
            display_math_blocks.append((idx, idx, line))
        elif c > 2:
            print(f"[{fname}] Line {idx}: Multiple $$ in single line: {line}")
            
    if in_block:
        print(f"[{fname}] UNCLOSED display math block starting at line {block_start_line}!")
        print(f"Content starting at line {block_start_line}:\n" + "\n".join(current_block[:5]))

    # 2. Check inline math $ ... $
    # Find unescaped single dollars
    # Strip out display math blocks first
    clean_text = re.sub(r'\$\$[\s\S]*?\$\$', '', content)
    # Strip out code blocks
    clean_text = re.sub(r'```[\s\S]*?```', '', clean_text)
    # Strip out inline code
    clean_text = re.sub(r'`[^`]+`', '', clean_text)
    
    # Check for unescaped $ count
    single_dollars = len(re.findall(r'(?<!\\)\$', clean_text))
    if single_dollars % 2 != 0:
        print(f"[{fname}] UNBALANCED inline math $ count: {single_dollars}")

    # 3. Check for common LaTeX typos
    latex_patterns = [
        (r'\\frac\{[^{}]*\}\s*$', "Incomplete \\frac"),
        (r'\\sqrt\{[^{}]*\}\s*$', "Incomplete \\sqrt"),
        (r'\\sum_\{[^{}]*$', "Unclosed \\sum subscript"),
        (r'\\begin\{(aligned|matrix|bmatrix|pmatrix|array)\}(?![\s\S]*\\end\{\1\})', "Unclosed LaTeX environment"),
        (r'\\text\{[^{}]*$', "Unclosed \\text"),
        (r'\\mathbf\{[^{}]*$', "Unclosed \\mathbf"),
        (r'\\left[([{](?![\s\S]*\\right[)\]}])', "Unmatched \\left delimiter"),
    ]
    
    for pat, desc in latex_patterns:
        matches = re.findall(pat, content)
        if matches:
            print(f"[{fname}] LaTeX error: {desc} -> {matches}")

print("Deep math audit complete.")
