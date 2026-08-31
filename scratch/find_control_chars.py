import os
import glob
import re

base_dir = r"d:\File of ML\Machine-Learning-Note"
cs231n_notes = os.path.join(base_dir, "notes", "cs231n")

note_files = sorted(glob.glob(os.path.join(cs231n_notes, "*.md")))

print("=== CHECKING ACCIDENTAL ESCAPE CONTROL CHARACTERS ===")

corrupted_files = {}

for nf in note_files:
    fname = os.path.basename(nf)
    with open(nf, "rb") as f:
        raw_bytes = f.read()
    
    # Check for \x0c (form feed \f), \x08 (backspace \b), \x0b (vertical tab \v), \t (tab inside math), \a (\x07)
    issues_found = []
    if b'\x0c' in raw_bytes:
        issues_found.append("FormFeed (\\x0c, from \\f)")
    if b'\x08' in raw_bytes:
        issues_found.append("Backspace (\\x08, from \\b)")
    if b'\x07' in raw_bytes:
        issues_found.append("Bell (\\x07, from \\a)")
    if b'\x0b' in raw_bytes:
        issues_found.append("VerticalTab (\\x0b, from \\v)")
        
    text = raw_bytes.decode('utf-8', errors='ignore')
    # Check for \t followed by ext, anh, heta, au, etc.
    tab_latex = re.findall(r'\t(ext|anh|heta|au|imes|o|op|ilde|frac|lambda)', text)
    if tab_latex:
        issues_found.append(f"Tab in LaTeX (\\t from {set(tab_latex)})")
        
    # Check for missing backslash before begin/text/frac/tanh/theta/tau/rho
    missing_bs = re.findall(r'(?<=\s)(egin\{|ext\{|anh\(|heta|au|ho\(|igma)', text)
    if missing_bs:
        issues_found.append(f"Missing backslash before command ({set(missing_bs)})")

    if issues_found:
        corrupted_files[fname] = issues_found
        print(f"[CORRUPTED] {fname}: {', '.join(issues_found)}")

print(f"\nTotal corrupted note files: {len(corrupted_files)} / {len(note_files)}")
