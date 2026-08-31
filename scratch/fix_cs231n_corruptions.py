import os
import glob
import re

base_dir = r"d:\File of ML\Machine-Learning-Note"
cs231n_notes = os.path.join(base_dir, "notes", "cs231n")

note_files = sorted(glob.glob(os.path.join(cs231n_notes, "*.md")))

print("=== REPAIRING ESCAPED CONTROL CHARACTERS IN CS231N NOTES ===")

for nf in note_files:
    fname = os.path.basename(nf)
    with open(nf, "r", encoding="utf-8") as f:
        content = f.read()
    
    orig = content
    
    # 1. Fix FormFeed (\x0c -> \f)
    content = content.replace('\x0crac', r'\frac')
    content = content.replace('\x0c', r'\f')
    
    # 2. Fix Backspace (\x08 -> \b)
    content = content.replace('\x08egin', r'\begin')
    content = content.replace('\x08eta', r'\beta')
    content = content.replace('\x08', r'\b')
    
    # 3. Fix Bell (\x07 -> \a)
    content = content.replace('\x07pprox', r'\approx')
    content = content.replace('\x07lpha', r'\alpha')
    content = content.replace('\x07', r'\a')
    
    # 4. Fix Tab in LaTeX keywords
    content = re.sub(r'\text(?=[^a-zA-Z]|$|\{)', r'\\text', content) # in case
    content = content.replace('\text', r'\text')
    content = content.replace('\tanh', r'\tanh')
    content = content.replace('\theta', r'\theta')
    content = content.replace('\tau', r'\tau')
    content = content.replace('\times', r'\times')
    content = content.replace('\tilde', r'\tilde')
    content = content.replace('\top', r'\top')
    content = content.replace('\to ', r'\to ')
    content = content.replace('\to$', r'\to$')
    
    # 5. Fix Carriage Return / newline artifacts from \r
    # e.g., \r + ightarrow -> \rightarrow
    content = re.sub(r'[\r\n]+\s*ightarrow', r'\\rightarrow', content)
    content = re.sub(r'[\r\n]+\s*ight', r'\\right', content)
    content = re.sub(r'[\r\n]+\s*ho\b', r'\\rho', content)
    content = re.sub(r'(?<=[(\s$])ho\(([^)]+)\)', r'\\rho(\1)', content)
    content = re.sub(r'(?<=[(\s$])ho\b', r'\\rho', content)
    
    # 6. Fix \b artifacts
    content = re.sub(r'(?<=[(\s$])egin\{', r'\\begin{', content)
    
    # 7. Specific fixes for lecture 7 LSTM and RNN equations
    content = re.sub(r'\\left\(\s*1\s*-\s*\\tanh\^2\(([^)]+)\)\s*(?!\\right)', r'\\left(1 - \\tanh^2(\1)\\right) ', content)
    content = re.sub(r'\\left\(\s*1\s*-\s*\\tanh\^2\(\\cdot\)\s*(?!\\right)', r'\\left(1 - \\tanh^2(\\cdot)\\right) ', content)
    content = re.sub(r'\\left\(\s*W\s*\\begin\{pmatrix\}', r'\\left( W \\begin{pmatrix}', content)
    content = re.sub(r'\\begin\{pmatrix\}\s*h\\_\{t-1\}\s*\\\s*x_t\s*\\end\{pmatrix\}\s*\+\s*b\s*(?!\\right)', r'\\begin{pmatrix} h\\_{t-1} \\\\ x_t \\end{pmatrix} + b \\right)', content)
    
    # 8. Specific fixes for lecture 17 RL equations
    content = re.sub(r'\\left\[\s*\\sum\\_\{t=0\}\^\{T\}\s*\\gamma\^t\s*\\mathcal\{R\}\(s_t,\s*a_t\)\s*(?!\\right)', r'\\left[ \\sum\\_{t=0}^{T} \\gamma^t \\mathcal{R}(s_t, a_t) \\right]', content)
    content = re.sub(r'\\left\[\s*\\max\\_\{a\'\}\s*Q\^\*\(s\',\s*a\'\)\s*(?!\\right)', r'\\left[ \\max\\_{a\'} Q^*(s\', a\') \\right]', content)
    content = re.sub(r'\\left\[\s*\\left\(\s*r\s*\+\s*\\gamma\s*\\max\\_\{a\'\}\s*Q\\_\{[^}]+\}\(s\',\s*a\'\)\s*-\s*Q_\\theta\(s,\s*a\)\s*(?!\\right)', r'\\left[ \\left( r + \\gamma \\max\\_{a\'} Q\\_{\\theta^-}(s\', a\') - Q_\\theta(s, a) \\right)^2 \\right]', content)
    content = re.sub(r'\\left\(\s*\\nabla_\\theta\s*\\log\s*\\pi_\\theta\(a_t\s*\|\s*s_t\)\s*(?!\\right)', r'\\left( \\nabla_\\theta \\log \\pi_\\theta(a_t | s_t) \\right)', content)

    if content != orig:
        with open(nf, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed corruptions in: {fname}")

print("All CS231N notes updated!")
