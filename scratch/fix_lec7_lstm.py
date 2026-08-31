import io

p = r"d:\File of ML\Machine-Learning-Note\notes\cs231n\cs231n-lecture-7-notes.md"
with open(p, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

line = lines[50]
print("BEFORE repr:", repr(line))

# Fix 1: pmatrix i f o g single-backslash separators -> double backslash
old1 = "\\begin{pmatrix} i \\ f \\ o \\ g \\end{pmatrix}"
new1 = "\\begin{pmatrix} i \\\\ f \\\\ o \\\\ g \\end{pmatrix}"
assert old1 in line, "pmatrix i f o g not found"
line = line.replace(old1, new1)

# Fix 2: pmatrix sigma row separators -> double backslash
old2 = "\\begin{pmatrix} \\sigma \\ \\sigma \\ \\sigma \\ \\tanh \\end{pmatrix}"
new2 = "\\begin{pmatrix} \\sigma \\\\ \\sigma \\\\ \\sigma \\\\ \\tanh \\end{pmatrix}"
assert old2 in line, "pmatrix sigma not found"
line = line.replace(old2, new2)

# Fix 3: remove extra \right). The segment is  + b \right) \right) \
# trailing backslash (line continuation) must be preserved as single backslash
bs = chr(92)  # single backslash
old3 = " + b " + bs + "right) " + bs + "right) " + bs
new3 = " + b " + bs + "right) " + bs
assert old3 in line, "extra right) not found"
line = line.replace(old3, new3)

lines[50] = line
with open(p, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("AFTER  repr:", repr(line))
