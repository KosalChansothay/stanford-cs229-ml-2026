import io

p = r"d:\File of ML\Machine-Learning-Note\notes\cs231n\cs231n-lecture-17-notes.md"
with open(p, "r", encoding="utf-8") as f:
    content = f.read()

bs = chr(92)  # single backslash

# Fix 1 (line 32): remove extra \right] after the expected \right]
#   ... \right] \right]  ->  ... \right]
old1 = bs + "right] " + bs + "right]"
new1 = bs + "right]"
# only fix the FIRST occurrence (the RL objective), but this pattern also matches line 37.
# We'll replace all occurrences of "\right] \right]" -> "\right]" (the stray doubles)
count1 = content.count(old1)
content = content.replace(old1, new1)
print("Fixed \\right] \\right] -> \\right]  (count):", count1)

# Fix 3 (line 39): the garbled trailing \right)^2 \right]
#   \right)^2 \right]  ->  \right]
old2 = bs + "right)^2 " + bs + "right]"
new2 = bs + "right]"
count2 = content.count(old2)
content = content.replace(old2, new2)
print("Fixed \\right)^2 \\right] -> \\right]  (count):", count2)

with open(p, "w", encoding="utf-8") as f:
    f.write(content)

# Verify balance of \left[ and \right] and \left( \right) per line
print("\n--- Verify delimiter balance on affected lines ---")
lines = content.split("\n")
for i, line in enumerate(lines, 1):
    if i in (32, 37, 39):
        lf = line.count(bs + "left[")
        rf = line.count(bs + "right]")
        lp = line.count(bs + "left(")
        rp = line.count(bs + "right)")
        print(f"L{i}: left[={lf} right]={rf} left(={lp} right)={rp}")
