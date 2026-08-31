import re

for n in [12, 13, 14]:
    c = open(r"d:\File of ML\Machine-Learning-Note\js\cs231n\cs231n-lecture%d-charts.js" % n, encoding="utf-8").read()
    print("=== Lecture %d ===" % n)
    for m in re.finditer(r"method:\s*'(update|restyle|relayout)'", c):
        print("  method:", m.group(1))
    for i, l in enumerate(c.splitlines(), 1):
        s = l.strip()
        if "sliders" in s or "Plotly.react" in s or "Plotly.newPlot" in s or "steps" in s:
            print("  L%d: %s" % (i, s[:110]))
