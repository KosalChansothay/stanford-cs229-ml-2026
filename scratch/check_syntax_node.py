with open(r"d:\File of ML\Machine-Learning-Note\js\cs231n\cs231n-lecture1-charts.js", "r", encoding="utf-8") as f:
    code = f.read()

# Check JS syntax
import subprocess
try:
    p = subprocess.run(["node", "-c", r"d:\File of ML\Machine-Learning-Note\js\cs231n\cs231n-lecture1-charts.js"], capture_output=True, text=True)
    print("Node syntax check return code:", p.returncode)
    if p.stderr:
        print("Node syntax error:", p.stderr)
    else:
        print("Syntax is valid JS!")
except Exception as e:
    print("Could not run node:", e)
