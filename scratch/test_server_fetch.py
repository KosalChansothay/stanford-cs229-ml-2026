import urllib.request

urls = [
    "http://localhost:8000/courses/cs231n/lecture-1.html",
    "http://localhost:8000/notes/cs231n/cs231n-lecture-1-notes.md",
    "http://localhost:8000/js/script.js",
    "http://localhost:8000/js/cs231n/cs231n-lecture1-charts.js"
]

print("=== TESTING LOCAL SERVER HTTP RESPONSES ===")

for url in urls:
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            content = resp.read()
            print(f"[HTTP {resp.status}] {url} -> {len(content)} bytes")
            if "script.js" in url:
                text = content.decode('utf-8')
                if "markdownContent.innerHTML = marked.parse(markdown);" in text:
                    print("  -> script.js contains marked.parse(markdown) [OK!]")
                else:
                    print("  -> script.js DOES NOT CONTAIN marked.parse(markdown) [OUTDATED!]")
    except Exception as e:
        print(f"[ERROR] {url} -> {e}")
