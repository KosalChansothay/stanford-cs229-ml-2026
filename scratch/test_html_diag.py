import urllib.request
import re

html_url = "http://localhost:8000/courses/cs231n/lecture-1.html"
with urllib.request.urlopen(html_url) as r:
    html = r.read().decode('utf-8')

print("HTML length:", len(html))
print("data-markdown:", re.findall(r'data-markdown="([^"]+)"', html))
print("scripts:", re.findall(r'<script[^>]*src="([^"]+)"', html))
print("markdown container:", re.findall(r'<div[^>]*class="markdown-body"[^>]*>', html))
