import urllib.request, re

req = urllib.request.Request(
    'https://mapping.ncua.gov/main.cf06f2f0e760421d.js',
    headers={'User-Agent': 'Mozilla/5.0'}
)
with urllib.request.urlopen(req, timeout=30) as r:
    src = r.read().decode('utf-8', errors='ignore')

# Look for environment/config objects with API base URLs
# Angular typically has something like {production:true,apiUrl:"https://..."}
env_patterns = [
    r'apiUrl\s*[:=]\s*"([^"]{5,100})"',
    r'baseUrl\s*[:=]\s*"([^"]{5,100})"',
    r'apiBase\s*[:=]\s*"([^"]{5,100})"',
    r'serviceUrl\s*[:=]\s*"([^"]{5,100})"',
    r'endpoint\s*[:=]\s*"(https?://[^"]{5,100})"',
    r'"(https?://[^"]{5,60}ncua[^"]{0,60})"',
]
for pat in env_patterns:
    matches = re.findall(pat, src)
    if matches:
        print(f'Pattern {pat[:40]}:')
        for m in list(set(matches))[:5]:
            print(f'  {m}')
