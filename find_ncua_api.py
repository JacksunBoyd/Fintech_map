import urllib.request, re

req = urllib.request.Request(
    'https://mapping.ncua.gov/main.cf06f2f0e760421d.js',
    headers={'User-Agent': 'Mozilla/5.0'}
)
with urllib.request.urlopen(req, timeout=30) as r:
    src = r.read().decode('utf-8', errors='ignore')

# Look for relative API paths
api_paths = re.findall(r'"/[a-zA-Z][a-zA-Z0-9_/-]{3,60}"', src)
uniq = sorted(set(api_paths))
keywords = ['credit','union','locate','search','institution','member','cu','map','state','branch']
likely = [p for p in uniq if any(x in p.lower() for x in keywords)]
print('Likely API paths:')
for p in likely[:40]:
    print(' ', p)

# Also look for environment/config blocks with base URLs
env_matches = re.findall(r'apiUrl["\s:]+["\x27]([^"\x27]{5,80})["\x27]', src)
print('\nAPI URL config:', env_matches[:10])
