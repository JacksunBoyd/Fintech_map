import urllib.request, re

req = urllib.request.Request(
    'https://www.ncua.gov/analysis/credit-union-corporate-call-report-data/call-report-data-for-download',
    headers={'User-Agent': 'Mozilla/5.0'}
)
with urllib.request.urlopen(req, timeout=15) as r:
    html = r.read().decode('utf-8', errors='ignore')

# Find all file links
links = re.findall(r'href="(/files/[^"]+\.(zip|csv|xlsx|xls))"', html)
print('File links:')
for link, ext in links[:20]:
    print(f'  {link}')

# Also look for any data download API
api_links = re.findall(r'href="(https?://[^"]+)"', html)
print('\nExternal links:')
for link in api_links[:10]:
    print(f'  {link}')
