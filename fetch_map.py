"""
Georgia Banks & Credit Unions Map
----------------------------------
Data sources:
  - FDIC BankFind API  (bank branch locations + total assets)
  - NCUA quarterly call report ZIP  (credit union branches + total assets)
  - US Census Bureau batch geocoder  (lat/lon for credit union addresses)

Output: georgia_banks_map.html  (open in any browser, no server needed)

Two highlighted data points per institution:
  1. Total Assets  — size/financial strength
  2. Established Year  — institution longevity
"""

import csv
import io
import json
import math
import urllib.parse
import urllib.request
import zipfile

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get(url, timeout=30, headers=None):
    h = {"User-Agent": "Mozilla/5.0"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Georgia boundary polygon (lat, lon).
# Western border follows the Chattahoochee River — it bends significantly EAST near
# Columbus GA (~32.47°N, -84.97°W) before swinging back northwest.  Using a straight
# -85.61 line would incorrectly include large parts of Alabama.
# Eastern border follows the Chattooga River then the Savannah River into the coast.
_GA_POLY = [
    (34.99, -85.61),  # NW  — GA/TN/AL tripoint
    # North border (~35°N)
    (34.99, -84.29),  # end of TN segment
    (34.98, -83.11),  # begin NC segment
    (34.97, -82.97),  # NE  — GA/NC/SC tripoint (Chattooga headwaters)
    # East border — Chattooga River then Savannah River
    (34.48, -82.70),  # Chattooga / Lake Hartwell
    (34.04, -82.40),  # Savannah River
    (33.96, -82.19),  # Savannah River
    (33.47, -81.96),  # Savannah River
    (32.88, -81.47),  # Savannah River
    (32.53, -81.35),  # near Savannah
    (32.03, -81.10),  # Savannah coast
    (31.55, -81.18),  # Atlantic coast
    (30.78, -81.49),  # Cumberland Island area
    (30.72, -81.46),  # GA/FL/ocean tripoint — St. Marys River mouth (NOT 30.36!)
    # Southeast border follows the St. Marys River westward before the 31st-parallel line
    (30.60, -81.72),  # St. Marys River
    (30.50, -81.93),  # St. Marys River
    (30.44, -82.02),  # St. Marys River near Folkston
    # South border (Florida state line ~30.36°N)
    (30.36, -82.20),
    (30.36, -83.30),
    (30.36, -84.00),
    (30.36, -84.86),  # SW corner — GA/FL/AL tripoint
    # West border — Chattahoochee River (south to north)
    (30.89, -84.87),
    (31.08, -85.08),
    (31.50, -85.03),  # Eufaula / Georgetown area
    (31.80, -85.05),  # Fort Gaines area
    (32.00, -85.00),  # Fort Benning area
    (32.47, -84.97),  # KEY: Columbus / Phenix City — river bends far east here
    (32.80, -85.14),  # river swings back northwest
    (33.20, -85.28),
    (33.54, -85.38),  # near Cedartown
    (34.00, -85.54),
    (34.50, -85.60),
    (34.99, -85.61),  # back to NW corner
]

def _in_georgia(lat, lon):
    """Ray-casting point-in-polygon test against the simplified GA outline."""
    n = len(_GA_POLY)
    inside = False
    px, py = lon, lat
    j = n - 1
    for i in range(n):
        xi, yi = _GA_POLY[i][1], _GA_POLY[i][0]
        xj, yj = _GA_POLY[j][1], _GA_POLY[j][0]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def compute_underbanked(institutions, grid_n=40, top_n=8):
    """
    Lay a grid over Georgia, compute distance from each cell centre to the
    nearest bank/CU, then return the top_n most-isolated spots as
    [{lat, lon, miles}, ...] using greedy non-overlapping selection.
    """
    # Start the grid at 30.72°N — the southernmost inhabited GA latitude
    # (St. Marys River mouth / Folkston area). This prevents cell centres
    # from landing right at the FL/GA border where large radii bleed into Florida.
    GA_LAT = (30.72, 35.00)
    GA_LON = (-85.61, -80.84)

    pts = [(i["lat"], i["lon"]) for i in institutions if i.get("lat") and i.get("lon")]
    if not pts:
        return []

    # Precompute radians + cos(lat) for every institution once
    b_lats = [math.radians(p[0]) for p in pts]
    b_lons = [math.radians(p[1]) for p in pts]
    b_cos  = [math.cos(b) for b in b_lats]

    def nearest(glat, glon):
        gr_lat = math.radians(glat)
        gr_lon = math.radians(glon)
        cg     = math.cos(gr_lat)
        best   = float("inf")
        for i in range(len(b_lats)):
            dlat = b_lats[i] - gr_lat
            dlon = b_lons[i] - gr_lon
            a = math.sin(dlat/2)**2 + cg * b_cos[i] * math.sin(dlon/2)**2
            d = 3958.8 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            if d < best:
                best = d
                if best < 0.1:
                    return best
        return best

    lat_step = (GA_LAT[1] - GA_LAT[0]) / grid_n
    lon_step = (GA_LON[1] - GA_LON[0]) / grid_n

    print(f"  Analysing {grid_n*grid_n} grid points for underbanked areas...")
    grid = []
    for i in range(grid_n):
        for j in range(grid_n):
            glat = GA_LAT[0] + (i + 0.5) * lat_step
            glon = GA_LON[0] + (j + 0.5) * lon_step
            if not _in_georgia(glat, glon):
                continue
            grid.append((nearest(glat, glon), glat, glon))
        if (i + 1) % 8 == 0:
            print(f"    {(i+1)*grid_n}/{grid_n*grid_n} done...")

    grid.sort(reverse=True)

    # Greedy pick: skip any point whose centre falls inside an already-chosen circle
    circles = []
    for d, glat, glon in grid:
        if not any(haversine_miles(glat, glon, c["lat"], c["lon"]) < c["miles"]
                   for c in circles):
            circles.append({"lat": glat, "lon": glon, "miles": round(d, 1)})
            if len(circles) >= top_n:
                break

    print(f"  Underbanked regions found: {len(circles)}  "
          f"(worst: {circles[0]['miles']:.1f} mi from nearest bank)")
    return circles


def clip_circle_to_georgia(center_lat, center_lon, radius_miles, n_angles=120):
    """
    Return a list of [lat, lon] pairs forming a polygon that is the circle
    clipped to Georgia's boundary.  For each bearing we binary-search the
    maximum radius that stays inside Georgia, so the polygon never crosses
    the state line.
    """
    cos_lat = math.cos(math.radians(center_lat))
    R_LAT   = 69.0  # approx miles per degree latitude

    def pt(angle_deg, r):
        ar = math.radians(angle_deg)
        return (
            center_lat + r * math.cos(ar) / R_LAT,
            center_lon + r * math.sin(ar) / (R_LAT * cos_lat),
        )

    poly = []
    for i in range(n_angles):
        angle = i * 360.0 / n_angles
        lat_f, lon_f = pt(angle, radius_miles)
        if _in_georgia(lat_f, lon_f):
            poly.append([round(lat_f, 6), round(lon_f, 6)])
        else:
            # Binary search for the last radius still inside Georgia
            r_lo, r_hi = 0.0, radius_miles
            for _ in range(22):
                r_mid = (r_lo + r_hi) / 2.0
                if _in_georgia(*pt(angle, r_mid)):
                    r_lo = r_mid
                else:
                    r_hi = r_mid
            poly.append([round(pt(angle, r_lo)[0], 6),
                         round(pt(angle, r_lo)[1], 6)])
    return poly


def parse_year(date_str):
    """
    Extract a 4-digit year from FDIC/NCUA date strings.
    Handles: YYYY-MM-DD, MM/DD/YYYY, MM/YYYY, bare YYYY, datetime stamps.
    """
    if not date_str:
        return "N/A"
    for part in date_str.replace("-", "/").split("/"):
        p = part.strip()[:4]
        if len(p) == 4 and p.isdigit() and 1800 <= int(p) <= 2030:
            return p
    return "N/A"


def assets_label(val_thousands):
    """
    Format asset value (in $thousands) as human-readable string.
    v is in $thousands, so:
      v >= 1,000,000,000 → trillions  (÷ 1B)
      v >= 1,000,000     → billions   (÷ 1M)
      v >= 1,000         → millions   (÷ 1K)
      else               → thousands  (raw)
    """
    if val_thousands is None:
        return "N/A"
    try:
        v = float(val_thousands)
    except (TypeError, ValueError):
        return "N/A"
    if v >= 1_000_000_000:
        return f"${v/1_000_000_000:.2f}T"
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}B"
    if v >= 1_000:
        return f"${v/1_000:.1f}M"
    return f"${v:.0f}K"


def classify_bank_type(assets_thousands):
    """National (>$100B), Regional ($1B-$100B), or Community (<$1B) bank."""
    try:
        a = float(assets_thousands)
    except (TypeError, ValueError):
        return "Community Bank"
    if a >= 100_000_000:   # $100B+
        return "National Bank"
    if a >= 1_000_000:     # $1B+
        return "Regional Bank"
    return "Community Bank"


def classify_ga_region(lat, lon):
    """Map a Georgia lat/lon to a general geographic region."""
    if 33.45 <= lat <= 34.25 and -85.0 <= lon <= -83.6:
        return "Metro Atlanta"
    if lon >= -82.1:
        return "Coastal Georgia"
    if lat <= 31.5:
        return "Rural South Georgia"
    if lat >= 34.25:
        return "North Georgia"
    return "Central Georgia"


def classify_cu_membership(fom_value):
    """
    Map NCUA TOM_CODE (Type of Membership) to a readable category.

    Observed GA values and their meanings (per NCUA chartering regs):
      00 → open community charter (no specific common bond)
      01 → civic/neighborhood associational
      02 → fraternal associational
      03 → agricultural occupational
      04, 34 → education/teachers occupational
      12, 20, 21, 23 → government/public-sector occupational
      15 → single-employer occupational
      36 → municipal/county employees occupational
      51–54 → specific-employer occupational (GP = Georgia Power, etc.)
      98 → open / associational (expanded)
      99 → multiple common bond (grew beyond original single group)
    """
    if not fom_value:
        return "N/A"
    v = fom_value.strip()
    _COMMUNITY    = {"00"}
    _OCCUPATIONAL = {"03", "04", "12", "15", "20", "21", "23", "34",
                     "36", "51", "52", "53", "54"}
    _ASSOC        = {"01", "02"}
    _MULTI        = {"06", "98", "99"}
    if v in _COMMUNITY:
        return "Community-Based"
    if v in _OCCUPATIONAL:
        return "Occupational"
    if v in _ASSOC:
        return "Associational"
    if v in _MULTI:
        return "Multiple Common Bond"
    return "Other"


# ---------------------------------------------------------------------------
# FDIC: Georgia bank branches
# ---------------------------------------------------------------------------

FDIC_BASE = "https://banks.data.fdic.gov/api"


def fetch_fdic_institutions():
    """
    Map cert -> {assets_thousands, est_year} for ALL US institutions.

    No state filter here — national banks (Chase, BofA, Wells Fargo, etc.)
    are headquartered outside Georgia but have GA branches.  Filtering to
    only GA-HQ'd institutions would leave those branches with no asset data.
    The full US dataset is ~4,500 active institutions and fits in one page.
    """
    lookup = {}
    limit, offset = 10000, 0
    print("  Fetching FDIC institution financial data (all US)...")
    while True:
        params = urllib.parse.urlencode({
            "fields": "CERT,ASSET,ESTYMD",
            "limit": limit,
            "offset": offset,
            "output": "json",
        })
        data = json.loads(get(f"{FDIC_BASE}/institutions?{params}"))
        records = data.get("data", [])
        if not records:
            break
        for rec in records:
            d = rec.get("data", rec)
            cert = str(d.get("CERT", ""))
            lookup[cert] = {
                "assets_thousands": d.get("ASSET"),
                "est_year": parse_year(d.get("ESTYMD", "")),
            }
        offset += limit
        if len(records) < limit:
            break
    print(f"  Institution records: {len(lookup)}")
    return lookup


def fetch_fdic_branches(inst_lookup):
    """Return list of dicts for all active Georgia bank branch locations."""
    branches = []
    limit, offset = 10000, 0
    print("  Fetching FDIC branch locations...")
    while True:
        params = urllib.parse.urlencode({
            "filters": "STALP:GA",
            "fields": "NAME,ADDRESS,CITY,ZIP,LATITUDE,LONGITUDE,CERT,ESTYMD",
            "limit": limit,
            "offset": offset,
            "output": "json",
        })
        data = json.loads(get(f"{FDIC_BASE}/locations?{params}"))
        records = data.get("data", [])
        if not records:
            break
        for rec in records:
            d = rec.get("data", rec)
            try:
                lat = float(d["LATITUDE"])
                lon = float(d["LONGITUDE"])
            except (KeyError, TypeError, ValueError):
                continue
            cert = str(d.get("CERT", ""))
            inst = inst_lookup.get(cert, {})
            est_year = parse_year(d.get("ESTYMD", "")) or inst.get("est_year", "N/A")
            branches.append({
                "name":    d.get("NAME", "Unknown"),
                "address": f"{d.get('ADDRESS','')}, {d.get('CITY','')}, GA {d.get('ZIP','')}".strip(", "),
                "lat": lat,
                "lon": lon,
                "assets_thousands": inst.get("assets_thousands"),
                "est_year": est_year,
                "type": "Bank",
                "subtype": classify_bank_type(inst.get("assets_thousands")),
                "region":  classify_ga_region(lat, lon),
            })
        print(f"    {len(branches)} branches so far...")
        offset += limit
        if len(records) < limit:
            break
    print(f"  Total FDIC branches: {len(branches)}")
    return branches


# ---------------------------------------------------------------------------
# NCUA: Georgia credit union branches  (no lat/lon — need geocoding)
# ---------------------------------------------------------------------------

NCUA_ZIP_URL = "https://www.ncua.gov/files/publications/analysis/call-report-data-2025-09.zip"


def fetch_ncua_data():
    """
    Download the NCUA quarterly call report ZIP and extract:
      - GA branch addresses (Credit Union Branch Information.txt)
      - Total assets per CU_NUMBER (FS220.txt  ACCT_010 = total assets, in $thousands)
      - Established year per CU_NUMBER (FOICU.txt  YEAR_OPENED)
    Returns list of dicts with name, address, cu_number, assets_thousands, est_year.
    """
    print("  Downloading NCUA quarterly call report ZIP (~8 MB)...")
    raw = get(NCUA_ZIP_URL, timeout=120)
    print(f"  Downloaded {len(raw)/1e6:.1f} MB")

    zf = zipfile.ZipFile(io.BytesIO(raw))

    # --- Total assets: FS220.txt, field ACCT_010 (dollars, not thousands) ---
    fs220_raw = zf.read("FS220.txt").decode("latin-1")
    assets_by_cu = {}
    for row in csv.DictReader(fs220_raw.splitlines()):
        cu = row.get("CU_NUMBER", "").strip()
        val = row.get("ACCT_010", "")
        try:
            # NCUA reports dollars; convert to thousands for consistency with FDIC
            assets_by_cu[cu] = float(val) / 1000
        except (ValueError, TypeError):
            pass

    # --- Established year + membership focus: FOICU.txt ---
    foicu_raw = zf.read("FOICU.txt").decode("latin-1")
    est_by_cu = {}
    fom_by_cu = {}
    foicu_rows = list(csv.DictReader(foicu_raw.splitlines()))
    # NCUA uses different field names across data versions — try each in order
    _FOM_CANDIDATES = [
        "TOM_CODE",                                         # quarterly call report (confirmed 2025-09)
        "FIELD_OF_MEMBERSHIP_TYPE", "FIELD_OF_MEMBERSHIP_CODE",
        "MEMBERSHIP_TYPE", "FOM_TYPE", "TYPE_GROUP",
    ]
    _fom_field = next(
        (f for f in _FOM_CANDIDATES if foicu_rows and f in foicu_rows[0]),
        None
    )
    if _fom_field:
        print(f"  NCUA membership field found: {_fom_field}")
    else:
        # Show available fields so future versions can be mapped
        sample_keys = list(foicu_rows[0].keys()) if foicu_rows else []
        print(f"  NCUA membership field not found; available: {sample_keys}")
    for row in foicu_rows:
        cu = row.get("CU_NUMBER", "").strip()
        yr = row.get("YEAR_OPENED", "").strip()
        if yr:
            est_by_cu[cu] = yr
        if _fom_field:
            fom_by_cu[cu] = classify_cu_membership(row.get(_fom_field, ""))

    # --- Branch addresses ---
    branch_raw = zf.read("Credit Union Branch Information.txt").decode("latin-1")
    ga_branches = []
    for row in csv.DictReader(branch_raw.splitlines()):
        if row.get("PhysicalAddressStateCode", "").strip() != "GA":
            continue
        cu = row.get("CU_NUMBER", "").strip()
        name = row.get("CU_NAME", "Unknown").strip()
        addr = row.get("PhysicalAddressLine1", "").strip()
        city = row.get("PhysicalAddressCity", "").strip()
        zipcode = row.get("PhysicalAddressPostalCode", "").strip()[:5]
        if not addr or not city:
            continue
        ga_branches.append({
            "cu_number": cu,
            "name": name,
            "street": addr,
            "city": city,
            "state": "GA",
            "zip": zipcode,
            "address": f"{addr}, {city}, GA {zipcode}",
            "assets_thousands": assets_by_cu.get(cu),
            "est_year": est_by_cu.get(cu, "N/A"),
            "subtype": fom_by_cu.get(cu, "N/A"),
        })

    print(f"  GA credit union branches to geocode: {len(ga_branches)}")
    return ga_branches


# ---------------------------------------------------------------------------
# Geocoding: US Census Bureau batch geocoder (free, no key required)
# ---------------------------------------------------------------------------

CENSUS_GEOCODE_URL = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
BATCH_SIZE = 1000


def geocode_batch(rows):
    """
    Geocode a batch of address dicts using the Census batch geocoder.
    Input: list of dicts with keys street, city, state, zip, plus any extra keys.
    Returns the same list with lat/lon added where matched.
    """
    # Build CSV payload
    csv_lines = []
    for i, r in enumerate(rows):
        csv_lines.append(f'{i},"{r["street"]}","{r["city"]}","{r["state"]}","{r["zip"]}"')
    csv_payload = "\n".join(csv_lines).encode("utf-8")

    boundary = "----CensusBatch"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="addressFile"; filename="addr.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + csv_payload + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="benchmark"\r\n\r\n'
        f"Public_AR_Current\r\n"
        f"--{boundary}--\r\n"
    ).encode()

    req = urllib.request.Request(
        CENSUS_GEOCODE_URL,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "Mozilla/5.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result_csv = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    Geocoding batch failed: {e}")
        return rows

    # Parse results — use csv.reader so quoted commas in address fields don't break splits
    import csv as _csv
    coord_map = {}
    for row in _csv.reader(result_csv.splitlines()):
        # Format: ID, input_addr, Match/No_Match, MatchType, matched_addr, "lon,lat", TigerID, Side
        if len(row) >= 6 and row[2].strip().lower() == "match":
            idx = row[0].strip()
            coords = row[5].strip()
            if "," in coords:
                try:
                    lon, lat = coords.split(",")
                    coord_map[idx] = (float(lat.strip()), float(lon.strip()))
                except ValueError:
                    pass

    for i, r in enumerate(rows):
        if str(i) in coord_map:
            r["lat"], r["lon"] = coord_map[str(i)]

    return rows


def geocode_all(branches):
    """Geocode all branches in batches."""
    total = len(branches)
    matched = 0
    print(f"  Geocoding {total} addresses via Census Bureau batch API...")
    for start in range(0, total, BATCH_SIZE):
        batch = branches[start: start + BATCH_SIZE]
        geocode_batch(batch)
        batch_matched = sum(1 for r in batch if "lat" in r)
        matched += batch_matched
        print(f"    Batch {start//BATCH_SIZE + 1}: {batch_matched}/{len(batch)} matched  (total {matched}/{start+len(batch)})")
    print(f"  Geocoding complete: {matched}/{total} matched")
    return branches


# ---------------------------------------------------------------------------
# Map generation (self-contained Leaflet HTML)
# ---------------------------------------------------------------------------

def write_log(institutions, output_file="georgia_institutions_log.csv"):
    """
    Write every institution's full data set to a CSV log.
    One row per branch/location.  All asset figures are in raw $thousands
    plus a human-readable column for easy reading.
    """
    fieldnames = [
        "name", "type", "subtype", "region",
        "address", "city", "state", "zip",
        "lat", "lon",
        "total_assets_thousands", "total_assets_formatted",
        "established_year",
    ]
    rows = []
    for inst in institutions:
        # Parse city / zip back out of the combined address string where possible
        addr_full = inst.get("address", "")
        # Address format: "STREET, CITY, GA ZIP"
        parts = [p.strip() for p in addr_full.split(",")]
        city  = parts[1] if len(parts) >= 2 else ""
        zip_  = parts[-1].replace("GA", "").strip() if len(parts) >= 3 else ""

        assets_k = inst.get("assets_thousands")
        rows.append({
            "name":                    inst.get("name", ""),
            "type":                    inst.get("type", ""),
            "subtype":                 inst.get("subtype", "N/A"),
            "region":                  inst.get("region", "N/A"),
            "address":                 addr_full,
            "city":                    city,
            "state":                   "GA",
            "zip":                     zip_,
            "lat":                     inst.get("lat", ""),
            "lon":                     inst.get("lon", ""),
            "total_assets_thousands":  f"{int(assets_k):,}" if assets_k is not None else "N/A",
            "total_assets_formatted":  assets_label(assets_k),
            "established_year":        inst.get("est_year", "N/A"),
        })

    # Sort: banks first, then credit unions; alphabetically within each group
    rows.sort(key=lambda r: (r["type"] != "Bank", r["name"].lower()))

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    banks = sum(1 for r in rows if r["type"] == "Bank")
    cus   = len(rows) - banks
    print(f"Log saved -> {output_file}  ({banks} bank branches, {cus} credit union branches)")


def build_map(institutions, output_file="georgia_banks_map.html"):
    if not institutions:
        print("No institutions to map.")
        return

    # Filter to those with coordinates
    mapped = [i for i in institutions if i.get("lat") and i.get("lon")]
    print(f"\nTotal with coordinates: {len(mapped)} / {len(institutions)}")

    lats = [i["lat"] for i in mapped]
    lons = [i["lon"] for i in mapped]
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2

    features = []
    for inst in mapped:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [inst["lon"], inst["lat"]]},
            "properties": {
                "name":     inst["name"],
                "address":  inst.get("address", ""),
                "assets":   assets_label(inst.get("assets_thousands")),
                "est_year": inst.get("est_year", "N/A"),
                "type":     inst["type"],
                "subtype":  inst.get("subtype", "N/A"),
                "region":   inst.get("region", "N/A"),
            },
        })

    geojson    = json.dumps({"type": "FeatureCollection", "features": features})
    underbanked = compute_underbanked(mapped, top_n=3)
    print("  Clipping underbanked circles to Georgia boundary...")
    for c in underbanked:
        c["poly"] = clip_circle_to_georgia(c["lat"], c["lon"], c["miles"])
    ub_json = json.dumps(underbanked)
    bank_locs  = [i for i in mapped if i["type"] == "Bank"]
    cu_locs    = [i for i in mapped if i["type"] == "Credit Union"]
    bank_count = len(bank_locs)
    cu_count   = len(cu_locs)
    bank_inst  = len(set(i["name"] for i in bank_locs))
    cu_inst    = len(set(i["name"] for i in cu_locs))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Georgia Banks &amp; Credit Unions</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family: Arial, sans-serif; }}
    #map {{ width:100vw; height:100vh; }}
    #controls {{
      position:absolute; top:10px; right:10px; z-index:1000;
      background:white; padding:12px 16px; border-radius:8px;
      box-shadow:0 2px 10px rgba(0,0,0,.3); min-width:210px;
    }}
    #controls h3 {{ margin-bottom:8px; font-size:14px; color:#222; }}
    .row {{ display:flex; align-items:center; gap:8px; margin:4px 0; font-size:13px; cursor:pointer; }}
    .dot {{ width:12px; height:12px; border-radius:50%; flex-shrink:0; }}
    .bank-dot {{ background:#2563eb; }}
    .cu-dot   {{ background:#16a34a; }}
    #stats {{ margin-top:10px; font-size:12px; color:#555; border-top:1px solid #eee; padding-top:8px; line-height:1.8; }}
    #stats table {{ border-collapse:collapse; width:100%; }}
    #stats th {{ text-align:left; font-weight:600; color:#333; padding-bottom:3px; font-size:11px; text-transform:uppercase; letter-spacing:.4px; }}
    #stats td {{ padding:1px 4px 1px 0; }}
    #stats td.num {{ text-align:right; font-weight:600; color:#111; padding-left:8px; }}
  </style>
</head>
<body>
<div id="map"></div>
<div id="controls">
  <h3>Georgia Financial Institutions</h3>
  <label class="row">
    <input type="checkbox" id="chk-bank" checked>
    <span class="dot bank-dot"></span> Banks
  </label>
  <label class="row">
    <input type="checkbox" id="chk-cu" checked>
    <span class="dot cu-dot"></span> Credit Unions
  </label>
  <label class="row">
    <input type="checkbox" id="chk-ub" checked>
    <span class="dot" style="background:#dc2626;opacity:.7;border-radius:2px;"></span> Underbanked Areas
  </label>
  <div id="stats">
    <table>
      <tr>
        <th></th><th class="num">Institutions</th><th class="num">Locations</th>
      </tr>
      <tr>
        <td>Banks</td>
        <td class="num">{bank_inst:,}</td>
        <td class="num">{bank_count:,}</td>
      </tr>
      <tr>
        <td>Credit Unions</td>
        <td class="num">{cu_inst:,}</td>
        <td class="num">{cu_count:,}</td>
      </tr>
      <tr style="border-top:1px solid #ddd;">
        <td><b>Total</b></td>
        <td class="num"><b>{bank_inst + cu_inst:,}</b></td>
        <td class="num"><b>{bank_count + cu_count:,}</b></td>
      </tr>
    </table>
  </div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const map = L.map('map').setView([{center_lat:.4f}, {center_lon:.4f}], 7);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  attribution: '&copy; OpenStreetMap contributors', maxZoom: 19
}}).addTo(map);

const data = {geojson};

function icon(color) {{
  return L.divIcon({{
    className: '',
    html: `<svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="${{color}}" stroke="white" stroke-width="1"/></svg>`,
    iconSize: [10,10], iconAnchor: [5,5]
  }});
}}
const bankIcon = icon('#2563eb');
const cuIcon   = icon('#16a34a');

const bankLayer = L.layerGroup();
const cuLayer   = L.layerGroup();

data.features.forEach(f => {{
  const p = f.properties;
  const [lon,lat] = f.geometry.coordinates;
  const mk = L.marker([lat,lon], {{icon: p.type==='Bank' ? bankIcon : cuIcon}});
  const typeLabel = p.type === 'Bank' ? 'Bank Type' : 'Membership';
  mk.bindPopup(`
    <b>${{p.name}}</b><br>
    <span style="color:#666;font-size:12px">${{p.type}}</span><br><br>
    <b>${{typeLabel}}:</b> ${{p.subtype}}<br>
    <b>Region:</b> ${{p.region}}<br>
    <b>Address:</b> ${{p.address}}<br>
    <b>Total Assets:</b> ${{p.assets}}<br>
    <b>Established:</b> ${{p.est_year}}
  `);
  (p.type==='Bank' ? bankLayer : cuLayer).addLayer(mk);
}});

bankLayer.addTo(map);
cuLayer.addTo(map);

// Underbanked areas — polygons pre-clipped to Georgia's boundary in Python,
// so they never extend into neighbouring states.
const ubData = {ub_json};
const ubLayer = L.layerGroup();
ubData.forEach(c => {{
  L.polygon(c.poly, {{
    color: '#dc2626',
    fillColor: '#dc2626',
    fillOpacity: 0.18,
    weight: 2,
    dashArray: '6 4',
  }})
  .bindPopup(`<b style="color:#dc2626">Underbanked Area</b><br>Nearest bank or credit union is approximately <b>${{c.miles}} miles</b> away.`)
  .addTo(ubLayer);
}});
ubLayer.addTo(map);

document.getElementById('chk-bank').addEventListener('change', e =>
  e.target.checked ? bankLayer.addTo(map) : map.removeLayer(bankLayer));
document.getElementById('chk-cu').addEventListener('change', e =>
  e.target.checked ? cuLayer.addTo(map) : map.removeLayer(cuLayer));
document.getElementById('chk-ub').addEventListener('change', e =>
  e.target.checked ? ubLayer.addTo(map) : map.removeLayer(ubLayer));
</script>
</body>
</html>
"""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Map saved -> {output_file}")
    print("Open it in any browser (no server needed).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Fetching FDIC bank data ===")
    inst_lookup = fetch_fdic_institutions()
    banks = fetch_fdic_branches(inst_lookup)

    print("\n=== Fetching NCUA credit union data ===")
    cu_branches = fetch_ncua_data()
    cu_branches = geocode_all(cu_branches)
    for b in cu_branches:
        b["type"] = "Credit Union"
        if b.get("lat") and b.get("lon"):
            b["region"] = classify_ga_region(b["lat"], b["lon"])
        else:
            b["region"] = "N/A"

    all_institutions = banks + cu_branches
    print(f"\n=== Writing institution log ===")
    write_log(all_institutions)
    print(f"\n=== Building map ({len(all_institutions)} total) ===")
    build_map(all_institutions)
