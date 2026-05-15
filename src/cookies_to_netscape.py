#!/usr/bin/env python3
"""Convert a raw browser 'Cookie:' header into a Netscape cookies.txt for yt-dlp.

Usage: cookies_to_netscape.py <header_file> <out_cookies.txt>
Prints only a count + presence of key auth cookies (never the values).
"""
import sys
import time

header_file, out_file = sys.argv[1], sys.argv[2]
raw = open(header_file, encoding="utf-8").read().strip()
if raw.lower().startswith("cookie:"):
    raw = raw.split(":", 1)[1].strip()

cookies = []
for part in raw.split("; "):
    part = part.strip()
    if not part or "=" not in part:
        continue
    name, value = part.split("=", 1)  # split first '=' only (PREF value contains '=')
    cookies.append((name.strip(), value.strip()))

exp = int(time.time()) + 365 * 24 * 3600
with open(out_file, "w", encoding="utf-8") as f:
    f.write("# Netscape HTTP Cookie File\n")
    f.write("# Generated for yt-dlp from a browser Cookie header.\n")
    for name, value in cookies:
        # domain, include_subdomains, path, secure, expiry, name, value
        f.write(f".youtube.com\tTRUE\t/\tTRUE\t{exp}\t{name}\t{value}\n")

key = {"SID", "SAPISID", "__Secure-1PSID", "__Secure-1PSIDTS", "LOGIN_INFO"}
present = sorted(k for k, _ in cookies if k in key)
print(f"wrote {len(cookies)} cookies -> {out_file}")
print(f"key auth cookies present: {present}")
