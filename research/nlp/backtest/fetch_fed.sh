#!/usr/bin/env bash
# Offline research data (public domain, federalreserve.gov). Never packaged into the image.
# Usage: bash backtest/fetch_fed.sh   (from research/nlp/)
set -euo pipefail
cd "$(dirname "$0")"
D=data/external/fed
mkdir -p "$D/pages" "$D/statements_raw" "$D/statements_raw2"
UA="Mozilla/5.0 (Macintosh) research"
get() { [ -s "$2" ] || { curl -sS -L --max-time 60 -A "$UA" -o "$2" "$1"; sleep 0.5; }; }

get https://www.federalreserve.gov/monetarypolicy/openmarket.htm "$D/openmarket.htm"
get https://www.federalreserve.gov/monetarypolicy/openmarket_archive.htm "$D/openmarket_archive.htm"
for y in $(seq 2000 2020); do get "https://www.federalreserve.gov/monetarypolicy/fomchistorical$y.htm" "$D/pages/hist$y.htm"; done
get https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm "$D/pages/calendars.htm"

grep -ho 'href="[^"]*"' "$D"/pages/*.htm | sed 's/href="//;s/"$//' \
  | grep -E 'boarddocs/press/(monetary|general)/[0-9]{4}/[0-9]{8}/?(default\.htm)?$|pressreleases/monetary[0-9]{8}a\.htm$' \
  | sort -u > "$D/statement_links.txt"
while read -r l; do
  d=$(echo "$l" | grep -oE '[0-9]{8}' | tail -1); [ "${d:0:4}" -gt 2024 ] && continue
  get "https://www.federalreserve.gov$l" "$D/statements_raw/$d.htm"
done < "$D/statement_links.txt"

# 2006-2010 releases use /newsevents/press/monetary/YYYYMMDD[a|b].htm
grep -ho 'href="[^"]*"' "$D"/pages/hist*.htm | sed 's/href="//;s/"$//' \
  | grep -E 'newsevents/press/monetary/[0-9]{8}[a-z]?\.htm$' | sort -u > "$D/links_2.txt"
while read -r l; do get "https://www.federalreserve.gov$l" "$D/statements_raw2/$(basename "$l")"; done < "$D/links_2.txt"
for f in "$D"/statements_raw2/*.htm; do
  d=$(basename "$f" .htm); base=${d:0:8}
  if grep -q "Federal Open Market Committee" "$f"; then
    if [ ! -s "$D/statements_raw/$base.htm" ] || [ "${d:8:1}" = "a" ]; then cp "$f" "$D/statements_raw/$base.htm"; fi
  fi
done
echo "statements: $(ls "$D/statements_raw" | wc -l)"
