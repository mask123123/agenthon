"""Offline only: clean FOMC statement HTML (federalreserve.gov, public domain) into dated plain text."""
import html, json, pathlib, re

EXT = pathlib.Path(__file__).resolve().parent / "data" / "external" / "fed"
START = re.compile(r"(The Federal Open Market Committee|The Federal Reserve|Information received|Recent indicators|"
                   r"Recent data|Economic activity|The Board of Governors|Inflation remains|Although|In a joint statement)", re.I)
END = re.compile(r"(Voting for the FOMC monetary policy action|Voting for this action|Voting for the action|"
                 r"Voting against|Implementation Note|For media inquiries|Last Update:|"
                 r"Board of Governors of the Federal Reserve System\s+20th Street)", re.I)


def clean(raw: str) -> str:
    body_marker = 'class="col-xs-12 col-sm-8 col-md-8"'
    new_format = body_marker in raw  # 2006+ pages: the statement is the main column; skip navigation/title
    if new_format:
        raw = raw[raw.index(body_marker):]
    t = re.sub(r"(?is)<(script|style|nav|header|footer).*?</\1>", " ", raw)
    t = re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", t)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    t = re.sub(r"[ \t\xa0]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)
    m = None if new_format else START.search(t)
    body = t[m.start():] if m else t
    e = END.search(body)
    body = re.sub(r'^\s*class="[^"]*">\s*', "", body)
    return (body[: e.start()] if e else body[:6000]).strip()


if __name__ == "__main__":
    out = {}
    for f in sorted((EXT / "statements_raw").glob("*.htm")):
        d = f.stem
        text = clean(f.read_text(encoding="utf-8", errors="replace"))
        if "Committee" in text:  # drop non-FOMC releases (e.g. facility announcements)
            out[f"{d[:4]}-{d[4:6]}-{d[6:]}"] = text
    (EXT / "fomc_statements.json").write_text(json.dumps(out, indent=1))
    lens = sorted(len(v) for v in out.values())
    print(len(out), "statements; chars min/med/max", lens[0], lens[len(lens) // 2], lens[-1])
    for k in ("2001-01-03", "2008-12-16", "2015-12-16", "2022-06-15"):
        print("=====", k, "\n", out.get(k, "MISSING")[:500])
