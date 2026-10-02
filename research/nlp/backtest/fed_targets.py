"""Offline only: parse federalreserve.gov open-market pages into dated Fed funds target changes."""
import html, pathlib, re
import pandas as pd

EXT = pathlib.Path(__file__).resolve().parent / "data" / "external" / "fed"


def _num(s: str) -> int:
    m = re.findall(r"\d+", s)
    return int(m[0]) if m else 0  # "75-100" (Dec 2008 range) -> 75 lower bound of the cut


def parse(path: pathlib.Path) -> list[dict]:
    t = path.read_text(encoding="utf-8", errors="replace")
    out = []
    for m in re.finditer(r"<table.*?</table>", t, re.S):
        yrs = re.findall(r">\s*((?:19|20)\d{2})\s*<", t[: m.start()][-3000:])
        if not yrs:
            continue
        for tr in re.findall(r"<tr.*?</tr>", m.group(0), re.S):
            c = [html.unescape(re.sub(r"<.*?>", "", x)).strip() for x in re.findall(r"<t[dh].*?>(.*?)</t[dh]>", tr, re.S)]
            if len(c) == 4 and c[0] != "Date":
                day = re.sub(r"[^A-Za-z0-9 ]", "", c[0])
                out.append({"date": pd.Timestamp(f"{day} {yrs[-1]}"), "change_bp": _num(c[1]) - _num(c[2]),
                            "level_upper": float(re.findall(r"[\d.]+", c[3])[-1])})
    return out


if __name__ == "__main__":
    df = pd.DataFrame(parse(EXT / "openmarket.htm") + parse(EXT / "openmarket_archive.htm"))
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df.to_csv(EXT / "fed_target_changes.csv", index=False)
    print(len(df), df.date.min().date(), df.date.max().date())
    print(df[df.date >= "1999"].groupby(df.date.dt.year).change_bp.agg(["count", "sum"]).T.to_string())
