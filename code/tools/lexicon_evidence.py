#!/usr/bin/env python3
"""Verify proposed dialect lexicon items and collect per-item evidence.

For each item (form, SAE equivalent, variety) it records:
  - Wiktionary: does the page have a sense carrying one of the variety's regional labels?
    (label found, gloss text, URL) -- en.wiktionary.org API, cached.
  - Corpus attestation: whole-word, case-insensitive counts in the local authentic corpora
    under exp/data_raw/ (only those relevant to the variety).
Usage:
  lexicon_evidence.py <variety> items.tsv > evidence.json
  items.tsv: form<TAB>sae<TAB>type[<TAB>wiktionary_title]   (# comments allowed)
Varieties/labels are configured in LABELS below. Network: Wiktionary only (throttled 0.35 s/call).
"""
import json, re, sys, time, urllib.parse, urllib.request
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "exp" / "data_raw"
CACHE = ROOT / "notes" / "lexicons" / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)
UA = "P3-lexicon-research/0.1 (academic research; contact via project)"

# Wiktionary label aliases per variety (lowercase, as they appear in {{lb|en|...}} / {{lb|pcm|...}}).
LABELS = {
    "aae": ["aae", "aave", "african-american vernacular", "african american vernacular english", "african-american", "aave slang", "ebonics", "black english"],
    "sge": ["singapore", "singlish", "colloquial singapore english", "malaysia", "malaysian english", "singapore english"],
    "ngp": ["nigeria", "nigerian pidgin", "west africa", "nigerian"],
    "sco": ["scotland", "scottish", "scots", "scottish english", "northern england", "northern ireland", "ulster"],
    "app": ["appalachia", "appalachian", "southern us", "southern united states", "us south", "southern us dialect", "rural us", "southern american english", "midland us"],
    "ind": ["india", "indian english", "south asia", "pakistan"],
    "ire": ["ireland", "irish", "hiberno-english", "irish english", "northern ireland"],
    "bre": ["uk", "british", "britain", "england", "commonwealth", "british english"],
    "aus": ["australia", "australian", "new zealand", "australian english", "nz"],
    "inf": ["informal", "colloquial", "slang", "internet slang", "text messaging", "internet", "casual"],
    "for": ["formal"],
}
LANG_SECTION = {"ngp": ["Nigerian Pidgin", "English"]}  # pcm entries live in their own section

# Corpora relevant per variety (paths relative to exp/data_raw); None = none local.
CORPORA = {
    "aae": ["groenwold2020/EMNLP-AAVE-files/aave_samples.txt", "coraal/txt", "twitteraae_ud/TwitterAAE-UD-v1/aa250_gold.conllu"],
    "sae_ref": ["groenwold2020/EMNLP-AAVE-files/sae_samples.txt"],
    "sge": ["cosem/CoSEM_v5@SG", "nus_sms/smsCorpus_en_2015.03.09_all.xml@Singapore|SG"],
    "ngp": ["ud_naija_nsc", "mafand_en_pcm"],
    "ind": ["nus_sms/smsCorpus_en_2015.03.09_all.xml@India|IN"],
    "aae_tw": ["groenwold2020/EMNLP-AAVE-files/aave_samples.txt", "twitteraae_ud/TwitterAAE-UD-v1/aa250_gold.conllu"],
    "sae_tw": ["groenwold2020/EMNLP-AAVE-files/sae_samples.txt", "twitteraae_ud/TwitterAAE-UD-v1/wh250_gold.conllu"],
    "coraal": ["coraal/txt"],
    "mafand_en": ["mafand_en_pcm@EN"],
    "nus_us": ["nus_sms/smsCorpus_en_2015.03.09_all.xml@United States|US|USA"],
    "nge": ["ice_nigeria/ICE-Nigeria-txt-and-xml-files_version_Nov_03_2015/ice-nig/txt - without speaker tags"],
}


def api(params):
    key = urllib.parse.urlencode(sorted(params.items()))
    f = CACHE / ("wk_" + re.sub(r"[^A-Za-z0-9]+", "_", key)[:180] + ".json")
    if f.exists():
        return json.loads(f.read_text())
    url = "https://en.wiktionary.org/w/api.php?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                d = json.load(r)
            break
        except Exception as e:
            time.sleep(3 * (a + 1)); d = {"error": str(e)}
    if "error" not in d:
        f.write_text(json.dumps(d))
    time.sleep(1.0)
    return d


def strip_wikitext(s):
    s = re.sub(r"\{\{(?:lb|lbl|label)\|[^}]*\}\}", "", s)
    s = re.sub(r"\{\{(?:l|m|w|q|qualifier|gloss|gl|n-g|non-gloss)\|(?:[a-z-]+\|)?([^}|]*)[^}]*\}\}", r"\1", s)
    s = re.sub(r"\{\{[^}]*\}\}", "", s)
    s = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"'''?|<[^>]+>", "", s)
    return " ".join(s.split())[:160]


def wiktionary(title, variety):
    d = api({"action": "parse", "page": title, "prop": "wikitext", "redirects": 1})
    if "parse" not in d:
        return {"found": False, "reason": "no page"}
    wt = d["parse"]["wikitext"]["*"]
    sections = LANG_SECTION.get(variety, ["English"])
    labels = LABELS[variety]
    for sec in sections:
        m = re.search(rf"^=={re.escape(sec)}==\s*$(.*?)(?=^==[^=]|\Z)", wt, re.S | re.M)
        if not m:
            continue
        body = m.group(1)
        for line in body.splitlines():
            if not line.startswith("#") or line.startswith(("#:", "#*")):
                continue
            lbs = [x.lower() for grp in re.findall(r"\{\{(?:lb|lbl|label)\|[a-z-]+\|([^}]*)\}\}", line) for x in grp.split("|")]
            hit = [l for l in lbs if l.strip() in labels]
            if hit or (sec != "English"):   # any sense in a non-English section counts (e.g. Nigerian Pidgin)
                return {"found": True, "section": sec, "labels": hit or [sec], "gloss": strip_wikitext(line.lstrip("# ")),
                        "url": f"https://en.wiktionary.org/wiki/{urllib.parse.quote(d['parse']['title'].replace(' ', '_'))}#{sec.replace(' ', '_')}"}
        # page exists, section exists, but no labelled sense: report the labels that are there
        on_page = sorted({x.strip().lower() for grp in re.findall(r"\{\{(?:lb|lbl|label)\|[a-z-]+\|([^}]*)\}\}", body) for x in grp.split("|")})
        return {"found": False, "reason": f"no sense labelled for {variety} in {sec}", "labels_on_page": on_page[:15],
                "url": f"https://en.wiktionary.org/wiki/{urllib.parse.quote(d['parse']['title'].replace(' ', '_'))}"}
    return {"found": False, "reason": f"no {sections} section"}


@lru_cache(maxsize=None)
def corpus_text(key):
    parts = []
    for spec in CORPORA.get(key, []):
        rel, _, filt = spec.partition("@")
        allowed = set(filt.split("|")) if filt else None
        p = RAW / rel
        files = [p] if p.is_file() else [x for x in p.rglob("*") if x.is_file() and not x.name.startswith("._")
                                         and x.suffix in (".txt", ".conllu", ".json", ".xml", "")]
        for fp in files:
            try:
                t = fp.read_text(encoding="utf-8-sig", errors="ignore")
            except Exception:
                continue
            if fp.suffix == ".conllu":   # keep only the FORM column
                t = "\n".join(l.split("\t")[1] for l in t.splitlines() if l and l[0].isdigit() and "\t" in l)
            elif fp.suffix == ".xml":   # NUS SMS: keep messages whose sender country is allowed
                msgs = re.findall(r"<message .*?</message>", t, re.S)
                keep = []
                for m in msgs:
                    c = re.search(r"<country>([^<]*)</country>", m); tx = re.search(r"<text>([^<]*)</text>", m)
                    if tx and (allowed is None or (c and c.group(1).strip() in allowed)):
                        keep.append(tx.group(1))
                t = "\n".join(keep)
            elif allowed is not None and "COSEM" in t[:200]:   # CoSEM: nationality = chars 2-3 of the 3rd tag field
                keep = []
                for l in t.splitlines():
                    m = re.match(r"<+COSEM:([^>]+)>+\t?(.*)", l)
                    if m and len(m.group(1).split("-")) > 2 and m.group(1).split("-")[2][2:4] in allowed:
                        keep.append(m.group(2))
                t = "\n".join(keep)
            elif fp.suffix == ".json":
                side = "en" if allowed == {"EN"} else "pcm"
                t = "\n".join(json.loads(l)["translation"].get(side, "") for l in t.splitlines() if l.strip())
            parts.append(t.lower())
    return "\n".join(parts)


def count(form, key):
    text = corpus_text(key)
    if not text:
        return None
    pat = r"(?<![a-z'])" + re.escape(form.lower()).replace(r"\ ", r"\s+") + r"(?![a-z'])"
    return len(re.findall(pat, text))


# Lead's gate (EXPERIMENT_PLAN §2.3): authentic-corpus rate >= 5x the genre-matched control.
GATES = {
    "aae": [("aae_tw", "sae_tw"), ("coraal", "nus_us")],
    "sge": [("sge", "nus_us")],
    "ind": [("ind", "nus_us")],
    "ngp": [("ngp", "mafand_en")],
    "inf": [("nus_us", "mafand_en")],
}


@lru_cache(maxsize=None)
def ntokens(key):
    return len(re.findall(r"[a-z0-9']+", corpus_text(key)))


def rate_ratio(form, a, c):
    ca, cc = count(form, a), count(form, c)
    na, nc = ntokens(a), ntokens(c)
    if not na or not nc:
        return None
    ra, rc = (ca + 0.5) / na * 1e6, (cc + 0.5) / nc * 1e6
    return {"auth": a, "ctrl": c, "auth_count": ca, "ctrl_count": cc, "auth_per_M": round(ra, 2), "ctrl_per_M": round(rc, 2),
            "ratio": round(ra / rc, 2), "auth_tokens": na, "ctrl_tokens": nc}


def main():
    variety, path = sys.argv[1], sys.argv[2]
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        form, sae, typ = cols[0].strip(), cols[1].strip(), cols[2].strip()
        title = cols[3].strip() if len(cols) > 3 and cols[3].strip() else form
        ev = {"form": form, "sae": sae, "type": typ, "wiktionary": wiktionary(title, variety)}
        corp = {"aae": ["aae", "sae_ref"], "sge": ["sge"], "ngp": ["ngp"], "ind": ["ind"], "inf": ["nus_us"]}.get(variety, [])
        if variety == "ngp":
            corp.append("nge")
        ev["attestation"] = {k: count(form, k) for k in corp}
        ev["gate"] = [rate_ratio(form, a, c) for a, c in GATES.get(variety, [])]
        out.append(ev)
        print(f"{form:22s} wk={'Y' if ev['wiktionary']['found'] else 'n'} {ev['attestation']}", file=sys.stderr)
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
