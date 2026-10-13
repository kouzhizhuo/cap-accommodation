#!/usr/bin/env python3
"""P1-style table colours for every table not styled by make_tables.py (user 2026-10-13). Idempotent; run after
make_tables.py (tools/regen.sh). Results tables: orange (cCmpA) log-scale shading of the key result column, value read
from numbers/results.tex for \\res-style cells. Setup/licence/manifest tables: light grey header row only."""
import math, re
from pathlib import Path
T = Path(__file__).resolve().parents[1]
RES = dict(re.findall(r"\\resdef\{([^}]*)\}\{([^}]*)\}", (T / 'numbers/results.tex').read_text()))
SHADE = {'tab_update': 5, 'tab_authentic': 1, 'tab_comp': 1, 'tab_controls': 2, 'tab_mainfull': 3,
         'tab_measures': 2, 'tab_mechanism': 4, 'tab_mitigation': 2, 'tab_sizes': 3}          # table -> key column index
HEADER = ['tab_sensitivity', 'tab_audit', 'tab_detectors', 'tab_licences', 'tab_operationalization', 'tab_revisions', 'tab_capsetup']

def value(cell):
    m = re.search(r"\\(?:res|resT|rcell|rpm|rhl)\{([^}]*)\}", cell)
    t = RES.get(m.group(1), '') if m else cell
    t = t.replace('\\ensuremath{-}', '-').replace('$<$', '').replace('<', '')
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    return float(m.group(0)) if m else None

def shade_file(name, col):
    p = T / f'tables/{name}.tex'; L = p.read_text().split('\n'); rows = []
    for i, l in enumerate(L):
        s = l.strip()
        if s.startswith('%') or '&' not in s or not s.endswith('\\\\') or 'multicolumn' in s or '\\textbf{' in s.split('&')[0] and i < 12 and 'toprule' not in s and any('\\midrule' in x for x in L[i + 1:i + 3]):
            continue
        cells = s[:-2].split('&')
        if len(cells) <= col or 'cellcolor' in cells[col]: continue
        v = value(cells[col])
        if v is not None: rows.append((i, cells, v))
    if not rows: return 0
    vmax = max(abs(v) for _, _, v in rows) or 1
    for i, cells, v in rows:
        k = int(round(2 + 25 * math.log1p(abs(v)) / math.log1p(vmax))) if vmax else 2
        cells[col] = f" \\cellcolor{{cCmpA!{k}}}" + cells[col].lstrip()
        ind = L[i][:len(L[i]) - len(L[i].lstrip())]
        L[i] = ind + '&'.join(c if j == 0 else c for j, c in enumerate(cells)).strip() + ' \\\\'
    p.write_text('\n'.join(L)); return len(rows)

def header_file(name):
    p = T / f'tables/{name}.tex'; L = p.read_text().split('\n')
    for i, l in enumerate(L):
        if '\\midrule' in l:
            j = i - 1
            while j > 0 and ('cmidrule' in L[j] or not L[j].strip()): j -= 1
            if 'rowcolor' not in L[j] and '&' in L[j]:
                ind = L[j][:len(L[j]) - len(L[j].lstrip())]; L[j] = ind + '\\rowcolor{gray!10} ' + L[j].lstrip()
            break
    p.write_text('\n'.join(L))

if __name__ == '__main__':
    n = {k: shade_file(k, c) for k, c in SHADE.items()}
    for h in HEADER: header_file(h)
    print('shaded cells:', n, '| header-tinted:', len(HEADER))
