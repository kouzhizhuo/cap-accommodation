#!/usr/bin/env python3
"""Word counts of every \\caption in the paper sources (main vs appendix), macros stripped. Run from paper/."""
import re
from pathlib import Path
MAIN = ['sections/1_intro.tex', 'sections/3_method.tex', 'sections/4_replies.tex', 'sections/5_updates.tex', 'sections/6_validity.tex',
        'tables/tab_main.tex', 'tables/tab_comp.tex', 'tables/tab_update.tex', 'tables/tab_sizes.tex', 'tables/tab_audit.tex', 'tables/tab_measures.tex', 'tables/tab_controls.tex']
def captions(text):
    out, i = [], 0
    while True:
        i = text.find('\\caption{', i)
        if i < 0: return out
        j, depth = i + 9, 1
        while depth:
            depth += {'{': 1, '}': -1}.get(text[j], 0); j += 1
        out.append(text[i + 9:j - 1]); i = j
def words(c):
    c = re.sub(r'\\(res|resT|rci|Cref|cref|citep|citet|ref)\{[^}]*\}', 'X', c)
    c = re.sub(r'\$[^$]*\$', 'X', c); c = re.sub(r'\\[a-zA-Z]+\*?', ' ', c); c = re.sub(r'[{}]', ' ', c)
    return len(c.split())
files = sorted(f for f in set(map(str, Path('sections').glob('*.tex'))) | set(map(str, Path('tables').glob('*.tex'))) if not Path(f).name.startswith('._') and 'tab_main_src' not in f)
for f in files:
    for k, c in enumerate(captions(Path(f).read_text())):
        print(f"{'MAIN' if f in MAIN else 'APP '} {words(c):4d}  {f}#{k}  {c[:70]!r}")
