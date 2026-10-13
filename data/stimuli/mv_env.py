"""Import shim for Multi-VALUE 0.3 on <host> (env `p3mv`). Import this BEFORE `multivalue`.
- Stanza 1.8.2 checkpoints need torch.load(weights_only=False) under torch>=2.6 (official Stanford files via hf-mirror).
- MV_COREF=0 (default for bulk scoring): replaces the Stanza pipeline by a stub so no coref model (ELECTRA-large) is loaded;
  rules that need coreference clusters (null referential pronouns, object-pronoun drop, ...) then see no clusters and do not fire.
  MV_COREF=1 (stimulus generation): full pipeline, models from $STANZA_RESOURCES_DIR, ELECTRA via the hf-mirror cache.
- CPU only: never touches a GPU.
"""
import os
os.environ.setdefault("STANZA_RESOURCES_DIR", "<HOME>/arr_oct/runs/P3/stanza_resources")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com"); os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import functools, torch
torch.load = functools.partial(torch.load, weights_only=False)
if os.environ.get("MV_COREF", "0") == "0":
    import stanza

    class _NoPipeline:
        def __init__(self, *a, **k): pass
    stanza.Pipeline = _NoPipeline
    import multivalue.BaseDialect as _bd
    _bd.BaseDialect.create_coref_cluster = lambda self, string: None


import collections
import multivalue.BaseDialect as _bd2
_orig_clear = _bd2.BaseDialect.clear


def _clear(self):  # rules may emit sub-rule names that are not in the whitelist; do not crash on them
    _orig_clear(self)
    self.modification_counter = collections.defaultdict(int, self.modification_counter)


_bd2.BaseDialect.clear = _clear


def get_dialect(name, seed=1234):
    from multivalue import Dialects
    return getattr(Dialects, name)(morphosyntax=True, seed=seed)
