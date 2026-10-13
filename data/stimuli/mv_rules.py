import mv_env, json, sys
from multivalue import Dialects
out = {}
for cls in ["AfricanAmericanVernacular", "ColloquialSingaporeDialect", "NigerianDialect", "IndianDialect", "ScottishDialect", "AppalachianDialect", "IrishDialect"]:
    d = getattr(Dialects, cls)(morphosyntax=True, seed=1234)
    out[cls] = {k: round(float(v), 3) for k, v in d.morphosyntax_transforms.items()}
    print(cls, len(out[cls]), flush=True)
json.dump(out, open("mv_rules_by_variety.json", "w"), indent=1)
