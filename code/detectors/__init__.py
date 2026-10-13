import os, sys
_here = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(_here, "..", "stimuli"), os.path.join(_here, "..")):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)
