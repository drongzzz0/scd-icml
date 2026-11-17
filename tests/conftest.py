from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parents[1]
_src = _repo_root / 'src'
for _p in (str(_repo_root), str(_src)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
