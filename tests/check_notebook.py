"""Scan an executed notebook: execution coverage, errors, stderr warnings, and suspicious 'nan' text in rendered reflections."""
import re
import sys
from pathlib import Path

import nbformat

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "notebooks" / "Landmark_Customer_Intelligence.ipynb"
nb = nbformat.read(path, 4)
code = [(i, c) for i, c in enumerate(nb.cells) if c.cell_type == "code"]
print(f"{path.name}: {len(nb.cells)} cells, {len(code)} code, executed {sum(1 for _, c in code if c.get('execution_count'))}")
for i, c in code:
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            print(f"  ERROR cell {i}: {o.get('ename')}: {o.get('evalue')[:200]}")
        if o.get("output_type") == "stream" and o.get("name") == "stderr":
            txt = o.get("text", "")
            if txt.strip():
                print(f"  stderr cell {i}: {txt.strip()[:220]!r}")
        data = o.get("data", {})
        md = data.get("text/markdown", "")
        if md and re.search(r"\bnan\b", md):
            print(f"  'nan' in rendered markdown, cell {i}: {md[:160]!r}")
