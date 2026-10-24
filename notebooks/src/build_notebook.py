"""Assemble notebooks/Landmark_Customer_Intelligence.ipynb from the ordered part files in notebooks/src/.

Part-file format:  a line '%%MD' starts a markdown cell, a line '%%CODE' starts a code cell.
"""
from pathlib import Path
import nbformat as nbf

SRC = Path(__file__).resolve().parent
OUT = SRC.parent / "Landmark_Customer_Intelligence.ipynb"


def parse(text: str):
    cells, kind, buf = [], None, []
    for line in text.splitlines():
        if line.strip() in ("%%MD", "%%CODE"):
            if kind is not None and "".join(buf).strip():
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind, buf = ("md" if line.strip() == "%%MD" else "code"), []
        else:
            buf.append(line)
    if kind is not None and "".join(buf).strip():
        cells.append((kind, "\n".join(buf).strip("\n")))
    return cells


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for part in sorted(SRC.glob("part*.txt")):
        for kind, src in parse(part.read_text()):
            nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nbf.write(nb, OUT)
    print(f"wrote {OUT} with {len(nb.cells)} cells")


if __name__ == "__main__":
    main()
