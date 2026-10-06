"""The mosaic grammar: many slices in one figure, from one line of text.

NiiVue's syntax (BSD-2), so a mosaic line written for NiiVue works here:

* ``A`` / ``C`` / ``S`` choose axial, coronal or sagittal for what follows;
* a number is a slice position in millimetres along that plane's axis;
* ``;`` starts a new row;
* ``X`` draws, on the following tiles, where the other tiles cut;
* ``L-`` hides the slice-position labels (``L+`` shows them again);
* ``H 0.3`` overlaps neighbouring tiles by 30 percent;
* ``R`` makes the following numbers 3-D renders. The CPU mosaic has no
  renderer, so render tiles are skipped and :func:`parse` reports them.

Pure functions; :mod:`bidsmgr.gui.viz.canvases.mosaic` draws the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_PLANE = {"A": "axial", "C": "coronal", "S": "sagittal"}


@dataclass(frozen=True)
class Tile:
    plane: str
    mm: float
    cross: bool = False


@dataclass
class Mosaic:
    rows: list[list[Tile]] = field(default_factory=list)
    labels: bool = True
    overlap: float = 0.0
    skipped_renders: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def tiles(self) -> list[Tile]:
        return [t for row in self.rows for t in row]


def parse(text: str) -> Mosaic:
    """Parse a mosaic line. Unknown tokens are reported, never fatal."""
    out = Mosaic()
    row: list[Tile] = []
    plane = "axial"
    render = False
    cross = False
    tokens = (text or "").replace(";", " ; ").split()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        up = tok.upper()
        if up == ";":
            if row:
                out.rows.append(row)
            row = []
            render = False
            cross = False
        elif up in _PLANE:
            plane = _PLANE[up]
            render = False
        elif up == "R":
            render = True
        elif up == "X":
            cross = True
        elif up in ("L-", "L+", "L"):
            out.labels = up != "L-"
        elif up in ("H", "V"):
            if i + 1 < len(tokens):
                try:
                    if up == "H":
                        out.overlap = max(0.0, min(0.9, float(tokens[i + 1])))
                    i += 1
                except ValueError:
                    out.errors.append(f"{tok} needs a number")
        else:
            try:
                mm = float(tok)
            except ValueError:
                out.errors.append(f"not understood: {tok}")
            else:
                if render:
                    out.skipped_renders += 1
                else:
                    row.append(Tile(plane=plane, mm=mm, cross=cross))
        i += 1
    if row:
        out.rows.append(row)
    return out


__all__ = ["Mosaic", "Tile", "parse"]
