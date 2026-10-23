"""Native terminal microscope companion: bold silhouette, ruby and ivory.

Drawn on the cell grid rather than downsampled from detailed artwork.
The ivory lens joint suggests an eye; the stage and foot identify a microscope.
"""

_PALETTE = {".": "transparent", "r": "#ce3d60", "w": "#fff1d6", "d": "#582b43"}


def _pixels(rows: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(_PALETTE[cell] for cell in row) for row in rows)


LOGO_PIXELS = _pixels((
    "...wwww.....",
    "...rrrr.....",
    "..rrrr......",
    "..rrrrrrr...",
    ".rrrrwwrrr..",
    ".rr..ww.rrr.",
    "........rrr.",
    ".wwwww..rrr.",
    "..rr...rrr..",
    "..rrrrrrr...",
    ".rrrrrrrrrr.",
    "............",
))

COMPACT_LOGO_PIXELS = LOGO_PIXELS
