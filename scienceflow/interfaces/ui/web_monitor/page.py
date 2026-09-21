"""Self-contained browser UI. No CDN, model calls, or arbitrary file serving."""

from html import escape
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files

from ..logo import LOGO_PIXELS


def _brand_logo_svg():
    """Render the same pixel artwork as the terminal without external assets."""
    rows = len(LOGO_PIXELS)
    columns = len(LOGO_PIXELS[0])
    cells = "".join(
        f'<rect x="{x}" y="{y}" width="1" height="1" fill="{color}"/>'
        for y, row in enumerate(LOGO_PIXELS)
        for x, color in enumerate(row)
    )
    return (
        f'<svg class="brand-logo" viewBox="0 0 {columns} {rows}" '
        f'role="img" aria-label="ScienceFlow microscope" shape-rendering="crispEdges">{cells}</svg>'
    )


try:
    _VERSION = version("scienceflow")
except PackageNotFoundError:
    _VERSION = "dev"

_SCRIPTS = (
    "core.js",
    "metric.js",
    "lineage_layout.js",
    "lineage_viewport.js",
    "lineage.js",
    "view.js",
    "refresh.js",
)


def build_page() -> str:
    """Bundle packaged assets inline; HTTP exposes no arbitrary asset paths."""
    assets = files(__package__).joinpath("assets")
    script = "\n".join(
        assets.joinpath(name).read_text(encoding="utf-8") for name in _SCRIPTS
    )
    replacements = {
        "__SCIENCEFLOW_CSS__": assets.joinpath("theme.css").read_text(encoding="utf-8"),
        "__SCIENCEFLOW_JS__": "(() => {\n'use strict';\n" + script + "\n})();",
        "__SCIENCEFLOW_LOGO__": _brand_logo_svg(),
        "__SCIENCEFLOW_VERSION__": escape(_VERSION),
    }
    page = assets.joinpath("shell.html").read_text(encoding="utf-8")
    for marker, content in replacements.items():
        page = page.replace(marker, content)
    return page


PAGE = build_page()
