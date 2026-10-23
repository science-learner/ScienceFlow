"""Render canonical Markdown as a self-contained, printable report."""

from __future__ import annotations

import base64
from html import escape

from markdown_it import MarkdownIt


_STYLE = """
:root{color-scheme:light;--ink:#28313e;--muted:#67717e;--line:#dedee5}
*{box-sizing:border-box}body{margin:0;background:#f3f2ef;color:var(--ink);
font:16px/1.7 system-ui,sans-serif}article{max-width:1080px;margin:32px auto;
padding:48px 56px;background:white;box-shadow:0 3px 18px #0000000a}
h1,h2,h3{font-family:Georgia,serif;line-height:1.3}h1{font-size:36px}
h2{margin-top:36px;border-bottom:1px solid var(--line);padding-bottom:8px}
a{color:#635389}img{max-width:100%;height:auto}table{border-collapse:collapse;
width:100%;font-size:13px;overflow-wrap:anywhere}th,td{text-align:left;
border-bottom:1px solid var(--line);padding:8px}th{background:#f5f3f8}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f3f8;padding:16px}
code{font:13px ui-monospace,monospace}blockquote{margin-left:0;padding-left:18px;
border-left:3px solid #9588ac;color:var(--muted)}p,li{overflow-wrap:anywhere}
@media(max-width:720px){article{margin:0;padding:24px}h1{font-size:28px}}
@page{margin:15mm}
@media print{body{background:white}article{max-width:none;margin:0;padding:0;
box-shadow:none}h1,h2,h3{break-after:avoid}img,tr{break-inside:avoid}
pre{max-height:none;overflow:visible}thead{display:table-header-group}}
"""


def render_html(markdown: str, svgs: dict[str, str]) -> str:
    """Allow only generated report figures; never fetch Markdown image URLs."""
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")

    def image(tokens, index, options, env):
        token = tokens[index]
        source = token.attrGet("src") or ""
        prefix = "report_assets/"
        name = source[len(prefix) :] if source.startswith(prefix) else ""
        alt = escape(token.content, quote=True)
        if name not in svgs:
            return f'<span class="image-unavailable">{alt}</span>'
        data = base64.b64encode(svgs[name].encode("utf-8")).decode("ascii")
        return f'<img alt="{alt}" src="data:image/svg+xml;base64,{data}">'

    parser.renderer.rules["image"] = image
    body = parser.render(markdown)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" '
        "content=\"default-src 'none'; style-src 'unsafe-inline'; img-src data:\">"
        "<title>ScienceFlow research report</title>"
        f"<style>{_STYLE}</style></head><body><article>{body}</article></body></html>"
    )
