"""Workspace evaluator signature projection."""

from __future__ import annotations

import ast
from pathlib import Path


def extract_eval_signature(py_path: Path) -> str | None:
    """Extract the ``eval`` definition line and docstring from a Python file."""
    try:
        source = py_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return None

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "eval":
            continue
        lines = source.splitlines()
        signature_lines: list[str] = []
        for index in range(node.lineno - 1, min(node.lineno + 10, len(lines))):
            signature_lines.append(lines[index])
            if lines[index].rstrip().endswith(":"):
                break
        signature = "\n".join(signature_lines)
        docstring = ast.get_docstring(node)
        if docstring:
            signature += '\n    """\n'
            for line in docstring.splitlines():
                signature += f"    {line}\n"
            signature += '    """'
        return signature
    return None


__all__ = ["extract_eval_signature"]
