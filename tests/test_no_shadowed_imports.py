"""Regression test: a function must not import a name and later rebind it.

sft.train_lora once imported `log` (the progress logger) and also used a list
called `log`, so the final log() call crashed after training finished. The GPU
code paths cannot run in CI, so this checks the pattern statically.
"""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "rdgen"


def _shadowed(fn: ast.FunctionDef) -> set[str]:
    imported, assigned = set(), set()
    for node in ast.walk(fn):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            assigned.add(node.id)
    return imported & assigned


def test_no_function_rebinds_an_imported_name():
    bad = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for name in _shadowed(node):
                    bad.append(f"{path.relative_to(SRC)}:{node.lineno} {node.name}() rebinds '{name}'")
    assert not bad, "imported names reassigned inside functions:\n" + "\n".join(bad)
