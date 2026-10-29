"""Isolation audit: verifies AST imports and repository cleanliness.

Ensures ATLAS depends ONLY on allowed read-only interfaces:
- dual_store.jepa_v3
- envs.crafter_wrapper
- eval.metrics
All other legacy modules (planning, training, models, zephyr, godfather_jepa) are forbidden.
"""

import ast
import subprocess
from pathlib import Path

FORBIDDEN_ROOTS = {
    "models",
    "planning",
    "training",
    "scripts",
    "zephyr",
    "godfather_jepa",
    "independent_jepa",
}

ALLOWED_DUAL_STORE_MODULES = {"dual_store.jepa_v3"}


def get_all_imports(file_path: Path):
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def test_atlas_ast_isolation():
    atlas_dir = Path(__file__).resolve().parent.parent
    py_files = list(atlas_dir.rglob("*.py"))
    assert len(py_files) > 0, "No python files found in atlas/"

    for py_file in py_files:
        imports = get_all_imports(py_file)
        for imp in imports:
            root_pkg = imp.split(".")[0]
            assert root_pkg not in FORBIDDEN_ROOTS, (
                f"Forbidden import '{imp}' found in {py_file.name}. "
                f"ATLAS must not depend on legacy {root_pkg}."
            )
            if root_pkg == "dual_store":
                assert any(imp.startswith(allowed) for allowed in ALLOWED_DUAL_STORE_MODULES), (
                    f"Forbidden dual_store import '{imp}' in {py_file.name}. "
                    f"Only {ALLOWED_DUAL_STORE_MODULES} is permitted."
                )


def test_git_cleanliness():
    repo_root = Path(__file__).resolve().parent.parent.parent
    res = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=True,
    )
    # Ensure atlas didn't modify core read-only modules: dual_store, envs, eval, models, planning, training
    read_only_targets = {"dual_store", "envs", "eval", "models", "planning", "training"}
    for line in res.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            status, path = parts
            if "?" not in status:
                first_dir = path.split("/")[0]
                assert first_dir not in read_only_targets, (
                    f"Forbidden modification to read-only repo module: {path}"
                )
