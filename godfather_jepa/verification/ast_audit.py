"""
Automated AST Static Import Checker for Godfather JEPA.

Enforces strict zero-touch isolation from legacy code by statically inspecting
abstract syntax trees of all files in the package to guarantee zero imports
of forbidden legacy modules and zero dynamic import bypasses.
"""
from __future__ import annotations

import ast
import os
import sys
from typing import List, NamedTuple, Optional, Sequence, Set, Union

# The 10 canonical forbidden legacy modules mandated by dispatch and ORIGINAL_REQUEST.md
FORBIDDEN_LEGACY_MODULES: frozenset[str] = frozenset({
    "models",
    "planning",
    "training",
    "zephyr",
    "dual_store",
    "scripts",
    "adaptation",
    "baselines",
    "diagnostics",
    "notebooks",
})

# Extended repository-root legacy modules that must not be imported at top-level
EXTENDED_LEGACY_MODULES: frozenset[str] = frozenset({
    "envs",
    "eval",
    "experiments",
})

# Default comprehensive set of forbidden modules
ALL_FORBIDDEN_MODULES: frozenset[str] = FORBIDDEN_LEGACY_MODULES | EXTENDED_LEGACY_MODULES


class ImportViolation(NamedTuple):
    """Represents an isolation import violation discovered by the AST auditor."""
    file_path: str
    line_number: int
    module: str
    source_line: str
    reason: str

    def __str__(self) -> str:
        return f"{self.file_path}:{self.line_number}: [{self.module}] {self.reason} -> '{self.source_line}'"


def get_max_relative_level(file_path: str, package_root: str) -> int:
    """
    Calculates the maximum relative import level (dots) permitted for a file
    such that the import does not escape package_root.

    Package root file (e.g. godfather_jepa/__init__.py): max level 1 ('from . import ...')
    Subdir file (e.g. godfather_jepa/models/predictor.py): max level 2 ('from .. import ...')
    Nested subdir file (e.g. godfather_jepa/models/sub/core.py): max level 3 ('from ... import ...')
    """
    rel = os.path.relpath(file_path, package_root)
    dirname = os.path.dirname(rel)
    if not dirname or dirname == ".":
        return 1
    parts = [p for p in dirname.split(os.sep) if p and p != "."]
    return len(parts) + 1


class IsolationASTVisitor(ast.NodeVisitor):
    """
    AST visitor that traverses syntax trees detecting any static or dynamic imports
    of forbidden legacy modules or relative imports escaping the package root.
    """

    def __init__(
        self,
        file_path: str,
        source_lines: Sequence[str],
        package_root: str,
        forbidden_modules: Union[Set[str], frozenset[str]],
    ) -> None:
        self.file_path = file_path
        self.source_lines = source_lines
        self.package_root = package_root
        self.forbidden_modules = forbidden_modules
        self.violations: List[ImportViolation] = []
        self.max_rel_level = get_max_relative_level(file_path, package_root)

    def _get_source_line(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return "<unknown>"

    def visit_Import(self, node: ast.Import) -> None:
        """
        Inspects statements like:
        import models
        import models.jepa_v3 as jepa
        import os, sys, planning
        """
        for alias in node.names:
            top_level = alias.name.split(".")[0]
            if top_level in self.forbidden_modules:
                self.violations.append(
                    ImportViolation(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        module=alias.name,
                        source_line=self._get_source_line(node.lineno),
                        reason=f"Direct import of forbidden legacy module '{top_level}'",
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """
        Inspects statements like:
        from models import jepa_v3
        from . import crafter_env
        from ..models.encoder import Conv4Encoder
        from ...models import legacy
        """
        # Case A: Relative import (level > 0)
        if node.level > 0:
            if node.level > self.max_rel_level:
                self.violations.append(
                    ImportViolation(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        module=node.module or "<relative>",
                        source_line=self._get_source_line(node.lineno),
                        reason=(
                            f"Relative import level {node.level} escapes package root "
                            f"(max allowed {self.max_rel_level} for this directory depth)"
                        ),
                    )
                )
            # Safe intra-package relative import
            self.generic_visit(node)
            return

        # Case B: Dynamic import functions from importlib or builtins
        if node.module in ("importlib", "builtins"):
            for alias in node.names:
                if alias.name in ("import_module", "__import__"):
                    self.violations.append(
                        ImportViolation(
                            file_path=self.file_path,
                            line_number=node.lineno,
                            module=f"{node.module}.{alias.name}",
                            source_line=self._get_source_line(node.lineno),
                            reason="Importing dynamic import functions ('import_module', '__import__') violates strict static isolation",
                        )
                    )

        # Case C: Absolute import (level == 0)
        if node.module:
            top_level = node.module.split(".")[0]
            if top_level in self.forbidden_modules:
                self.violations.append(
                    ImportViolation(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        module=node.module,
                        source_line=self._get_source_line(node.lineno),
                        reason=f"From-import of forbidden legacy module '{top_level}'",
                    )
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """
        Inspects dynamic import calls like:
        __import__('models')
        builtins.__import__('planning')
        importlib.import_module('zephyr')
        import_module('dual_store')
        exec('import models')
        getattr(importlib, 'import_module')('models')
        __builtins__['__import__']('models')
        """
        func = node.func

        # 1. Dynamic execution via exec / eval / compile
        if getattr(func, "id", None) in ("exec", "eval", "compile"):
            self.violations.append(
                ImportViolation(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    module=getattr(func, "id"),
                    source_line=self._get_source_line(node.lineno),
                    reason="Dynamic execution via exec/eval/compile is prohibited",
                )
            )

        # 2. Reflective getattr import
        elif (
            isinstance(func, ast.Name)
            and func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in ("import_module", "__import__")
        ):
            self.violations.append(
                ImportViolation(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    module=f"getattr({node.args[1].value!r})",
                    source_line=self._get_source_line(node.lineno),
                    reason="Reflective getattr import is prohibited",
                )
            )

        # 3. Subscript access to dynamic import function
        elif isinstance(func, ast.Subscript) and (
            (isinstance(func.slice, ast.Constant) and func.slice.value in ("import_module", "__import__"))
            or (hasattr(node, "slice") and isinstance(getattr(node, "slice"), ast.Constant) and getattr(node, "slice").value in ("import_module", "__import__"))
        ):
            slice_val = func.slice.value if isinstance(func.slice, ast.Constant) else getattr(node, "slice").value
            self.violations.append(
                ImportViolation(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    module=f"subscript({slice_val!r})",
                    source_line=self._get_source_line(node.lineno),
                    reason="Subscript access to dynamic import function is prohibited",
                )
            )

        # 4. __import__ detection
        elif (isinstance(func, ast.Name) and func.id == "__import__") or (
            isinstance(func, ast.Attribute) and func.attr == "__import__"
        ):
            target = "<dynamic>"
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                target = node.args[0].value
            self.violations.append(
                ImportViolation(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    module=f"__import__({target!r})",
                    source_line=self._get_source_line(node.lineno),
                    reason="Dynamic '__import__' call is prohibited in strict isolation mode",
                )
            )

        # 5. import_module detection
        elif (isinstance(func, ast.Attribute) and func.attr == "import_module") or (
            isinstance(func, ast.Name) and func.id == "import_module"
        ):
            target = "<dynamic>"
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                target = node.args[0].value
            self.violations.append(
                ImportViolation(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    module=f"import_module({target!r})",
                    source_line=self._get_source_line(node.lineno),
                    reason="Dynamic 'import_module' call is prohibited in strict isolation mode",
                )
            )

        self.generic_visit(node)


def audit_package_isolation(
    package_dir: str,
    forbidden_modules: Optional[Union[Set[str], frozenset[str]]] = None,
) -> List[ImportViolation]:
    """
    Recursively walks package_dir, parsing every .py file and detecting
    any import referencing forbidden legacy modules or dynamic import calls.

    Raises FileNotFoundError if package_dir does not exist.
    """
    if not os.path.exists(package_dir):
        raise FileNotFoundError(f"Package directory does not exist: {package_dir}")

    if forbidden_modules is None:
        forbidden_modules = ALL_FORBIDDEN_MODULES

    package_root = os.path.abspath(package_dir)
    violations: List[ImportViolation] = []

    for root, dirs, files in os.walk(package_root):
        # Skip hidden and cache directories
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for fname in sorted(files):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, OSError) as e:
                violations.append(
                    ImportViolation(
                        file_path=fpath,
                        line_number=0,
                        module="<read_error>",
                        source_line="",
                        reason=f"Failed to read file: {e}",
                    )
                )
                continue

            source_lines = content.splitlines()
            try:
                tree = ast.parse(content, filename=fpath)
            except SyntaxError as e:
                violations.append(
                    ImportViolation(
                        file_path=fpath,
                        line_number=e.lineno or 0,
                        module="<syntax_error>",
                        source_line=e.text.strip() if e.text else "",
                        reason=f"Python syntax error in package: {e}",
                    )
                )
                continue

            visitor = IsolationASTVisitor(fpath, source_lines, package_root, forbidden_modules)
            visitor.visit(tree)
            violations.extend(visitor.violations)

    return violations


def format_violation_report(violations: List[ImportViolation]) -> str:
    """Formats violations into a structured human-readable report."""
    if not violations:
        return "PASS: 0 isolation violations detected. Package is 100% clean."

    lines = [f"FAIL: {len(violations)} isolation violation(s) detected:"]
    for idx, v in enumerate(violations, start=1):
        lines.append(f"  {idx}. {v.file_path}:{v.line_number} -> {v.module}")
        if v.source_line:
            lines.append(f"     Code:   {v.source_line}")
        lines.append(f"     Reason: {v.reason}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entrypoint for AST isolation checking.

    Usage:
        python -m godfather_jepa.verification.ast_audit [package_directory]

    Exit codes:
        0: Clean (0 violations)
        1: Isolation violations found
        2: Error (directory not found or invalid args)
    """
    if argv is None:
        argv = sys.argv[1:]

    if argv:
        target_dir = argv[0]
    else:
        # Default: package root of godfather_jepa
        target_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isdir(target_dir):
        print(f"ERROR: Target directory does not exist: {target_dir}", file=sys.stderr)
        return 2

    violations = audit_package_isolation(target_dir)
    report = format_violation_report(violations)
    print(report)

    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
