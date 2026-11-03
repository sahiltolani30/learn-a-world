#!/usr/bin/env python3
"""
Exhaustive Forensic AST Static Analysis Script for Project ATLAS / Prometheus JEPA.
Validates 100% compliance with AGENTS.md, GEMINI.md, and ORIGINAL_REQUEST.md (Benchmark Mode).
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

REPO_ROOT = Path("/Users/harshitatolani/Sahil/ml 2").resolve()

# Directories to skip
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", ".vscode", ".idea"}

# Banned names from AGENTS.md, GEMINI.md, and Godfather JEPA verification
BANNED_ASSIGNMENT_NAMES = {
    "CRAFT_REQUIRED_MATERIALS",
    "CRAFTING_RECIPES",
    "RECIPES",
    "RECIPE_TABLE",
    "KNOWN_FIXTURES",
    "SEMANTIC_MILESTONE_PRODUCTS",
    "RECIPE_MAPPING",
    "ACTION_TO_INGREDIENT",
    "ACTION_TO_INGREDIENTS",
    "TERRAIN_RULES",
    "TOOL_REQUIREMENTS",
}

# Suspicious substrings in variable names
BANNED_SUBSTRINGS = [
    "CRAFT_REQUIRED",
    "RECIPE_TABLE",
    "ACTION_TO_INGREDIENT",
    "TOOL_REQUIREMENT",
]

# Privileged keys
PRIVILEGED_KEYS = {
    "inventory",
    "achievements",
    "semantic",
    "player_pos",
    "recipes",
    "crafting_recipes",
}

class ASTScanner:
    def __init__(self, root: Path):
        self.root = root
        self.py_files: List[Path] = []
        self.violations: List[Dict[str, Any]] = []
        self.dict_literals: List[Dict[str, Any]] = []
        self.act_methods: List[Dict[str, Any]] = []
        self.material_gating_refs: List[Dict[str, Any]] = []
        self.privileged_accesses: List[Dict[str, Any]] = []

    def collect_files(self):
        for root, dirs, files in os.walk(self.root):
            # Prune skip dirs
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".agents")]
            for f in files:
                if f.endswith(".py"):
                    self.py_files.append(Path(root) / f)
        self.py_files.sort()

    def scan_all(self):
        for fpath in self.py_files:
            rel_path = fpath.relative_to(self.root)
            try:
                content = fpath.read_text(encoding="utf-8")
                tree = ast.parse(content, filename=str(fpath))
            except Exception as e:
                self.violations.append({
                    "file": str(rel_path),
                    "line": 0,
                    "type": "PARSE_ERROR",
                    "details": str(e)
                })
                continue

            self.inspect_tree(tree, rel_path, content.splitlines())

    def inspect_tree(self, tree: ast.AST, rel_path: Path, lines: List[str]):
        is_test_file = "test" in rel_path.name or "tests" in rel_path.parts
        is_verification_file = "verification" in rel_path.parts or "canaries" in rel_path.name

        for node in ast.walk(tree):
            # 1. Check assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    tname = getattr(target, "id", None)
                    if tname:
                        # Check banned exact match
                        if tname in BANNED_ASSIGNMENT_NAMES:
                            if not (is_test_file or is_verification_file):
                                self.violations.append({
                                    "file": str(rel_path),
                                    "line": node.lineno,
                                    "type": "BANNED_VARIABLE_ASSIGNMENT",
                                    "details": f"Assignment to banned cheat variable '{tname}'",
                                    "code": lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                                })
                        for sub in BANNED_SUBSTRINGS:
                            if sub in tname:
                                if not (is_test_file or is_verification_file):
                                    self.violations.append({
                                        "file": str(rel_path),
                                        "line": node.lineno,
                                        "type": "SUSPICIOUS_VARIABLE_NAME",
                                        "details": f"Assignment to suspicious name '{tname}' containing '{sub}'",
                                        "code": lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                                    })

            # 2. Check dictionary literals
            if isinstance(node, ast.Dict):
                # Inspect keys of the dictionary
                keys_evaluated = []
                for k in node.keys:
                    if isinstance(k, ast.Constant):
                        keys_evaluated.append(k.value)
                    elif isinstance(k, ast.Name):
                        keys_evaluated.append(k.id)
                    else:
                        keys_evaluated.append("<expr>")

                # Check if keys are integers or actions or look like crafting recipes
                # e.g., keys are action indices {10: ..., 11: ...} or material strings
                key_types = {type(k) for k in keys_evaluated}
                # If keys contain crafter action names or item names
                recipe_suspect_words = {
                    "wood", "stone", "iron", "diamond", "craft_wood_pickaxe",
                    "craft_stone_pickaxe", "craft_iron_pickaxe", "craft_wood_sword",
                    "craft_stone_sword", "craft_iron_sword", "place_table",
                    "place_furnace", "place_plant", "make_wood_pickaxe", "make_stone_pickaxe"
                }
                found_suspects = set(str(k).lower() for k in keys_evaluated).intersection(recipe_suspect_words)
                if found_suspects and not is_test_file and not is_verification_file:
                    self.dict_literals.append({
                        "file": str(rel_path),
                        "line": node.lineno,
                        "keys": keys_evaluated,
                        "suspects": list(found_suspects),
                        "code": lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    })

            # 3. Check act() methods
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "act":
                    args = [a.arg for a in node.args.args]
                    self.act_methods.append({
                        "file": str(rel_path),
                        "line": node.lineno,
                        "args": args,
                        "code": lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    })
                    # Check if signature violates pure visual boundary
                    # Must NOT have info, inventory, achievements, true_state
                    forbidden_args = set(args).intersection({"info", "inventory", "achievements", "true_state", "crafter_info"})
                    if forbidden_args:
                        self.violations.append({
                            "file": str(rel_path),
                            "line": node.lineno,
                            "type": "PRIVILEGED_ACT_SIGNATURE",
                            "details": f"act() method accepts privileged arguments: {forbidden_args}",
                            "code": lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                        })

            # 4. Check references to material gating
            if isinstance(node, ast.Name) and "material_gating" in node.id:
                self.material_gating_refs.append({
                    "file": str(rel_path),
                    "line": node.lineno,
                    "name": node.id,
                    "code": lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                })


if __name__ == "__main__":
    scanner = ASTScanner(REPO_ROOT)
    scanner.collect_files()
    print(f"Total Python files discovered (excluding .agents): {len(scanner.py_files)}")
    scanner.scan_all()
    print(f"Total Violations: {len(scanner.violations)}")
    for v in scanner.violations:
        print(f"VIOLATION: {v['file']}:{v['line']} - [{v['type']}] {v['details']}")
    print(f"\nSuspect Dict Literals: {len(scanner.dict_literals)}")
    for d in scanner.dict_literals:
        print(f"DICT: {d['file']}:{d['line']} - Suspects: {d['suspects']} -> Keys: {d['keys']}")
    print(f"\nDiscovered act() methods: {len(scanner.act_methods)}")
    for a in scanner.act_methods:
        print(f"ACT: {a['file']}:{a['line']} - args: {a['args']}")
