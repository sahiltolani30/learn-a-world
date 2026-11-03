#!/usr/bin/env python3
"""
Deep Forensic AST Analyzer for Project ATLAS / Prometheus JEPA.
Performs exhaustive structural AST inspection across every Python file in the repository.
"""

import ast
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple, Optional

REPO_ROOT = Path("/Users/harshitatolani/Sahil/ml 2").resolve()
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", ".vscode", ".idea"}

CRAFTER_ACTIONS = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep", "place_stone", "place_table", "place_furnace",
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
    "make_wood_sword", "make_stone_sword", "make_iron_sword"
]

CRAFTER_ACHIEVEMENTS = [
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace",
    "place_plant", "place_stone", "place_table", "wake_up"
]

CRAFTER_ITEMS = [
    "wood", "stone", "iron", "diamond", "sapling", "coal",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword"
]

BANNED_EXACT_NAMES = {
    "CRAFT_REQUIRED_MATERIALS",
    "CRAFTING_RECIPES",
    "RECIPES",
    "RECIPE_TABLE",
    "KNOWN_FIXTURES",
    "SEMANTIC_MILESTONE_PRODUCTS",
    "ACTION_TO_INGREDIENT",
    "ACTION_TO_INGREDIENTS",
    "TERRAIN_RULES",
    "TOOL_REQUIREMENTS",
}

class DeepASTForensics:
    def __init__(self, root: Path):
        self.root = root
        self.py_files: List[Path] = []
        self.total_lines_scanned = 0
        
        # Findings
        self.all_act_defs: List[Dict[str, Any]] = []
        self.all_act_calls: List[Dict[str, Any]] = []
        self.suspicious_assignments: List[Dict[str, Any]] = []
        self.all_dict_literals_examined: int = 0
        self.flagged_dict_literals: List[Dict[str, Any]] = []
        self.material_gating_checks: List[Dict[str, Any]] = []
        self.privileged_info_accesses: List[Dict[str, Any]] = []
        self.banned_name_matches: List[Dict[str, Any]] = []

    def collect_files(self):
        for root, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".agents")]
            for f in files:
                if f.endswith(".py"):
                    self.py_files.append(Path(root) / f)
        self.py_files.sort()

    def run_forensics(self):
        for fpath in self.py_files:
            rel = fpath.relative_to(self.root)
            try:
                content = fpath.read_text(encoding="utf-8")
                lines = content.splitlines()
                self.total_lines_scanned += len(lines)
                tree = ast.parse(content, filename=str(fpath))
            except Exception as e:
                self.suspicious_assignments.append({
                    "file": str(rel),
                    "line": 0,
                    "issue": f"Syntax/parse error: {e}"
                })
                continue
            
            self.inspect_file(tree, rel, lines)

    def inspect_file(self, tree: ast.AST, rel: Path, lines: List[str]):
        is_test = "test" in rel.name or "tests" in rel.parts
        is_verification = "verification" in rel.parts or "canaries" in rel.name

        for node in ast.walk(tree):
            # 1. Inspect Assignments
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    tname = getattr(target, "id", None)
                    if not tname:
                        continue
                    if tname in BANNED_EXACT_NAMES:
                        self.banned_name_matches.append({
                            "file": str(rel),
                            "line": node.lineno,
                            "var": tname,
                            "is_test_or_verification": (is_test or is_verification),
                            "code": lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                        })
                    
                    # Check suspicious patterns in variable names
                    upper = tname.upper()
                    if ("RECIPE" in upper or "CRAFT" in upper or "INGREDIENT" in upper) and not (is_test or is_verification):
                        # Inspect the assigned value
                        if isinstance(node.value, ast.Dict):
                            self.suspicious_assignments.append({
                                "file": str(rel),
                                "line": node.lineno,
                                "var": tname,
                                "val_type": "Dict",
                                "code": lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                            })

            # 2. Inspect Dict Literals
            if isinstance(node, ast.Dict):
                self.all_dict_literals_examined += 1
                keys = []
                for k in node.keys:
                    if isinstance(k, ast.Constant):
                        keys.append(k.value)
                    elif isinstance(k, ast.Name):
                        keys.append(k.id)
                    else:
                        keys.append(None)
                
                # Check if keys are Crafter actions or items or int action indices
                str_keys = [str(k) for k in keys if k is not None]
                lower_keys = set(k.lower() for k in str_keys)
                
                # Action indices as keys
                int_keys = [k for k in keys if isinstance(k, int)]
                
                has_craft_items = lower_keys.intersection(set(CRAFTER_ITEMS))
                has_craft_actions = lower_keys.intersection(set(CRAFTER_ACTIONS))
                has_action_indices = len(int_keys) >= 3 and any(k in range(17) for k in int_keys)

                if (has_craft_items or (has_craft_actions and len(has_craft_actions) >= 2) or has_action_indices) and not (is_test or is_verification):
                    # Inspect values to see if they define recipes/ingredients
                    self.flagged_dict_literals.append({
                        "file": str(rel),
                        "line": node.lineno,
                        "keys": str_keys[:10],
                        "has_craft_items": list(has_craft_items),
                        "has_craft_actions": list(has_craft_actions),
                        "has_action_indices": has_action_indices,
                        "code": lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    })

            # 3. Inspect Function Definitions (act, step, plan)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == "act":
                    args = [a.arg for a in node.args.args]
                    self.all_act_defs.append({
                        "file": str(rel),
                        "line": node.lineno,
                        "args": args,
                        "is_test": is_test,
                        "code": lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    })
                elif "material_gating" in node.name or "suppression" in node.name:
                    self.material_gating_checks.append({
                        "file": str(rel),
                        "line": node.lineno,
                        "func_name": node.name,
                        "args": [a.arg for a in node.args.args],
                        "code": lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    })

            # 4. Inspect Call Nodes to act()
            if isinstance(node, ast.Call):
                func = node.func
                is_act_call = False
                if isinstance(func, ast.Attribute) and func.attr == "act":
                    is_act_call = True
                elif isinstance(func, ast.Name) and func.id == "act":
                    is_act_call = True
                
                if is_act_call:
                    arg_reprs = []
                    for a in node.args:
                        if isinstance(a, ast.Name):
                            arg_reprs.append(a.id)
                        elif isinstance(a, ast.Attribute):
                            arg_reprs.append(f"{getattr(a.value, 'id', 'attr')}.{a.attr}")
                        elif isinstance(a, ast.Constant):
                            arg_reprs.append(repr(a.value))
                        else:
                            arg_reprs.append("<expr>")
                    kw_reprs = [f"{kw.arg}=" for kw in node.keywords]
                    self.all_act_calls.append({
                        "file": str(rel),
                        "line": node.lineno,
                        "args": arg_reprs,
                        "keywords": kw_reprs,
                        "is_test": is_test,
                        "code": lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    })

            # 5. Privileged Info accesses: e.g. info["inventory"], info["achievements"]
            if isinstance(node, ast.Subscript):
                slice_val = None
                if isinstance(node.slice, ast.Constant):
                    slice_val = node.slice.value
                elif hasattr(node, "slice") and isinstance(getattr(node, "slice"), ast.Constant):
                    slice_val = getattr(node, "slice").value
                
                if isinstance(slice_val, str) and slice_val.lower() in {"inventory", "achievements", "semantic"}:
                    # Check what object is subscripted
                    val_id = getattr(node.value, "id", None)
                    val_attr = getattr(node.value, "attr", None)
                    self.privileged_info_accesses.append({
                        "file": str(rel),
                        "line": node.lineno,
                        "target": slice_val,
                        "container": val_id or val_attr or "<expr>",
                        "is_test": is_test,
                        "code": lines[node.lineno - 1] if node.lineno <= len(lines) else ""
                    })

    def print_summary(self):
        print(f"=== DEEP AST FORENSICS SUMMARY ===")
        print(f"Files scanned: {len(self.py_files)}")
        print(f"Total lines scanned: {self.total_lines_scanned}")
        print(f"Total dict literals examined: {self.all_dict_literals_examined}")
        print(f"\n1. Banned Variable Names ({len(self.banned_name_matches)}):")
        for b in self.banned_name_matches:
            print(f"  - {b['file']}:{b['line']} -> {b['var']} (test_or_verification={b['is_test_or_verification']})")

        print(f"\n2. Suspicious Assignments ({len(self.suspicious_assignments)}):")
        for s in self.suspicious_assignments:
            print(f"  - {s['file']}:{s['line']} -> {s['var']} ({s['val_type']}): {s['code']}")

        print(f"\n3. Flagged Dict Literals in Non-Test Source ({len(self.flagged_dict_literals)}):")
        for f in self.flagged_dict_literals:
            print(f"  - {f['file']}:{f['line']} -> craft_items={f['has_craft_items']}, craft_actions={f['has_craft_actions']}, action_indices={f['has_action_indices']}")
            print(f"    Code: {f['code']}")
            print(f"    Keys: {f['keys']}")

        print(f"\n4. Discovered act() Definitions ({len(self.all_act_defs)}):")
        for a in self.all_act_defs:
            print(f"  - {a['file']}:{a['line']} (test={a['is_test']}) -> args={a['args']}")

        print(f"\n5. Act Calls in Non-Test Code ({len([c for c in self.all_act_calls if not c['is_test']])}):")
        for c in [c for c in self.all_act_calls if not c['is_test']]:
            print(f"  - {c['file']}:{c['line']} -> args={c['args']} kw={c['keywords']} | {c['code']}")

        print(f"\n6. Material Gating Functions ({len(self.material_gating_checks)}):")
        for m in self.material_gating_checks:
            print(f"  - {m['file']}:{m['line']} -> {m['func_name']}({m['args']})")

        print(f"\n7. Privileged Info Accesses in Non-Test Code ({len([p for p in self.privileged_info_accesses if not p['is_test']])}):")
        for p in [p for p in self.privileged_info_accesses if not p['is_test']]:
            print(f"  - {p['file']}:{p['line']} -> {p['container']}['{p['target']}'] | {p['code']}")

if __name__ == "__main__":
    analyzer = DeepASTForensics(REPO_ROOT)
    analyzer.collect_files()
    analyzer.run_forensics()
    analyzer.print_summary()
