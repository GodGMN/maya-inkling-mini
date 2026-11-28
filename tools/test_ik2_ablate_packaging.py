#!/usr/bin/env python3
"""Packaging checks for tools/ik2_ablate/: which of its files ship, and that the lab's imports resolve from the repository.

Nothing here runs or prints the lab's scripts or data: the checks read file names, the manifest, the packaging scripts,
and the import statements of the lab's Python files (through `ast`), and the staging runs on throwaway fixture trees.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LAB = ROOT / "tools" / "ik2_ablate"
STAGE = ROOT / "docker" / "stage-ik2-tools.sh"
MANIFEST = json.loads((LAB / "MANIFEST.json").read_text())


def manifest_names():
    runtime = set(MANIFEST["runtime_assets"])
    development = {name for names in MANIFEST["development_only"].values() for name in names}
    return runtime, development


def lab_files():
    return {p.relative_to(LAB).as_posix() for p in LAB.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and p.name != "MANIFEST.json"}


def docker_ignored(rel_path: str, patterns: list[str]) -> bool:
    """The .dockerignore rules this repository uses: `*` stays inside one path segment, a matched directory takes
    its files with it, `!` re-includes, the last rule that matches wins."""
    parts = rel_path.split("/")
    parents = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
    ignored = False
    for pattern in patterns:
        negate = pattern.startswith("!")
        body = pattern[1:] if negate else pattern
        regex = re.compile("".join("[^/]*" if c == "*" else re.escape(c) for c in body.rstrip("/")) + r"\Z")
        if any(regex.match(p) for p in parents):
            ignored = not negate
    return ignored


def dockerignore_patterns() -> list[str]:
    lines = (ROOT / ".dockerignore").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def run_stage(root: pathlib.Path, context: pathlib.Path) -> None:
    subprocess.run(["sh", str(STAGE), str(root), str(context)], check=True, capture_output=True)


def tree(path: pathlib.Path) -> set[str]:
    return {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}


class ManifestTests(unittest.TestCase):
    def test_every_lab_file_is_listed_once(self):
        runtime, development = manifest_names()
        self.assertFalse(runtime & development, "a file is both runtime and development-only")
        listed = runtime | development
        present = lab_files()
        self.assertEqual(sorted(present - listed), [], "files in tools/ik2_ablate/ that MANIFEST.json does not list")
        self.assertEqual(sorted(listed - present), [], "MANIFEST.json lists files that are not there")

    def test_runtime_assets_are_the_directions(self):
        runtime, _ = manifest_names()
        self.assertTrue(runtime)
        for name in runtime:
            self.assertTrue(name.startswith("directions/"), name)
        self.assertIn("directions/v21-pc4.f32", runtime)

    def test_service_image_reads_the_runtime_asset(self):
        # docker/start-ik2-gfx1100.sh opens /opt/maya/ablate/v21-pc4.f32; package-ik2-gfx1100.sh fills ablate/ from directions/
        start = (ROOT / "docker" / "start-ik2-gfx1100.sh").read_text()
        package = (ROOT / "docker" / "package-ik2-gfx1100.sh").read_text()
        self.assertIn("/opt/maya/ablate/v21-pc4.f32", start)
        self.assertIn('"$root/tools/ik2_ablate/directions/."', package)
        self.assertIn('"$context/ablate/"', package)


class StageTests(unittest.TestCase):
    def test_fixture_tree_leaves_the_lab_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, context = pathlib.Path(tmp, "repo"), pathlib.Path(tmp, "context")
            for rel in ["tools/ik2_ablate/directions/dir.f32", "tools/ik2_ablate/lab.sh", "tools/ik2_ablate/job.py",
                        "tools/ik2/template.txt", "tools/ik2_bench.py", "tools/ik2_nv_bench.hip",
                        "tools/strata_tokenizer.py", "tools/gguf_reader.py", "tools/gguf_writer.py",
                        "tools/maya_quants.py", "tools/maya_names.py", "tools/ablate_fit.py",
                        "tools/needle_bench.py", "tools/inkling2_gguf.py", "tools/test_unrelated.py"]:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n")
            run_stage(root, context)
            staged = tree(context)
            self.assertEqual({p for p in staged if p.startswith("tools/ik2_ablate")}, set())
            self.assertEqual(staged, {
                "tools/ik2/template.txt", "tools/ik2_bench.py", "tools/ik2_nv_bench.hip", "tools/strata_tokenizer.py",
                "tools/gguf_reader.py", "tools/gguf_writer.py", "tools/maya_quants.py", "tools/maya_names.py",
                "tools/ablate_fit.py", "tools/needle_bench.py", "tools/inkling2_gguf.py"})

    def test_repository_tools_stage_without_the_lab(self):
        with tempfile.TemporaryDirectory() as tmp:
            context = pathlib.Path(tmp)
            run_stage(ROOT, context)
            staged = tree(context)
            self.assertEqual(sorted(p for p in staged if "ik2_ablate" in p), [])
            for needed in ["tools/ik2_bench.py", "tools/ik2_compare_decode.py", "tools/ablate_fit.py",
                           "tools/gguf_reader.py", "tools/inkling2_gguf.py"]:
                self.assertIn(needed, staged)

    def test_package_script_uses_the_stage_helper(self):
        package = (ROOT / "docker" / "package-ik2-gfx1100.sh").read_text()
        self.assertIn('docker/stage-ik2-tools.sh', package)
        self.assertNotIn('"$root/tools"/ik2*', package)


class DockerignoreTests(unittest.TestCase):
    def test_release_builds_drop_the_lab_and_keep_the_runtime_assets(self):
        patterns = dockerignore_patterns()
        runtime, development = manifest_names()
        for name in sorted(development):
            self.assertTrue(docker_ignored(f"tools/ik2_ablate/{name}", patterns), name)
        for name in sorted(runtime):
            self.assertFalse(docker_ignored(f"tools/ik2_ablate/{name}", patterns), name)

    def test_other_tools_still_ship(self):
        patterns = dockerignore_patterns()
        for rel in ["tools/ik2_bench.py", "tools/ablate_fit.py", "tools/test_ablate_fit.py", "tools/gguf_reader.py",
                    "serve/server.py", "maya.py"]:
            self.assertFalse(docker_ignored(rel, patterns), rel)

    def test_tests_of_unshipped_code_stay_out(self):
        patterns = dockerignore_patterns()
        for rel in ["tools/test_ik2_grade.py", "tools/test_apply_lora_gguf.py", "tools/test_ik2_ablate_packaging.py"]:
            self.assertTrue(docker_ignored(rel, patterns), rel)

    def test_matcher_follows_docker_rules(self):
        rules = ["a/*", "!a/keep", "b.py"]
        self.assertTrue(docker_ignored("a/x.py", rules))
        self.assertFalse(docker_ignored("a/keep/f.bin", rules))
        self.assertTrue(docker_ignored("b.py", rules))
        self.assertFalse(docker_ignored("c/b.py", rules))


class ImportTests(unittest.TestCase):
    def sibling_modules(self):
        tools = {p.stem for p in (ROOT / "tools").glob("*.py")}
        lab = {p.stem for p in LAB.glob("*.py")}
        return tools, lab

    def imported_modules(self, path: pathlib.Path) -> set[str]:
        found = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                found.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
        return found

    def test_tools_modules_used_by_the_lab_are_reachable_from_the_repository(self):
        tools, lab = self.sibling_modules()
        problems = []
        for path in sorted(LAB.glob("*.py")):
            wanted = {m for m in self.imported_modules(path) if m in tools and m not in lab}
            if not wanted:
                continue
            source = path.read_text()
            reaches_tools = bool(re.search(r"sys\.path\.(insert|append)\(.*(parents\[1\]|\.parent\.parent)", source)
                                 or re.search(r"_sys\.path\.append\(.*parents\[1\]", source))
            if not reaches_tools:
                problems.append(f"{path.name}: imports {sorted(wanted)} with tools/ not on sys.path")
        self.assertEqual(problems, [])

    def test_lab_scripts_parse(self):
        for path in sorted(LAB.glob("*.py")):
            with self.subTest(script=path.name):
                ast.parse(path.read_text())

    def test_shim_imports_a_tools_module_from_a_clean_checkout(self):
        # a throwaway layout: repo/tools/ablate_fit.py (a stub) and repo/tools/ik2_ablate/job.py, which carries the
        # shim line the lab's scripts use, copied from final.py
        source = (LAB / "final.py").read_text().splitlines()
        index = next(i for i, line in enumerate(source) if line.startswith("from ablate_fit import"))
        shim_and_import = "\n".join(source[index - 1:index + 1])
        self.assertIn("parents[1]", shim_and_import)
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp)
            (repo / "tools" / "ik2_ablate").mkdir(parents=True)
            (repo / "tools" / "ablate_fit.py").write_text("def refused(text):\n    return False\n")
            job = repo / "tools" / "ik2_ablate" / "job.py"
            job.write_text(shim_and_import + "\nprint('ok', refused('fixture'))\n")
            result = subprocess.run([sys.executable, "-E", str(job)], cwd=tmp, capture_output=True, text=True)
            self.assertEqual((result.returncode, result.stdout.strip()), (0, "ok False"), result.stderr)
            # a copy beside the script (how the lab runs) wins over tools/
            (repo / "tools" / "ik2_ablate" / "ablate_fit.py").write_text("def refused(text):\n    return True\n")
            result = subprocess.run([sys.executable, "-E", str(job)], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.stdout.strip(), "ok True", result.stderr)


if __name__ == "__main__":
    unittest.main()
