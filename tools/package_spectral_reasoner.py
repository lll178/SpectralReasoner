"""Create a clean SpectralReasoner source release directory.

The default package is intentionally pure:

- no examples directory
- no datasets or local KB files
- no model bundles or run outputs
- no extra research artifacts
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ROOT_FILES = [
    "README.md",
    "LICENSE",
    "COMMERCIAL_LICENSE.md",
    "THIRD_PARTY_NOTICES.md",
    "requirements.txt",
    ".gitignore",
    ".gitattributes",
    "pyproject.toml",
]

DOC_FILES = [
    "docs/SpectralReasoner_User_Manual_ZH.md",
    "docs/SpectralReasoner_User_Manual_EN.md",
    "docs/OPEN_SOURCE_RELEASE_CHECKLIST.md",
]

TOOL_FILES = [
    "tools/package_spectral_reasoner.py",
]

EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
}

EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
    ".pt",
    ".pth",
    ".onnx",
    ".bin",
    ".zip",
    ".pdf",
}


def copy_file(src_rel: str, out_dir: Path) -> None:
    src = ROOT / src_rel
    if not src.exists():
        print(f"skip missing: {src_rel}")
        return
    dst = out_dir / src_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copy {src_rel}")


def ignore_release_files(dir_path: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path = Path(dir_path) / name
        if name in EXCLUDE_DIR_NAMES:
            ignored.add(name)
        elif path.is_file() and path.suffix.lower() in EXCLUDE_SUFFIXES:
            ignored.add(name)
    return ignored


def copy_tree(src_rel: str, out_dir: Path) -> None:
    src = ROOT / src_rel
    if not src.exists():
        print(f"skip missing tree: {src_rel}")
        return
    dst = out_dir / src_rel
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore_release_files)
    print(f"copy tree {src_rel}")


def assert_clean(out_dir: Path) -> None:
    forbidden_dirs = {"examples", "data", "runs", "pa" + "pers"}
    forbidden_suffixes = {".pt", ".pth", ".onnx", ".bin", ".pdf", ".zip"}
    problems: list[str] = []
    for path in out_dir.rglob("*"):
        rel_parts = set(path.relative_to(out_dir).parts)
        if path.is_dir() and path.name in forbidden_dirs:
            problems.append(str(path.relative_to(out_dir)))
        if rel_parts & forbidden_dirs:
            problems.append(str(path.relative_to(out_dir)))
        if path.is_file() and path.suffix.lower() in forbidden_suffixes:
            problems.append(str(path.relative_to(out_dir)))
    if problems:
        joined = "\n".join(sorted(set(problems))[:50])
        raise RuntimeError(f"release package is not clean:\n{joined}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("dist/spectral_reasoner_release"))
    parser.add_argument("--clean", action="store_true", help="Remove output directory before packaging.")
    args = parser.parse_args()

    out_dir = args.out
    if args.clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for rel in ROOT_FILES + DOC_FILES + TOOL_FILES:
        copy_file(rel, out_dir)
    copy_tree("src", out_dir)

    assert_clean(out_dir)
    print(f"release_dir={out_dir.resolve()}")


if __name__ == "__main__":
    main()
