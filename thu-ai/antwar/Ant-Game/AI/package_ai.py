from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def clean_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_dir() and path.name == "__pycache__":
            shutil.rmtree(path, ignore_errors=True)
            continue
        if path.is_file() and path.suffix in {".pyc", ".pyo", ".so", ".dylib", ".pyd"}:
            path.unlink(missing_ok=True)
            continue
        if path.is_file() and path.name == ".DS_Store":
            path.unlink(missing_ok=True)


def copy_file_mapping(output_dir: Path, source: Path, target: str) -> None:
    destination = output_dir / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree_mapping(output_dir: Path, source: Path, target: str) -> None:
    destination = output_dir / target
    shutil.copytree(source, destination, dirs_exist_ok=True)
    clean_tree(destination)


def assemble_layout(
    output_dir: Path,
    *,
    file_mappings: list[tuple[Path, str]],
    tree_mappings: list[tuple[Path, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_files = [
        (REPO_ROOT / "AI" / "main.py", "main.py"),
        (REPO_ROOT / "AI" / "common.py", "common.py"),
        (REPO_ROOT / "AI" / "protocol.py", "protocol.py"),
    ]
    base_trees = [
        (REPO_ROOT / "SDK", "SDK"),
        (REPO_ROOT / "tools", "tools"),
    ]
    for source, target in [*base_files, *file_mappings]:
        copy_file_mapping(output_dir, source, target)
    for source, target in [*base_trees, *tree_mappings]:
        copy_tree_mapping(output_dir, source, target)


def require_empty_dir(dir_path: Path) -> None:
    if dir_path.exists() and not dir_path.is_dir():
        raise SystemExit(f"output path exists and is not a directory: {dir_path}")
    dir_path.mkdir(parents=True, exist_ok=True)
    if any(dir_path.iterdir()):
        raise SystemExit(f"output directory must be empty: {dir_path}")


def target_spec(target: str) -> tuple[str, list[tuple[Path, str]], list[tuple[Path, str]]]:
    if target == "random":
        return "ai_rand.zip", [(REPO_ROOT / "AI" / "ai_random.py", "ai.py")], []
    if target == "mcts":
        return "ai_mcts.zip", [
            (REPO_ROOT / "AI" / "ai_mcts.py", "ai.py"),
            (REPO_ROOT / "checkpoints" / "ai_mcts_latest.npz", "ai_mcts_model.npz"),
        ], []
    if target == "greedy":
        return "ai_greedy.zip", [(REPO_ROOT / "AI" / "ai_greedy.py", "ai.py")], [
            (REPO_ROOT / "AI" / "ai_greedy", "ai_greedy"),
        ]
    if target == "example":
        return "ai_example.zip", [(REPO_ROOT / "AI" / "ai_example.py", "ai.py")], []
    if target == "adaptive":
        return "ai_adaptive.zip", [
            (REPO_ROOT / "AI" / "ai_adaptive.py", "ai.py"),
            (REPO_ROOT / "checkpoints" / "ai_mcts_latest.npz", "ai_mcts_model.npz"),
        ], []
    raise SystemExit(f"unknown target: {target}")


def write_zip(source_dir: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            archive.write(path, path.relative_to(source_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package an AI into the submission layout")
    parser.add_argument("target", choices=["random", "mcts", "greedy", "example", "adaptive"])
    parser.add_argument("output", nargs="?", default=None, help="optional output .zip path or empty directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_name, file_mappings, tree_mappings = target_spec(args.target)

    if args.output and not args.output.lower().endswith(".zip"):
        output_dir = Path(args.output).resolve()
        require_empty_dir(output_dir)
        assemble_layout(output_dir, file_mappings=file_mappings, tree_mappings=tree_mappings)
        print(output_dir)
        return 0

    output_zip = Path(args.output).resolve() if args.output else (SCRIPT_DIR / archive_name).resolve()
    staging_dir = output_zip.with_suffix(output_zip.suffix + ".staging")
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=False)
    try:
        assemble_layout(staging_dir, file_mappings=file_mappings, tree_mappings=tree_mappings)
        write_zip(staging_dir, output_zip)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    print(output_zip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
