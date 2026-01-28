from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a small HybridQA subset with N tables and passages."
    )
    parser.add_argument("--source", default="data/hybridqa", help="HybridQA root folder")
    parser.add_argument("--out", default="data/small", help="Output dataset folder")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=5, help="Number of tables to copy")
    args = parser.parse_args()

    source = Path(args.source)
    released_path = source / "HybridQA" / "released_data" / f"{args.split}.json"
    tables_root = source / "WikiTables-WithLinks" / "tables_tok"
    passages_root = source / "WikiTables-WithLinks" / "request_tok"

    if not released_path.exists():
        raise SystemExit(f"Missing questions file: {released_path}")
    if not tables_root.exists():
        raise SystemExit(f"Missing tables folder: {tables_root}")
    if not passages_root.exists():
        raise SystemExit(f"Missing passages folder: {passages_root}")

    out_root = Path(args.out)
    out_tables = out_root / "tables_tok"
    out_passages = out_root / "request_tok"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_passages.mkdir(parents=True, exist_ok=True)

    data = load_questions(released_path)
    seen: set[str] = set()
    copied = 0
    missing_tables = 0
    missing_passages = 0

    for item in data:
        table_id = item.get("table_id")
        if not table_id or table_id in seen:
            continue
        seen.add(table_id)
        table_path = tables_root / f"{table_id}.json"
        passage_path = passages_root / f"{table_id}.json"
        if not table_path.exists():
            missing_tables += 1
            continue
        shutil.copy2(table_path, out_tables / table_path.name)
        if passage_path.exists():
            shutil.copy2(passage_path, out_passages / passage_path.name)
        else:
            missing_passages += 1
        copied += 1
        if copied >= args.limit:
            break

    print(f"Copied tables: {copied}")
    if missing_tables:
        print(f"Missing tables skipped: {missing_tables}")
    if missing_passages:
        print(f"Missing passages skipped: {missing_passages}")
    print(f"Output: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
