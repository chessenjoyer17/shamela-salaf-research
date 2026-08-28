from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl
from huggingface_hub import hf_hub_download, list_repo_files
from rich.console import Console
from rich.table import Table

DATASET_ID = "AuthenticIlm/Shamela4_Full_DB"
DEFAULT_PRESET = Path(__file__).resolve().parent / "presets" / "salaf_space.json"
console = Console()

ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
HTML_TAG = re.compile(r"<[^>]+>")
BOOK_ID_RE = re.compile(r"/(\d+)__[^/]+/pages\.jsonl$")


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    out: list[str] = []
    idx: list[int] = []
    in_tag = False
    prev_space = False
    for i, ch in enumerate(text):
        if ch == "<":
            in_tag = True
            continue
        if in_tag:
            if ch == ">":
                in_tag = False
            continue
        if ch == "ـ" or ARABIC_DIACRITICS.match(ch) or unicodedata.combining(ch):
            continue
        ch = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ؤ": "و", "ئ": "ي"}.get(ch, ch)
        if ch.isspace():
            if prev_space:
                continue
            ch = " "
            prev_space = True
        else:
            prev_space = False
        out.append(ch)
        idx.append(i)
    return "".join(out), idx


def norm(s: str) -> str:
    return normalize_with_map(s)[0]


@dataclass
class Preset:
    positive: list[str]
    anti: list[str]
    spatial: list[str]
    names: list[str]

    @classmethod
    def load(cls, path: Path) -> "Preset":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            [norm(x) for x in obj["positive_terms"]],
            [norm(x) for x in obj["anti_spatial_terms"]],
            [norm(x) for x in obj["spatial_terms"]],
            [norm(x) for x in obj["early_names"]],
        )


def metadata_path(cache_dir: str | None = None) -> str:
    return hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename="_meta/book_metadata.parquet",
        cache_dir=cache_dir,
    )


def load_metadata(cache_dir: str | None = None) -> pl.DataFrame:
    return pl.read_parquet(metadata_path(cache_dir))


def cmd_categories(args: argparse.Namespace) -> int:
    df = load_metadata(args.cache_dir)
    out = (
        df.group_by(["category_id", "category_name_ar"])
        .agg(pl.len().alias("books"))
        .sort("category_id")
    )
    table = Table("ID", "Категория", "Книг")
    for row in out.iter_rows(named=True):
        table.add_row(str(row["category_id"]), str(row["category_name_ar"]), str(row["books"]))
    console.print(table)
    return 0


def page_files() -> list[str]:
    return [p for p in list_repo_files(DATASET_ID, repo_type="dataset") if p.endswith("/pages.jsonl")]


def map_book_paths(files: Iterable[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for p in files:
        m = BOOK_ID_RE.search("/" + p)
        if m:
            out[int(m.group(1))] = p
    return out


def filter_metadata(df: pl.DataFrame, args: argparse.Namespace) -> pl.DataFrame:
    if args.death_max is not None:
        df = df.filter(pl.col("main_author_death_hijri").is_not_null() & (pl.col("main_author_death_hijri") <= args.death_max))
    if args.death_min is not None:
        df = df.filter(pl.col("main_author_death_hijri").is_not_null() & (pl.col("main_author_death_hijri") >= args.death_min))
    if args.category:
        cond = None
        for c in args.category:
            x = pl.col("category_name_ar").str.contains(c, literal=True)
            cond = x if cond is None else (cond | x)
        df = df.filter(cond)
    if args.book_id:
        df = df.filter(pl.col("book_id").is_in(args.book_id))
    return df


def first_positions(hay: str, terms: list[str]) -> list[tuple[int, str]]:
    found = []
    for t in terms:
        pos = hay.find(t)
        if pos >= 0:
            found.append((pos, t))
    return sorted(found)


def classify(window: str, preset: Preset) -> tuple[str, list[str], list[str], list[str]]:
    anti = [t for t in preset.anti if t in window]
    spatial = [t for t in preset.spatial if t in window]
    names = [t for t in preset.names if t in window]
    if anti and names:
        label = "A_ATTRIBUTED_ANTI_SPATIAL"
    elif anti:
        label = "A_ANTI_SPATIAL"
    elif spatial and names:
        label = "C_ATTRIBUTED_SPATIAL_CONTEXT"
    elif spatial:
        label = "C_SPATIAL_CONTEXT"
    elif names:
        label = "B_ATTRIBUTED_UNRESOLVED"
    else:
        label = "D_UNRESOLVED"
    return label, anti, spatial, names


def excerpt_from_match(original: str, index_map: list[int], start: int, end: int, radius: int) -> str:
    if not index_map:
        return original[: radius * 2]
    a = index_map[max(0, start - radius)]
    b_norm = min(len(index_map) - 1, end + radius)
    b = index_map[b_norm] + 1
    return re.sub(r"\s+", " ", original[a:b]).strip()


def scan_file(repo_path: str, meta: dict, preset: Preset, args: argparse.Namespace) -> list[dict]:
    local = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=repo_path,
        cache_dir=args.cache_dir,
    )
    hits: list[dict] = []
    with open(local, "r", encoding="utf-8") as f:
        for line in f:
            try:
                page = json.loads(line)
            except json.JSONDecodeError:
                continue
            body = page.get("body") or ""
            foot = page.get("footnotes") or ""
            original = body + ("\n[FOOTNOTES]\n" + foot if foot else "")
            normalized, imap = normalize_with_map(original)
            positives = first_positions(normalized, preset.positive)
            if not positives:
                continue
            for pos, positive in positives:
                left = max(0, pos - args.window)
                right = min(len(normalized), pos + len(positive) + args.window)
                window = normalized[left:right]
                label, anti, spatial, names = classify(window, preset)
                if args.require_name and not names:
                    continue
                if args.only_anti and not anti:
                    continue
                if args.only_spatial and not spatial:
                    continue
                excerpt = excerpt_from_match(original, imap, pos, pos + len(positive), args.excerpt)
                hits.append({
                    "class": label,
                    "book_id": meta.get("book_id"),
                    "title": meta.get("title_ar"),
                    "author": meta.get("main_author_name_ar"),
                    "death_hijri": meta.get("main_author_death_hijri"),
                    "category": meta.get("category_name_ar"),
                    "repo_path": repo_path,
                    "page_id": page.get("page_id"),
                    "part": page.get("part"),
                    "page_num": page.get("page_num"),
                    "sequence_num": page.get("sequence_num"),
                    "positive": positive,
                    "anti_spatial": " | ".join(anti),
                    "spatial": " | ".join(spatial),
                    "early_names": " | ".join(names),
                    "excerpt": excerpt,
                })
                if args.max_hits_per_book and len(hits) >= args.max_hits_per_book:
                    return hits
    return hits


def write_results(hits: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "hits.jsonl"
    csv_path = out_dir / "hits.csv"
    with jsonl.open("w", encoding="utf-8") as f:
        for h in hits:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    if hits:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hits[0]))
            w.writeheader()
            w.writerows(hits)
    summary = {}
    for h in hits:
        summary[h["class"]] = summary.get(h["class"], 0) + 1
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_scan(args: argparse.Namespace) -> int:
    preset = Preset.load(Path(args.preset))
    df = filter_metadata(load_metadata(args.cache_dir), args)
    metas = {int(r["book_id"]): r for r in df.iter_rows(named=True)}
    console.print(f"[bold]Книг после фильтра:[/bold] {len(metas)}")
    paths = map_book_paths(page_files())
    jobs = [(bid, paths[bid]) for bid in metas if bid in paths]
    if args.max_books:
        jobs = jobs[: args.max_books]
    console.print(f"[bold]Файлов pages.jsonl к сканированию:[/bold] {len(jobs)}")

    all_hits: list[dict] = []
    if args.workers <= 1:
        for i, (bid, p) in enumerate(jobs, 1):
            all_hits.extend(scan_file(p, metas[bid], preset, args))
            if i % 25 == 0:
                console.print(f"{i}/{len(jobs)}; совпадений: {len(all_hits)}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(scan_file, p, metas[bid], preset, args): (bid, p) for bid, p in jobs}
            done = 0
            for fut in as_completed(futures):
                done += 1
                bid, p = futures[fut]
                try:
                    all_hits.extend(fut.result())
                except Exception as e:
                    console.print(f"[red]Ошибка[/red] book_id={bid} {p}: {e}", file=sys.stderr)
                if done % 25 == 0:
                    console.print(f"{done}/{len(jobs)}; совпадений: {len(all_hits)}")

    rank = {
        "A_ATTRIBUTED_ANTI_SPATIAL": 0,
        "A_ANTI_SPATIAL": 1,
        "B_ATTRIBUTED_UNRESOLVED": 2,
        "C_ATTRIBUTED_SPATIAL_CONTEXT": 3,
        "C_SPATIAL_CONTEXT": 4,
        "D_UNRESOLVED": 5,
    }
    all_hits.sort(key=lambda x: (rank.get(x["class"], 99), x.get("death_hijri") or 9999, x["book_id"], x.get("sequence_num") or 0))
    write_results(all_hits, Path(args.output))
    console.print(f"[green]Готово.[/green] Найдено: {len(all_hits)}. Результаты: {args.output}")
    return 0


def add_common_filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("--death-max", type=int, help="Максимальный год смерти автора по хиджре")
    p.add_argument("--death-min", type=int, help="Минимальный год смерти автора по хиджре")
    p.add_argument("--category", action="append", help="Подстрока арабского названия категории; можно повторять")
    p.add_argument("--book-id", action="append", type=int, help="Конкретный book_id; можно повторять")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Research Shamela 4 for baynuna/fawqiyya and spatial language")
    p.add_argument("--cache-dir", default=None, help="Каталог кеша Hugging Face")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("categories", help="Показать категории Shamela")
    c.set_defaults(func=cmd_categories)

    s = sub.add_parser("scan", help="Сканировать корпус")
    add_common_filters(s)
    s.add_argument("--preset", default=str(DEFAULT_PRESET))
    s.add_argument("--window", type=int, default=1400, help="Контекст вокруг ключевой фразы в нормализованных символах")
    s.add_argument("--excerpt", type=int, default=650, help="Радиус сохраняемой цитаты")
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--output", default="results")
    s.add_argument("--max-books", type=int, default=0, help="0 = без ограничения")
    s.add_argument("--max-hits-per-book", type=int, default=0, help="0 = без ограничения")
    s.add_argument("--require-name", action="store_true", help="Оставлять только места, где рядом есть имя раннего авторитета")
    s.add_argument("--only-anti", action="store_true", help="Оставлять только места с непространственной лексикой")
    s.add_argument("--only-spatial", action="store_true", help="Оставлять только места с пространственной лексикой (контрольная выборка)")
    s.set_defaults(func=cmd_scan)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "max_books", None) == 0:
        args.max_books = None
    if getattr(args, "max_hits_per_book", None) == 0:
        args.max_hits_per_book = None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
