from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download
from rich.console import Console

from .cli import (
    DATASET_ID,
    filter_metadata,
    load_metadata,
    map_book_paths,
    normalize_with_map,
    norm,
    page_files,
)

console = Console()
DEFAULT_PRESET = Path(__file__).resolve().parents[2] / "presets" / "concept_hunt_v3.json"


class ConceptPreset:
    def __init__(self, obj: dict):
        self.names = [norm(x) for x in obj["early_names"]]
        self.tanzih = [norm(x) for x in obj["tanzih_terms"]]
        self.tajsim = [norm(x) for x in obj["tajsim_terms"]]
        self.exact = [norm(x) for x in obj["exact_terms"]]
        self.context = [norm(x) for x in obj.get("context_terms", [])]
        self.opponent_markers = [norm(x) for x in obj.get("opponent_markers", [])]
        self.quote_markers = [norm(x) for x in obj.get("quote_markers", [])]

    @classmethod
    def load(cls, path: Path) -> "ConceptPreset":
        return cls(json.loads(path.read_text(encoding="utf-8")))


def all_positions(hay: str, terms: list[str]) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for term in terms:
        start = 0
        while True:
            pos = hay.find(term, start)
            if pos < 0:
                break
            found.append((pos, term))
            start = pos + max(1, len(term))
    found.sort(key=lambda x: (x[0], -len(x[1])))
    return found


def excerpt_from_match(original: str, index_map: list[int], start: int, end: int, radius: int) -> str:
    if not index_map:
        return re.sub(r"\s+", " ", original[: radius * 2]).strip()
    a_norm = max(0, start - radius)
    b_norm = min(len(index_map) - 1, end + radius)
    a = index_map[a_norm]
    b = index_map[b_norm] + 1
    return re.sub(r"\s+", " ", original[a:b]).strip()


def nearby_terms(positions: list[tuple[int, str]], center: int, window: int) -> list[str]:
    left = max(0, center - window)
    right = center + window
    out: list[str] = []
    seen: set[str] = set()
    for pos, term in positions:
        if left <= pos <= right and term not in seen:
            out.append(term)
            seen.add(term)
    return out


def speaker_flags(window: str, preset: ConceptPreset) -> list[str]:
    flags: list[str] = []
    if any(x in window for x in preset.opponent_markers):
        flags.append("POSSIBLE_OPPONENT_QUOTE")
    if any(x in window for x in preset.quote_markers):
        flags.append("QUOTATION_OR_ATTRIBUTION")
    return flags


def scan_file(repo_path: str, meta: dict, preset: ConceptPreset, args: argparse.Namespace) -> list[dict]:
    local = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=repo_path,
        cache_dir=args.cache_dir,
    )
    hits: list[dict] = []

    trigger_terms = {
        "tanzih": preset.tanzih,
        "tajsim": preset.tajsim,
        "exact": preset.exact,
    }[args.mode]

    with open(local, "r", encoding="utf-8") as f:
        for line in f:
            try:
                page = json.loads(line)
            except json.JSONDecodeError:
                continue

            body = page.get("body") or ""
            foot = page.get("footnotes") or ""
            sources = [("BODY", body)]
            if args.include_footnotes and foot:
                sources.append(("FOOTNOTE", foot))

            for source_field, original in sources:
                if not original:
                    continue
                normalized, imap = normalize_with_map(original)
                if not normalized:
                    continue

                triggers = all_positions(normalized, trigger_terms)
                if not triggers:
                    continue

                name_positions = all_positions(normalized, preset.names) if args.mode != "exact" else []
                if args.mode != "exact" and not name_positions:
                    continue
                context_positions = all_positions(normalized, preset.context)

                for pos, trigger in triggers:
                    names = nearby_terms(name_positions, pos, args.window) if args.mode != "exact" else []
                    if args.mode != "exact" and not names:
                        continue

                    left = max(0, pos - args.window)
                    right = min(len(normalized), pos + len(trigger) + args.window)
                    window_text = normalized[left:right]
                    context = nearby_terms(context_positions, pos, args.window)
                    flags = speaker_flags(window_text, preset)
                    excerpt = excerpt_from_match(original, imap, pos, pos + len(trigger), args.excerpt)

                    label = {
                        "tanzih": "TANZIH_CONCEPT",
                        "tajsim": "TAJSIM_CONCEPT",
                        "exact": "EXACT_FORMULA",
                    }[args.mode]
                    if source_field == "FOOTNOTE":
                        label += "_FOOTNOTE"

                    hits.append({
                        "class": label,
                        "mode": args.mode,
                        "source_field": source_field,
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
                        "trigger": trigger,
                        "early_names": " | ".join(names),
                        "context_terms": " | ".join(context),
                        "speaker_flags": " | ".join(flags),
                        "excerpt": excerpt,
                    })
                    if args.max_hits_per_book and len(hits) >= args.max_hits_per_book:
                        return hits
    return hits


def write_results(hits: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "hits.jsonl").open("w", encoding="utf-8") as f:
        for h in hits:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")

    if hits:
        with (out_dir / "hits.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(hits[0].keys()))
            writer.writeheader()
            writer.writerows(hits)

    classes: dict[str, int] = {}
    sources: dict[str, int] = {}
    for h in hits:
        classes[h["class"]] = classes.get(h["class"], 0) + 1
        sources[h["source_field"]] = sources.get(h["source_field"], 0) + 1

    payload = {
        "total_hits": len(hits),
        "classes": classes,
        "sources": sources,
        "unique_books": len({h["book_id"] for h in hits}),
        "possible_opponent_quotes": sum(
            "POSSIBLE_OPPONENT_QUOTE" in h["speaker_flags"] for h in hits
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def cmd_scan(args: argparse.Namespace) -> int:
    preset = ConceptPreset.load(Path(args.preset))
    df = filter_metadata(load_metadata(args.cache_dir), args)
    metas = {int(r["book_id"]): r for r in df.iter_rows(named=True)}
    console.print(f"[bold]Режим:[/bold] {args.mode}")
    console.print(f"[bold]Книг после фильтра:[/bold] {len(metas)}")

    paths = map_book_paths(page_files())
    jobs = [(bid, paths[bid]) for bid in metas if bid in paths]
    if args.max_books:
        jobs = jobs[: args.max_books]
    console.print(f"[bold]Файлов pages.jsonl к сканированию:[/bold] {len(jobs)}")

    all_hits: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(scan_file, path, metas[bid], preset, args): (bid, path)
            for bid, path in jobs
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            bid, path = futures[fut]
            try:
                all_hits.extend(fut.result())
            except Exception as exc:
                console.print(
                    f"[red]Ошибка[/red] book_id={bid} {path}: {exc}",
                    file=sys.stderr,
                )
            if done % 25 == 0:
                console.print(f"{done}/{len(jobs)}; совпадений: {len(all_hits)}")

    all_hits.sort(
        key=lambda x: (
            1 if "POSSIBLE_OPPONENT_QUOTE" in x["speaker_flags"] else 0,
            x.get("death_hijri") or 99999,
            x["book_id"],
            x.get("sequence_num") or 0,
            x["trigger"],
        )
    )
    write_results(all_hits, Path(args.output))
    console.print(
        f"[green]Готово.[/green] Найдено: {len(all_hits)}. Результаты: {args.output}"
    )
    return 0


def add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--death-max", type=int)
    parser.add_argument("--death-min", type=int)
    parser.add_argument("--category", action="append")
    parser.add_argument("--book-id", action="append", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shamela v3 concept-first theological scanner"
    )
    parser.add_argument("--cache-dir", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan")
    add_filters(scan)
    scan.add_argument("--preset", default=str(DEFAULT_PRESET))
    scan.add_argument("--mode", choices=["tanzih", "tajsim", "exact"], required=True)
    scan.add_argument("--window", type=int, default=1100)
    scan.add_argument("--excerpt", type=int, default=800)
    scan.add_argument("--workers", type=int, default=12)
    scan.add_argument("--output", required=True)
    scan.add_argument("--max-books", type=int, default=0)
    scan.add_argument("--max-hits-per-book", type=int, default=0)
    scan.add_argument("--include-footnotes", action="store_true")
    scan.set_defaults(func=cmd_scan)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "max_books", 0) == 0:
        args.max_books = None
    if getattr(args, "max_hits_per_book", 0) == 0:
        args.max_hits_per_book = None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
