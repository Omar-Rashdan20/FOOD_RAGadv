from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Food RAG — CLI and server entrypoint.")
    parser.add_argument("query", nargs="*", help="Food recommendation query.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-cross-encoder", action="store_true")
    parser.add_argument("--eval", metavar="FILE")
    parser.add_argument("--cache-stats", action="store_true")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server.")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.serve:
        return _serve(args)

    try:
        from src.rag_pipeline import build_pipeline
    except ModuleNotFoundError as exc:
        print(f"Missing dependency '{exc.name}'. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    from src.config import get_settings
    settings = get_settings()

    try:
        pipeline = build_pipeline(
            settings=settings,
            rebuild_index=args.rebuild_index,
            enable_cross_encoder=not args.no_cross_encoder,
        )
    except ModuleNotFoundError as exc:
        print(f"Missing dependency '{exc.name}'. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    if args.eval:
        return _run_eval(pipeline, args.eval)

    use_cache = not args.no_cache

    if args.query:
        query = " ".join(args.query)
        try:
            response = pipeline.rag_recommend(query, n_results=args.top_k, use_cache=use_cache)
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(response)
        if args.cache_stats:
            print(f"\n[Cache] {pipeline.cache.stats()}")
        return 0

    return _interactive(pipeline, top_k=args.top_k, use_cache=use_cache, cache_stats=args.cache_stats)


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install -r requirements.txt", file=sys.stderr)
        return 1

    from src.config import get_settings
    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port

    uvicorn.run("api.app:app", host=host, port=port, reload=False)
    return 0


def _interactive(pipeline, top_k, use_cache, cache_stats) -> int:
    print("Food RAG Chatbot — type a food request or 'exit' to quit.")
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if query.lower() in {"exit", "quit", "q"}:
            return 0
        if not query:
            continue

        try:
            response = pipeline.rag_recommend(query, n_results=top_k, use_cache=use_cache)
        except Exception as exc:
            response = f"Error: {exc}"

        print(f"\nAssistant:\n{response}")
        if cache_stats:
            print(f"[Cache] {pipeline.cache.stats()}")


def _run_eval(pipeline, eval_file: str) -> int:
    try:
        from src.evaluator import build_eval_samples, run_full_eval
    except ImportError as exc:
        print(f"Eval import error: {exc}", file=sys.stderr)
        return 1

    path = Path(eval_file)
    if not path.exists():
        print(f"Eval file not found: {path}", file=sys.stderr)
        return 1

    with path.open() as f:
        raw_samples = json.load(f)

    samples = build_eval_samples(pipeline, raw_samples, n_results=10)
    report = run_full_eval(samples, llm_client=pipeline.llm_client)
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
