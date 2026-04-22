#!/usr/bin/env python3
"""
Run grab_object.py once for each model provider, choosing the fastest responsive model
for that provider discovered from an OpenAI-compatible /v1/models endpoint.

Usage examples:
  python kinova_middleware/llm_clients/run_grab_by_provider.py
  python kinova_middleware/llm_clients/run_grab_by_provider.py --grab-script ./tools/grab_object.py --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import concurrent.futures
import subprocess
from pathlib import Path
from typing import Any
from urllib import request, error

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_PROMPT = "Reply with exactly OK."
DEFAULT_GRAB_SCRIPT = str(
    Path(__file__).resolve().parents[3] / "grab_object.py"
)  # repo-root-level grab_object.py by default


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "kinova-run-grab/1.0",
    }


def http_json(method: str, url: str, headers: dict[str, str], payload: dict[str, Any] | None, timeout: float) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=body, headers=headers, method=method)
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def discover_models(base_url: str, api_key: str, timeout: float) -> list[str]:
    data = http_json("GET", f"{base_url}/models", build_headers(api_key), None, timeout)
    items = data.get("data", []) if isinstance(data, dict) else []
    models = sorted({item.get("id", "").strip() for item in items if isinstance(item, dict) and item.get("id")})
    if not models:
        raise RuntimeError("No models were returned by /v1/models.")
    return models


def extract_text_fragment(payload: dict[str, Any]) -> str:
    # lightweight extractor for the most useful text fields returned by chat completions
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
            # fallback fields in streamed/delta style
            delta = first.get("delta")
            if isinstance(delta, dict):
                for key in ("content", "reasoning_content", "thinking"):
                    v = delta.get(key)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
    for key in ("output_text", "text"):
        v = payload.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def test_model_once(base_url: str, api_key: str, model: str, prompt: str, max_tokens: int, timeout: float) -> tuple[bool, float, str, str]:
    """
    Send a single non-streaming chat request to the model. Returns:
      (success, total_seconds, output_preview, error_message)
    """
    url = f"{base_url}/chat/completions"
    headers = build_headers(api_key)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }

    started = time.perf_counter()
    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            total_s = time.perf_counter() - started
            try:
                payload_json = json.loads(body)
            except Exception:
                return False, total_s, "", "invalid-json-response"
            out = extract_text_fragment(payload_json)
            return True, total_s, out[:240], ""
    except error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = "<no body>"
        return False, float("inf"), "", f"HTTP {exc.code}: {body[:240]}"
    except error.URLError as exc:
        return False, float("inf"), "", f"Network error: {exc.reason}"
    except Exception as exc:
        return False, float("inf"), "", str(exc)


def provider_name(model: str) -> str:
    if "/" not in model:
        return "unknown"
    provider, _ = model.split("/", 1)
    return provider or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grab_object.py for one model per provider (fastest responsive).")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Short prompt used for health check.")
    parser.add_argument("--max-tokens", type=int, default=8, help="Max tokens to request per model.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent requests when probing models.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional explicit list of models to test.")
    parser.add_argument("--grab-script", default=DEFAULT_GRAB_SCRIPT, help="Path to grab_object.py to run (default: repo-level grab_object.py).")
    parser.add_argument("--keep-output", action="store_true", help="Do not print subprocess stdout/stderr; instead stream it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_local_env()

    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NIM_API_KEY")
    if not api_key:
        print("Error: set NVIDIA_API_KEY or NIM_API_KEY before running this script.", file=sys.stderr)
        return 1

    try:
        models = args.models or discover_models(args.base_url.rstrip("/"), api_key, args.timeout)
    except Exception as exc:
        print(f"Failed to discover models: {exc}", file=sys.stderr)
        return 1

    print(f"Discovered {len(models)} models. Probing responsiveness with {max(1, args.workers)} worker(s)...")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {
            ex.submit(test_model_once, args.base_url.rstrip("/"), api_key, model, args.prompt, args.max_tokens, args.timeout): model
            for model in models
        }
        for fut in concurrent.futures.as_completed(futures):
            model = futures[fut]
            try:
                success, total_s, preview, err = fut.result()
            except Exception as exc:
                success, total_s, preview, err = False, float("inf"), "", str(exc)
            results[model] = (success, total_s, preview, err)
            status = "ok" if success else "failed"
            latency = f"{total_s:.3f}s" if total_s != float("inf") else "-"
            print(f"{model} -> {status} ({latency})")

    # group successful models by provider and pick fastest (lowest total_s)
    providers: dict[str, tuple[str, float, str]] = {}  # provider -> (model, total_s, preview)
    for model, (success, total_s, preview, err) in results.items():
        if not success:
            continue
        prov = provider_name(model)
        current = providers.get(prov)
        if current is None or total_s < current[1]:
            providers[prov] = (model, total_s, preview)

    if not providers:
        print("No responsive models were found.", file=sys.stderr)
        return 2

    print()
    print("Selected fastest model per provider:")
    for prov, (model, total_s, preview) in sorted(providers.items()):
        print(f"- {prov}: {model} ({total_s:.3f}s) preview: {preview}")

    # Run grab_object.py once per provider using selected model
    grab_script = args.grab_script
    if not Path(grab_script).exists():
        print(f"Warning: grab script not found at {grab_script}. Attempting to run anyway.", file=sys.stderr)

    print()
    for prov, (model, total_s, preview) in sorted(providers.items()):
        print(f"Running grab script for provider '{prov}' with model '{model}'...")
        cmd = [sys.executable, grab_script, "--model", model]
        try:
            if args.keep_output:
                proc = subprocess.run(cmd)
                rc = proc.returncode
            else:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                rc = proc.returncode
                out = proc.stdout.strip()
                print(f"-> exit={rc} output (first 600 chars):")
                print(out[:600])
        except FileNotFoundError:
            print(f"Failed to run grab script: {grab_script} not found", file=sys.stderr)
            rc = 127
        except Exception as exc:
            print(f"Failed to run grab script for {model}: {exc}", file=sys.stderr)
            rc = 1

        if rc != 0:
            print(f"grab script returned non-zero exit ({rc}) for model {model}", file=sys.stderr)
        else:
            print(f"grab script finished successfully for {model}")

    return 0


if __name__ ==#!/usr/bin/env python3
"""
Benchmark NVIDIA Build / API Catalog chat models by responsiveness.

The script tries to discover models from the OpenAI-compatible `/v1/models`
endpoint, sends a minimal chat request to each candidate, measures latency,
and prints the fastest models first.

Examples:
  python kinova_middleware/llm_clients/benchmark_nvidia_models.py
  python kinova_middleware/llm_clients/benchmark_nvidia_models.py --top 10 --workers 2
  python kinova_middleware/llm_clients/benchmark_nvidia_models.py --models deepseek-ai/deepseek-v3.1 meta/llama-3.1-8b-instruct
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_PROMPT = "Reply with exactly OK."


@dataclass
class BenchmarkResult:
    model: str
    success: bool
    median_total_s: float | None = None
    median_ttft_s: float | None = None
    attempts: int = 0
    output_preview: str = ""
    error_message: str = ""


def load_local_env() -> None:
    """Load variables from the repo's .env file if they are not already exported."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def build_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "kinova-nvidia-model-benchmark/1.0",
    }


def http_json(method: str, url: str, headers: dict[str, str], payload: dict[str, Any] | None, timeout: float) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, data=body, headers=headers, method=method)
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def first_choice(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            return first
    return {}


def extract_text_fragment(payload: dict[str, Any]) -> str:
    """Extract the most useful human-readable text from a chat completion payload."""
    choice = first_choice(payload)
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    delta = choice.get("delta")
    if isinstance(delta, dict):
        for key in ("content", "reasoning_content", "thinking"):
            value = delta.get(key)
            if isinstance(value, str) and value.strip():
                return value

    for key in ("output_text", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def discover_models(base_url: str, api_key: str, timeout: float) -> list[str]:
    """Fetch model ids from the OpenAI-compatible models endpoint."""
    data = http_json("GET", f"{base_url}/models", build_headers(api_key), None, timeout)
    items = data.get("data", []) if isinstance(data, dict) else []
    models = sorted({item.get("id", "").strip() for item in items if isinstance(item, dict) and item.get("id")})
    if not models:
        raise RuntimeError("No models were returned by /v1/models.")
    return models


def benchmark_single_request(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    measure_ttft: bool,
) -> tuple[float, float | None, str]:
    url = f"{base_url}/chat/completions"
    headers = build_headers(api_key)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": measure_ttft,
    }

    started = time.perf_counter()
    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with request.urlopen(req, timeout=timeout) as response:
        if not measure_ttft:
            payload = json.loads(response.read().decode("utf-8"))
            total_s = time.perf_counter() - started
            content = extract_text_fragment(payload)
            return total_s, None, content

        ttft_s = None
        chunks: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            event_data = line[len("data:") :].strip()
            if event_data == "[DONE]":
                break
            event = json.loads(event_data)
            if ttft_s is None:
                ttft_s = time.perf_counter() - started
            content_piece = extract_text_fragment(event)
            if content_piece:
                chunks.append(content_piece)

        total_s = time.perf_counter() - started
        return total_s, ttft_s, "".join(chunks).strip()


def benchmark_model(
    model: str,
    base_url: str,
    api_key: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    repeats: int,
    measure_ttft: bool,
) -> BenchmarkResult:
    totals: list[float] = []
    ttfts: list[float] = []
    preview = ""

    for _ in range(repeats):
        try:
            total_s, ttft_s, output_preview = benchmark_single_request(
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                timeout=timeout,
                measure_ttft=measure_ttft,
            )
            totals.append(total_s)
            if ttft_s is not None:
                ttfts.append(ttft_s)
            if output_preview and not preview:
                preview = output_preview
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            return BenchmarkResult(
                model=model,
                success=False,
                attempts=len(totals),
                error_message=f"HTTP {exc.code}: {body[:240]}",
            )
        except error.URLError as exc:
            return BenchmarkResult(
                model=model,
                success=False,
                attempts=len(totals),
                error_message=f"Network error: {exc.reason}",
            )
        except Exception as exc:  # noqa: BLE001
            return BenchmarkResult(
                model=model,
                success=False,
                attempts=len(totals),
                error_message=str(exc),
            )

    return BenchmarkResult(
        model=model,
        success=True,
        attempts=repeats,
        median_total_s=statistics.median(totals),
        median_ttft_s=statistics.median(ttfts) if ttfts else None,
        output_preview=preview[:80],
    )


def format_seconds(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}s"


def provider_name(model: str) -> str:
    if "/" not in model:
        return "unknown"
    provider, _ = model.split("/", 1)
    return provider or "unknown"


def benchmark_sort_key(item: BenchmarkResult, measure_ttft: bool) -> tuple[float, float, str]:
    primary = (
        item.median_ttft_s
        if measure_ttft and item.median_ttft_s is not None
        else item.median_total_s if item.median_total_s is not None else float("inf")
    )
    secondary = item.median_total_s if item.median_total_s is not None else float("inf")
    return (primary, secondary, item.model)


def print_ranked_table(title: str, items: list[BenchmarkResult], measure_ttft: bool) -> None:
    print()
    print(title)
    print("=" * 96)
    print(f"{'Rank':<6}{'Model':<48}{'TTFT':<12}{'Total':<12}{'Attempts':<10}Preview")
    print("-" * 96)
    for index, item in enumerate(items, start=1):
        print(
            f"{index:<6}{item.model[:47]:<48}"
            f"{format_seconds(item.median_ttft_s):<12}"
            f"{format_seconds(item.median_total_s):<12}"
            f"{item.attempts:<10}"
            f"{item.output_preview}"
        )


def print_results(results: list[BenchmarkResult], top: int, show_failures: bool, measure_ttft: bool) -> None:
    successful = [item for item in results if item.success]
    failed = [item for item in results if not item.success]

    successful.sort(key=lambda item: benchmark_sort_key(item, measure_ttft))

    print_ranked_table("Fastest responsive models", successful[:top], measure_ttft)

    grouped: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for item in successful:
        grouped[provider_name(item.model)].append(item)

    providers = sorted(
        grouped.items(),
        key=lambda pair: (
            benchmark_sort_key(pair[1][0], measure_ttft),
            pair[0],
        ),
    )

    print()
    print("Working models by provider")
    print("=" * 96)
    for provider, provider_models in providers:
        provider_models.sort(key=lambda item: benchmark_sort_key(item, measure_ttft))
        best = provider_models[0]
        print(
            f"{provider} ({len(provider_models)} working models, fastest "
            f"{format_seconds(best.median_ttft_s if measure_ttft else best.median_total_s)})"
        )
        print("-" * 96)
        for index, item in enumerate(provider_models, start=1):
            print(
                f"{index:<6}{item.model[:47]:<48}"
                f"{format_seconds(item.median_ttft_s):<12}"
                f"{format_seconds(item.median_total_s):<12}"
                f"{item.attempts:<10}"
                f"{item.output_preview}"
            )
        print()

    print()
    print(f"Successful benchmarks: {len(successful)}")
    print(f"Failed benchmarks: {len(failed)}")

    if show_failures and failed:
        print()
        print("Failures")
        print("=" * 96)
        for item in failed:
            print(f"- {item.model}: {item.error_message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark NVIDIA Build / API Catalog models.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible base URL. Default: %(default)s")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Short prompt used for benchmarking.")
    parser.add_argument("--max-tokens", type=int, default=8, help="Max tokens to request per model. Default: %(default)s")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds. Default: %(default)s")
    parser.add_argument("--top", type=int, default=20, help="How many of the fastest successful models to print. Default: %(default)s")
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent model tests. Default: %(default)s")
    parser.add_argument("--repeats", type=int, default=1, help="How many times to test each model. Default: %(default)s")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional explicit list of models. If omitted, the script calls /v1/models to discover them.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming. When omitted, the script measures time-to-first-token as responsiveness.",
    )
    parser.add_argument("--show-failures", action="store_true", help="Print models that failed and their error messages.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_local_env()

    api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("NIM_API_KEY")
    if not api_key:
        print("Error: set NVIDIA_API_KEY or NIM_API_KEY before running this script.", file=sys.stderr)
        return 1

    measure_ttft = not args.no_stream

    try:
        models = args.models or discover_models(args.base_url.rstrip("/"), api_key, args.timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to discover models: {exc}", file=sys.stderr)
        print("Tip: pass explicit model ids with --models if /v1/models is unavailable.", file=sys.stderr)
        return 1

    print(f"Discovered {len(models)} models.")
    print(
        f"Benchmarking with {'streaming TTFT + total time' if measure_ttft else 'total response time only'} "
        f"using {args.workers} worker(s)..."
    )

    results: list[BenchmarkResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_model = {
            executor.submit(
                benchmark_model,
                model,
                args.base_url.rstrip("/"),
                api_key,
                args.prompt,
                args.max_tokens,
                args.timeout,
                max(1, args.repeats),
                measure_ttft,
            ): model
            for model in models
        }

        completed = 0
        total = len(future_to_model)
        for future in concurrent.futures.as_completed(future_to_model):
            result = future.result()
            results.append(result)
            completed += 1
            status = "ok" if result.success else "failed"
            latency = format_seconds(result.median_ttft_s if measure_ttft else result.median_total_s)
            print(f"[{completed:>3}/{total}] {result.model} -> {status} ({latency})")

    print_results(results, top=max(1, args.top), show_failures=args.show_failures, measure_ttft=measure_ttft)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
