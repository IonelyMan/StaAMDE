import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
# 配置文件,需要自行设置API调用的一些信息，本文通过阿里云百炼大模型平台调用LLM
from config import DEFAULT_BASE_URL,DEFAULT_MODEL, DASHSCOPE_API_KEY
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase3 import llm_evidence

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LLM evidence and call Aliyun DashScope through the OpenAI Python SDK."
    )
    parser.add_argument("--sample-id", default=None, help="APK sample id. If omitted, choose the highest-risk available prediction.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="llm_explain")
    parser.add_argument("--static-predictions", default="inference/lightgbm/predictions/test_predictions.csv")
    parser.add_argument("--graph-predictions", default="inference/tasgatv2/predictions/test_predictions.csv")
    parser.add_argument("--ensemble-predictions", default="ensemble/ensemble_predictions.csv")
    parser.add_argument("--ml-explain-dir", default="ml_explain", help="机器学习解释结果根目录")
    parser.add_argument("--top-static-positive", type=int, default=15)
    parser.add_argument("--top-static-negative", type=int, default=5)
    parser.add_argument("--top-graph-nodes", type=int, default=15)
    parser.add_argument("--top-graph-edges", type=int, default=10)
    parser.add_argument("--max-name-len", type=int, default=320)

    parser.add_argument("--api-key", default=None, help="Aliyun DashScope API key. Defaults to DASHSCOPE_API_KEY or ALIYUN_API_KEY.")
    parser.add_argument("--base-url", default=None, help=f"OpenAI-compatible base URL. Default: {DEFAULT_BASE_URL}")
    parser.add_argument("--model", default=None, help=f"Model name. Default: {DEFAULT_MODEL}")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--system-prompt",
        default="你是严谨的安卓恶意软件分析专家。基于给定的模型证据解释；若有联网能力，优先核对 Android 官方文档。不要编造未知事实或来源。",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only build evidence/prompt and print token estimate; do not call the API.")
    return parser.parse_args()


def resolve_api_key(args: argparse.Namespace) -> str:
    # api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY")
    api_key = DASHSCOPE_API_KEY
    if not api_key:
        raise SystemExit(
            "Missing API key. Set DASHSCOPE_API_KEY or ALIYUN_API_KEY, "
            "or pass --api-key. The key will not be written to output files."
        )
    return api_key


def resolve_base_url(args: argparse.Namespace) -> str:
    base_url = args.base_url or os.getenv("DASHSCOPE_BASE_URL") or os.getenv("ALIYUN_BASE_URL") or DEFAULT_BASE_URL
    base_url = base_url.rstrip("/")
    suffix = "/chat/completions"
    if base_url.endswith(suffix):
        base_url = base_url[: -len(suffix)]
    return base_url


def resolve_model(args: argparse.Namespace) -> str:
    return args.model or os.getenv("DASHSCOPE_MODEL") or os.getenv("ALIYUN_MODEL") or DEFAULT_MODEL


def build_evidence(args: argparse.Namespace) -> Tuple[str, Dict[str, object], str, Path]:
    prediction_paths = {
        "static": Path(args.static_predictions),
        "graph": Path(args.graph_predictions),
        "ensemble": Path(args.ensemble_predictions),
    }
    indexed_predictions = llm_evidence.index_predictions(prediction_paths)
    sample_id = llm_evidence.choose_sample_id(indexed_predictions, args.sample_id)
    modalities = indexed_predictions.get(sample_id, {})
    apk_name = (modalities.get("ensemble") or modalities.get("static") or modalities.get("graph") or {}).get("apk_name") or f"{sample_id}.apk"
    ml_explain_dir = Path(args.ml_explain_dir)
    sample_dir = ml_explain_dir / llm_evidence.safe_filename(str(apk_name))

    static_local, static_warnings = llm_evidence.load_static_local_evidence(
        sample_dir,
        sample_id,
        args.top_static_positive,
        args.top_static_negative,
        args.max_name_len,
    )
    graph_evidence, graph_warnings = llm_evidence.load_graph_evidence(
        sample_dir,
        sample_id,
        args.top_graph_nodes,
        args.top_graph_edges,
        args.max_name_len,
    )

    evidence = {
        "prediction": llm_evidence.build_prediction_summary(sample_id, modalities),
        "static_local_shap": static_local,
        "graph_attention": graph_evidence,
        "caveats": [
            "SHAP explains how static features changed the LightGBM malware score; it is not proof of malicious intent by itself.",
            "Graph attention indicates model focus in the graph classifier; high-attention nodes are candidates for review, not guaranteed malicious code.",
            "Obfuscated method/class names and incomplete failed extractions can reduce interpretability.",
            "API and permission semantics should be verified against authoritative documentation before behavioral inference.",
        ],
        "warnings": static_warnings + graph_warnings,
    }
    prompt = llm_evidence.build_prompt(evidence)
    output_dir = Path(args.output) / llm_evidence.safe_filename(str(apk_name))
    output_dir.mkdir(parents=True, exist_ok=True)
    return sample_id, evidence, prompt, output_dir


def count_tokens_with_tiktoken(messages: List[Dict[str, str]], model: str) -> Optional[int]:
    try:
        import tiktoken
    except ImportError:
        return None

    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    # ChatML overhead varies by provider/model; this is a practical estimate before the API returns usage.
    total = 0
    for message in messages:
        total += 4
        total += len(encoding.encode(str(message.get("role", ""))))
        total += len(encoding.encode(str(message.get("content", ""))))
    return total + 2


def estimate_tokens_fallback(messages: List[Dict[str, str]]) -> int:
    text = "\n".join(f"{message.get('role', '')}: {message.get('content', '')}" for message in messages)
    cjk_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ascii_chars = len(text) - cjk_chars
    return int(cjk_chars * 1.15 + ascii_chars / 4.0) + 16


def estimate_input_tokens(messages: List[Dict[str, str]], model: str) -> Tuple[int, str]:
    token_count = count_tokens_with_tiktoken(messages, model)
    if token_count is not None:
        return token_count, "tiktoken_estimate"
    return estimate_tokens_fallback(messages), "fallback_estimate"


def completion_to_dict(completion: object) -> Dict[str, object]:
    if hasattr(completion, "model_dump"):
        try:
            return completion.model_dump(mode="json")  # type: ignore[attr-defined]
        except TypeError:
            return completion.model_dump()  # type: ignore[attr-defined]
    if hasattr(completion, "to_dict"):
        return completion.to_dict()  # type: ignore[attr-defined]
    if hasattr(completion, "model_dump_json"):
        return json.loads(completion.model_dump_json())  # type: ignore[attr-defined]
    raise TypeError(f"Unsupported OpenAI SDK response type: {type(completion)!r}")


def call_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> Dict[str, object]:
    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: openai. Install it with `pip install -r requirements.txt` "
            "or `pip install openai`."
        ) from exc

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except APIStatusError as exc:
        response_text = getattr(exc.response, "text", "")
        raise SystemExit(f"Aliyun OpenAI-compatible API error {exc.status_code}: {response_text}") from exc
    except (APIConnectionError, APITimeoutError) as exc:
        raise SystemExit(f"Aliyun OpenAI-compatible API request failed: {exc}") from exc

    return completion_to_dict(completion)


def extract_reply(response: Dict[str, object]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and message.get("content") is not None:
                return str(message["content"])
            if first.get("text") is not None:
                return str(first["text"])
    return json.dumps(response, ensure_ascii=False, indent=2)


def redacted_config(args: argparse.Namespace, base_url: str, model: str, token_count: int, token_method: str) -> Dict[str, object]:
    return {
        "sdk": "openai-python",
        "base_url": base_url,
        "model": model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "input_tokens": token_count,
        "input_token_count_method": token_method,
        "api_key_source": "argument/env redacted",
        "sample_id": args.sample_id,
        "split": args.split,
        "top_static_positive": args.top_static_positive,
        "top_static_negative": args.top_static_negative,
        "top_graph_nodes": args.top_graph_nodes,
        "top_graph_edges": args.top_graph_edges,
    }


def main() -> None:
    args = parse_args()
    sample_id, evidence, prompt, output_dir = build_evidence(args)
    llm_evidence.write_json(output_dir / "llm_evidence.json", evidence)
    (output_dir / "llm_prompt.md").write_text(prompt, encoding="utf-8")

    base_url = resolve_base_url(args)
    model = resolve_model(args)
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": prompt})

    token_count, token_method = estimate_input_tokens(messages, model)
    print(f"sample_id: {sample_id}")
    print(f"prompt: {output_dir / 'llm_prompt.md'}")
    print(f"evidence: {output_dir / 'llm_evidence.json'}")
    print(f"input_tokens ({token_method}): {token_count}")

    config = redacted_config(args, base_url, model, token_count, token_method)
    # llm_evidence.write_json(output_dir / "llm_call_config.json", config)

    if args.dry_run:
        print("dry-run enabled; API call skipped.")
        return

    api_key = resolve_api_key(args)
    start = time.time()
    response = call_chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    elapsed = time.time() - start
    reply = extract_reply(response)

    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    if usage:
        print(json.dumps({"provider_usage": usage}, ensure_ascii=False, indent=2))
    print(f"elapsed_seconds: {elapsed:.2f}")

    # llm_evidence.write_json(output_dir / "llm_raw_response.json", response)
    (output_dir / "llm_response.md").write_text(reply, encoding="utf-8")
    response_meta = {
        "sample_id": sample_id,
        "model": model,
        "base_url": base_url,
        "elapsed_seconds": elapsed,
        "input_tokens_estimated": token_count,
        "input_token_count_method": token_method,
        "provider_usage": usage,
        "response_path": str(output_dir / "llm_response.md"),
        "raw_response_path": str(output_dir / "llm_raw_response.json"),
    }
    # llm_evidence.write_json(output_dir / "llm_response_meta.json", response_meta)
    print(f"response: {output_dir / 'llm_response.md'}")
    print(f"raw_response: {output_dir / 'llm_raw_response.json'}")


if __name__ == "__main__":
    main()
