from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import requests

from src.hf_inference import MossMusicHFInference, read_env_model_id, resolve_device

TITLE = "MOSS-Music Demo"

DEFAULT_QUESTION = "请从风格与速度、调性与和声、乐器编配、结构安排以及整体情绪几个方面描述这段音乐。"
DEFAULT_JSON_PROMPT = (
    "请按照 music-story/0.2 音乐理解 JSON 规范分析这段音乐，"
    "只输出合法、完整的 JSON，不要输出 Markdown 或解释文字。"
)
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 50
STAGED_JSON_STEPS = (
    (
        "profile",
        """请用简短中文纯文本分析音乐整体画像、人声和主要乐器。
    不要输出 JSON、Markdown、嵌套列表或重复内容。最多回答 8 行，每行一个事实或判断。
    覆盖：类型、情绪、速度、调式、人声、主要乐器、编曲特点和一句整体描述。
    无法确认时明确写“无法确认”，不要猜测，不要写故事。""",
    ),
    (
        "timeline",
        """请用简短中文纯文本分析歌曲曲式和情绪时间线。
    不要输出 JSON、Markdown、嵌套列表或重复内容。最多回答 8 行，每行一个段落。
    每行尽量包含段落名称、起止时间、能量、情绪和听到的变化。
    无法可靠判断时间时写“时间无法确认”，不要猜测，不要写故事。""",
    ),
    (
        "lyrics",
        """请用简短中文纯文本分析歌词主题和可用于故事创作的意象。
    不要输出 JSON、Markdown、嵌套列表或重复内容。最多回答 8 行，每行一个主题或灵感。
    只输出确认的短歌词片段或摘要；不要完整转载歌词。
    明确区分听到的内容和可能的故事灵感，不要写完整故事。""",
    ),
    (
        "mapping",
        """请用简短中文纯文本说明音乐如何映射到故事结构。
    不要输出 JSON、Markdown、嵌套列表或重复内容。最多回答 8 行。
    覆盖开场、发展、转折、高潮、结尾分别适合由什么音乐段落承担，以及不确定项。
    不要写完整故事，不要把推测当成音乐事实。""",
    ),
)
STAGED_TEXT_LIMITS = {
    "profile": 768,
    "timeline": 1024,
    "lyrics": 1024,
    "mapping": 768,
}
VIDEO_EXTENSIONS = {".mp4"}
DEFAULT_BACKEND = "hf"
DEFAULT_SGLANG_BASE_URL = "http://127.0.0.1:30000"
DEFAULT_SGLANG_API_KEY = ""
DEFAULT_REQUEST_TIMEOUT = 600


@lru_cache(maxsize=2)
def get_inference(model_name_or_path: str, device: str) -> MossMusicHFInference:
    return MossMusicHFInference(
        model_name_or_path=model_name_or_path,
        device=device,
        enable_time_marker=True,
    )


def format_status(
    backend: str,
    model_name_or_path: str,
    target: str,
    elapsed_seconds: float,
) -> str:
    return (
        f"Backend: `{backend}`  \n"
        f"Model: `{model_name_or_path}`  \n"
        f"Target: `{target}`  \n"
        f"Elapsed: `{elapsed_seconds:.2f}s`"
    )


def read_backend() -> str:
    backend = os.environ.get("MOSS_MUSIC_BACKEND", DEFAULT_BACKEND).strip().lower()
    if backend in {"", "auto"}:
        return "hf"
    return backend


def normalize_sglang_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def read_sglang_base_url() -> str:
    base_url = os.environ.get("MOSS_MUSIC_SGLANG_BASE_URL", DEFAULT_SGLANG_BASE_URL)
    return normalize_sglang_base_url(base_url)


def read_sglang_api_key() -> str:
    return os.environ.get("MOSS_MUSIC_SGLANG_API_KEY", DEFAULT_SGLANG_API_KEY)


def read_request_timeout() -> int:
    return int(os.environ.get("MOSS_MUSIC_REQUEST_TIMEOUT", str(DEFAULT_REQUEST_TIMEOUT)))


def build_sglang_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = read_sglang_api_key().strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def extract_response_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("text"), str):
        return payload["text"].strip()
    if isinstance(payload.get("generated_text"), str):
        return payload["generated_text"].strip()

    choices = payload.get("choices") or []
    if choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            if isinstance(first_choice.get("text"), str):
                return first_choice["text"].strip()

            message = first_choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_chunks: list[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_chunks.append(item.get("text", ""))
                if text_chunks:
                    return "".join(text_chunks).strip()

    raise ValueError("The SGLang response did not contain a text message.")


def generate_with_sglang(
    question: str,
    media_path: str | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[str, str, str]:
    base_url = read_sglang_base_url()
    payload = {
        "text": question,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        },
    }
    if media_path:
        payload["audio_data"] = media_path

    response = requests.post(
        f"{base_url}/generate",
        headers=build_sglang_headers(),
        json=payload,
        timeout=read_request_timeout(),
    )
    response.raise_for_status()
    model_name_or_path = (
        os.environ.get("MOSS_MUSIC_SGLANG_MODEL", "").strip()
        or read_env_model_id()
        or "default"
    )
    return extract_response_text(response.json()), model_name_or_path, f"{base_url}/generate"


def convert_media_to_mp3(media_path: str, output_path: str) -> None:
    command = [
        "ffmpeg", "-y", "-i", media_path,
        "-vn", "-acodec", "libmp3lame", output_path,
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise gr.Error(
            "Failed to extract audio from the uploaded media.\n"
            f"{exc.stderr}"
        ) from exc


def resolve_media_path(audio_path: str | None, video_path: str | None) -> str | None:
    if video_path:
        return video_path
    return audio_path


def generate_answer(
    media_path: str | None,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[str, str, str, float]:
    backend = read_backend()
    started_at = time.perf_counter()
    target = ""
    model_name_or_path = ""

    def run_hf() -> tuple[str, str, str]:
        nonlocal model_name_or_path, target
        model_name_or_path = read_env_model_id()
        device = resolve_device()
        target = device
        inference = get_inference(model_name_or_path, device)
        answer = inference.generate(
            question=question,
            audio_path=prepared_audio_path,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return answer, model_name_or_path, target

    with tempfile.TemporaryDirectory(prefix="moss-music-") as temp_dir:
        prepared_audio_path = media_path
        if media_path and Path(media_path).suffix.lower() in VIDEO_EXTENSIONS:
            prepared_audio_path = os.path.join(temp_dir, "input.mp3")
            convert_media_to_mp3(media_path, prepared_audio_path)

        if backend == "sglang":
            try:
                answer, model_name_or_path, target = generate_with_sglang(
                    question=question,
                    media_path=prepared_audio_path,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
            except (requests.RequestException, ValueError, OSError):
                if os.environ.get("MOSS_MUSIC_BACKEND", "").strip().lower() == "sglang":
                    raise
                answer, model_name_or_path, target = run_hf()
        else:
            answer, model_name_or_path, target = run_hf()

    return answer, model_name_or_path, target, time.perf_counter() - started_at


def parse_json_answer(answer: str) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    candidate = answer.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    try:
        return json.loads(candidate, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(
                    candidate[start : end + 1],
                    object_pairs_hook=reject_duplicate_keys,
                )
            except json.JSONDecodeError:
                pass
        raise ValueError(
            "Model output is not valid complete JSON. The response may be truncated "
            "or structurally malformed; retry with temperature 0 and preferably "
            "split the analysis into smaller stages."
        ) from exc


def run_cli(args: argparse.Namespace) -> None:
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.staged_json:
        run_staged_cli(args, audio_path, output_path)
        return

    answer, model_name_or_path, target, elapsed_seconds = generate_answer(
        media_path=str(audio_path),
        question=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    try:
        result = parse_json_answer(answer)
    except ValueError as exc:
        raw_path = output_path.with_suffix(output_path.suffix + ".raw.txt")
        raw_path.write_text(answer, encoding="utf-8")
        raise ValueError(
            f"{exc} Raw model output was saved to: {raw_path} "
            f"({len(answer)} characters)."
        ) from exc
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"JSON written to: {output_path}")
    print(f"Model: {model_name_or_path}")
    print(f"Target: {target}")
    print(f"Elapsed: {elapsed_seconds:.2f}s")


def run_staged_cli(
    args: argparse.Namespace,
    audio_path: Path,
    output_path: Path,
) -> None:
    started_at = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": "music-story/0.2",
        "analysis_status": "completed",
        "source": {"file": audio_path.name, "path": str(audio_path)},
    }
    model_name_or_path = read_env_model_id()
    target = resolve_device()
    for stage_name, stage_prompt in STAGED_JSON_STEPS:
        prompt = (
            f"{stage_prompt}\n\n音频文件名：{audio_path.name}\n"
            "请只根据音频内容回答。"
        )
        answer, _, _, _ = generate_answer(
            media_path=str(audio_path),
            question=prompt,
            max_new_tokens=min(args.max_new_tokens, STAGED_TEXT_LIMITS[stage_name]),
            temperature=0.0,
            top_p=0.9,
            top_k=40,
        )
        result[f"{stage_name}_analysis"] = {
            "format": "plain_text",
            "content": answer.strip(),
        }
        print(f"Completed staged step: {stage_name}")

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"JSON written to: {output_path}")
    print(f"Model: {model_name_or_path}")
    print(f"Target: {target}")
    print(f"Elapsed: {time.perf_counter() - started_at:.2f}s")


def run_inference(
    audio_path: str | None,
    video_path: str | None,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
):
    prompt = (question or "").strip() or DEFAULT_QUESTION
    media_path = resolve_media_path(audio_path, video_path)
    backend = read_backend()

    try:
        answer, model_name_or_path, target, elapsed_seconds = generate_answer(
            media_path=media_path,
            question=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    except Exception as exc:  # pragma: no cover
        raise gr.Error(
            f"{backend} inference failed. Please make sure the uploaded file is readable "
            f"and the format is supported.\n{exc}"
        ) from exc

    return answer, format_status(backend, model_name_or_path, target, elapsed_seconds)


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(
        "A music-specialised variant of MOSS-Audio for caption, tagging, "
        "lyrics ASR, structural analysis, and musical reasoning."
    )

    with gr.Row():
        with gr.Column(scale=5):
            audio_input = gr.Audio(
                label="Audio",
                sources=["upload", "microphone"],
                type="filepath",
            )
            with gr.Accordion("Optional Video Input (.mp4)", open=False):
                gr.Markdown(
                    "Upload an mp4 only when needed. If a video is provided, "
                    "its audio track will be extracted and used for inference."
                )
                video_input = gr.File(
                    label="Video File",
                    file_types=[".mp4"],
                    type="filepath",
                )
            question_input = gr.Textbox(
                label="Prompt",
                lines=4,
                value=DEFAULT_QUESTION,
                placeholder=(
                    "For example: Describe the mood and instrumentation; "
                    "transcribe the lyrics; what is the key and tempo?"
                ),
            )

            with gr.Accordion("Advanced Settings", open=False):
                max_new_tokens_input = gr.Slider(
                    minimum=64, maximum=8192,
                    value=DEFAULT_MAX_NEW_TOKENS, step=32,
                    label="Max New Tokens",
                )
                temperature_input = gr.Slider(
                    minimum=0.0, maximum=1.5,
                    value=DEFAULT_TEMPERATURE, step=0.1,
                    label="Temperature",
                )
                top_p_input = gr.Slider(
                    minimum=0.1, maximum=1.0,
                    value=DEFAULT_TOP_P, step=0.05,
                    label="Top-p",
                )
                top_k_input = gr.Slider(
                    minimum=1, maximum=100,
                    value=DEFAULT_TOP_K, step=1,
                    label="Top-k",
                )

            with gr.Row():
                submit_btn = gr.Button("Generate", variant="primary")
                gr.ClearButton(
                    [
                        audio_input, video_input, question_input,
                        max_new_tokens_input, temperature_input,
                        top_p_input, top_k_input,
                    ],
                    value="Clear",
                )

        with gr.Column(scale=5):
            output_text = gr.Textbox(label="Output", lines=16)
            status_text = gr.Markdown("Waiting for input.")

    gr.Examples(
        examples=[
            ["Please give a detailed musical description of this clip."],
            ["Transcribe the lyrics with timestamps of this song."],
            ["What is the key, tempo and mood of this track?"],
            ["Transcribe the Chords progression with timestamps of this song, use json format to output the result."],
            ["Segment the song into verse / chorus / bridge sections."],
        ],
        inputs=[question_input],
        label="Prompt Examples",
    )

    submit_btn.click(
        fn=run_inference,
        inputs=[
            audio_input, video_input, question_input,
            max_new_tokens_input, temperature_input,
            top_p_input, top_k_input,
        ],
        outputs=[output_text, status_text],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MOSS-Music inference.")
    parser.add_argument("--audio", help="Input audio or MP4 path for JSON mode.")
    parser.add_argument("--output", help="Output JSON path for JSON mode.")
    parser.add_argument(
        "--staged-json",
        action="store_true",
        help="Run four smaller JSON analysis steps and merge their results.",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_JSON_PROMPT,
        help="Question or JSON analysis prompt."
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    if args.audio:
        if not args.output:
            parser.error("--output is required when --audio is provided")
        try:
            run_cli(args)
        except Exception as exc:
            parser.exit(1, f"Inference failed: {exc}\n")
        raise SystemExit(0)

    server_name = os.environ.get("MOSS_MUSIC_SERVER_NAME", "127.0.0.1")
    server_port = int(os.environ.get("MOSS_MUSIC_SERVER_PORT", "7860"))
    demo.queue(max_size=8).launch(
        server_name=server_name,
        server_port=server_port,
    )
