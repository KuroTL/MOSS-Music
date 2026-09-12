from __future__ import annotations

import os
from pathlib import Path


def _configure_ffmpeg_dlls() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    candidates = []
    configured_path = os.environ.get("FFMPEG_PATH")
    if configured_path:
        candidates.append(Path(configured_path))

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.extend(
            Path(local_app_data).glob(
                "Microsoft/WinGet/Packages/Gyan.FFmpeg.Shared_*/"
                "ffmpeg-*-full_build-shared/bin"
            )
        )

    for candidate in candidates:
        if candidate.is_dir() and (candidate / "ffmpeg.exe").exists():
            os.add_dll_directory(str(candidate))
            return


_configure_ffmpeg_dlls()

import torchaudio


def load_audio(path: str, sample_rate: int):
    waveform, original_sample_rate = torchaudio.load(path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if original_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=original_sample_rate, new_freq=sample_rate
        )
    return waveform.squeeze(0).cpu().numpy()
