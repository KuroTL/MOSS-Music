# MOSS-Music Windows 使用指南

本文档记录在 Windows 上使用 MOSS-Music 的完整流程。推荐使用项目的 `.venv` 环境和本地 HF Transformers 推理方式。

## 1. 运行方式说明

项目支持两种运行方式：

| 方式 | Windows 原生支持 | 说明 |
| --- | --- | --- |
| HF Transformers | 支持 | 直接由 `app.py` 加载本地模型，适合 Windows 单机运行 |
| SGLang | 不建议原生 Windows | `sgl-kernel` 目前没有 `win_amd64` wheel，建议在 WSL2/Linux 中运行 |

本文默认使用 HF 模式，不需要启动 SGLang 服务。

## 2. 前置条件

建议配置：

- Windows 10/11
- NVIDIA GPU，建议显存至少 12 GB
- Python 3.12
- CUDA 12.8 对应的 PyTorch
- 项目已经下载好的 MOSS-Music 模型
- 可用的 FFmpeg shared DLL

项目根目录应为：

```text
D:\01_Project\01_Music2Story\MOSS-Music
```

进入项目目录：

```powershell
cd D:\01_Project\01_Music2Story\MOSS-Music
.\.venv\Scripts\python.exe -c "import os; os.add_dll_directory(os.environ['FFMPEG_PATH']); import torchcodec; print('torchcodec + FFmpeg OK')"

## 3. 创建并激活 `.venv`

如果项目还没有 `.venv`：

```powershell
uv venv --python 3.12 .venv
```

激活环境：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

看到下面类似提示，说明环境已激活：

```text
(.venv) PS D:\01_Project\01_Music2Story\MOSS-Music>
```

如果不想激活，也可以始终使用完整解释器路径：

```powershell
.\.venv\Scripts\python.exe
```

## 4. 安装项目依赖

先安装项目本身及基础依赖：

```powershell
uv pip install --python .venv\Scripts\python.exe -e .
```

安装项目要求的 CUDA 12.8 PyTorch 运行时：

```powershell
uv pip install --python .venv\Scripts\python.exe `
  --index-url https://download.pytorch.org/whl/cu128 `
  "torch==2.9.1+cu128" `
  "torchaudio==2.9.1+cu128" `
  "torchvision==0.24.1+cu128"
```

安装 Transformers、量化和音频依赖：

```powershell
uv pip install --python .venv\Scripts\python.exe `
  "transformers==4.57.1" `
  "huggingface-hub==0.36.0" `
  "torchcodec==0.9.*" `
  "bitsandbytes"
```

这里的版本需要保持一致：

```text
torch       2.9.1+cu128
torchaudio  2.9.1+cu128
torchvision 0.24.1+cu128
torchcodec  0.9.x
transformers 4.57.1
huggingface-hub 0.36.0
```

## 5. 安装 FFmpeg

### 5.1 Windows 推荐方式

README 使用的是 Conda：

```bash
conda install -c conda-forge "ffmpeg=7" -y
```

如果当前使用 `.venv` 而不是 Conda，可以使用 Windows shared FFmpeg：

```powershell
winget install --id Gyan.FFmpeg.Shared --exact --version 8.1.2 --scope user --accept-source-agreements --accept-package-agreements
```

TorchCodec 0.9 支持 FFmpeg 4、5、6、7、8，因此使用 shared FFmpeg 8.1.2 是兼容的。不要安装 FFmpeg 9，因为 TorchCodec 0.9 不支持 FFmpeg 9。

安装完成后，关闭并重新打开 VS Code 终端。如果仍然找不到 `ffmpeg`，可以在当前终端临时加入路径：

```powershell
$ffmpeg = Get-ChildItem `
  "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" `
  -Filter ffmpeg.exe `
  -Recurse `
  -ErrorAction SilentlyContinue `
  | Where-Object { $_.FullName -match 'FFmpeg.Shared' } `
  | Select-Object -First 1

$env:Path = "$($ffmpeg.Directory.FullName);$env:Path"
$env:FFMPEG_PATH = $ffmpeg.Directory.FullName
```

检查 FFmpeg：

```powershell
ffmpeg -version
```

## 6. 检查模型文件

本项目当前使用的本地模型路径是：

```text
D:\01_Project\01_Music2Story\MOSS-Music\weights\models\OpenMOSS--MOSS-Music-8B-Instruct\snapshots\master
```

该目录至少应包含：

```text
config.json
model.safetensors.index.json
model-00001-of-00004.safetensors
model-00002-of-00004.safetensors
model-00003-of-00004.safetensors
model-00004-of-00004.safetensors
chat_template.jinja
```

PowerShell 检查命令：

```powershell
$modelPath = Resolve-Path `
  ".\weights\models\OpenMOSS--MOSS-Music-8B-Instruct\snapshots\master"

$modelPath.Path
Get-ChildItem $modelPath -Filter "model-*.safetensors"
```

如果缺少模型分片，需要先完整下载模型。不要在推理时临时下载，以免程序看起来长时间无响应。

## 7. 运行环境检查

### 7.1 检查 PyTorch 和 CUDA

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"
```

预期结果类似：

```text
torch: 2.9.1+cu128
cuda: True
cuda version: 12.8
```

如果 `cuda: False`，HF 推理可能会退回 CPU，8B 模型会非常慢或无法加载。

### 7.2 检查 Transformers 和量化配置

```powershell
.\.venv\Scripts\python.exe -c "from transformers import BitsAndBytesConfig; print('transformers and BitsAndBytesConfig OK')"
```

项目使用 `BitsAndBytesConfig` 开启 4bit 量化，不再把已弃用的 `load_in_4bit=True`
直接作为 `from_pretrained()` 的顶层参数传入。`load_in_4bit=True` 仍会作为
`BitsAndBytesConfig` 的字段出现，这是当前推荐写法。

### 7.3 量化配置说明

本项目加载 8B 模型时使用 4bit bitsandbytes 量化，实际配置如下：

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_compute_dtype=torch.bfloat16,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True,
)

model = MossMusicModel.from_pretrained(
  model_path,
  trust_remote_code=True,
  dtype=torch.bfloat16,
  quantization_config=quantization_config,
  device_map="auto",
)
```

各参数作用：

| 参数 | 作用 |
| --- | --- |
| `load_in_4bit=True` | 将模型权重以 4bit 形式加载，显著降低显存占用 |
| `bnb_4bit_compute_dtype=torch.bfloat16` | 计算时使用 bfloat16，在支持的 NVIDIA GPU 上兼顾速度和稳定性 |
| `bnb_4bit_quant_type="nf4"` | 使用 NF4 量化类型，适合加载预训练模型 |
| `bnb_4bit_use_double_quant=True` | 对量化常数再次量化，进一步节省显存 |
| `dtype=torch.bfloat16` | 指定非量化计算部分和模型计算使用 bfloat16 |
| `device_map="auto"` | 由 Transformers/Accelerate 自动将模型分配到可用设备 |

需要区分两种写法：

```python
# 推荐写法：放在 BitsAndBytesConfig 中
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# 不推荐写法：直接传给 from_pretrained，当前会产生弃用警告
MossMusicModel.from_pretrained(model_path, load_in_4bit=True)
```

因此日志中即使看到 `load_in_4bit` 这个字段，也不一定表示配置错误。只要它位于
`BitsAndBytesConfig` 中，就是当前 Transformers 推荐的配置方式。

4bit 量化并不等于总显存只占模型参数大小。运行时还需要为音频编码、激活值、KV
Cache、CUDA 上下文和生成过程保留显存。12 GB 显存可以尝试运行，但实际占用取决于
输入音频长度、`max_new_tokens` 和其他进程占用。若出现 CUDA out of memory，可先关闭
其他 GPU 程序、缩短音频、降低 `max_new_tokens`，再重试。

量化加载还依赖：

- NVIDIA GPU 和可用 CUDA 环境；
- `bitsandbytes`；
- `accelerate`；
- 与当前 PyTorch/CUDA 组合兼容的 Transformers 版本。

量化配置本身的快速检查：

```powershell
.\.venv\Scripts\python.exe -c "import torch; from transformers import BitsAndBytesConfig; from bitsandbytes import __version__ as bnb_version; print(torch.__version__); print('bitsandbytes:', bnb_version); print(BitsAndBytesConfig(load_in_4bit=True))"
```

### 7.4 检查 FFmpeg、TorchCodec 和 MP3

```powershell
.\.venv\Scripts\python.exe -c "import os; os.add_dll_directory(os.environ['FFMPEG_PATH']); import torchcodec; print('torchcodec + FFmpeg OK')"
```

使用项目内测试音频验证完整读取路径：

```powershell
.\.venv\Scripts\python.exe -c "import os; os.add_dll_directory(os.environ['FFMPEG_PATH']); from src.audio_io import load_audio; audio=load_audio('test/tonghua.mp3', 16000); print(f'audio loaded: shape={audio.shape}, dtype={audio.dtype}')"
```

预期结果类似：

```text
audio loaded: shape=(3897003,), dtype=float32
```

## 8. 启动 HF Gradio 应用

打开一个 PowerShell，进入项目目录并激活环境：

```powershell
cd D:\01_Project\01_Music2Story\MOSS-Music
.\.venv\Scripts\Activate.ps1
```

设置 HF 后端和本地模型路径：

```powershell
$env:MOSS_MUSIC_BACKEND = "hf"
$env:MOSS_MUSIC_MODEL_ID = (Resolve-Path `
  ".\weights\models\OpenMOSS--MOSS-Music-8B-Instruct\snapshots\master"
).Path
```

如果当前终端还没有 FFmpeg 路径，执行第 5 节的临时 PATH 命令，然后启动：

```powershell
python app.py
```

不激活环境时，使用完整命令：

```powershell
.\.venv\Scripts\python.exe app.py
```

启动成功后会看到：

```text
* Running on local URL:  http://127.0.0.1:7860
```

浏览器打开：

```text
http://127.0.0.1:7860
```

注意：Gradio 页面启动不代表模型已经加载完成。第一次点击推理时，程序才会加载 8B 模型，可能需要等待几十秒并占用较多显存。

## 9. 命令行输入音频并输出 JSON 文件

除了网页模式，`app.py` 也支持命令行模式。命令行模式要求提供音频路径和输出 JSON
路径，推理结果会直接写入指定目录，不需要打开 Gradio 页面。

先设置本地 HF 后端和模型路径：

```powershell
$env:MOSS_MUSIC_BACKEND = "hf"
$env:MOSS_MUSIC_MODEL_ID = (Resolve-Path `
  ".\weights\models\OpenMOSS--MOSS-Music-8B-Instruct\snapshots\master"
).Path
```

然后执行：

```powershell
python .\app.py `
  --audio ".\test\tonghua.mp3" `
  --output ".\output\tonghua_music_analysis.json" `
  --prompt "请按照音乐故事 JSON 规范分析这段音乐，只输出合法完整 JSON。" `
  --max-new-tokens 4096 `
  --temperature 0 `
  --top-p 0.9 `
  --top-k 40
```

也可以使用绝对路径：

```powershell
python .\app.py `
  --audio "D:\01_Project\01_Music2Story\MOSS-Music\test\tonghua.mp3" `
  --output "D:\01_Project\01_Music2Story\MOSS-Music\output\tonghua_music_analysis.json" `
  --prompt "请输出完整的 music-story/0.2 音乐理解 JSON。" `
  --max-new-tokens 4096 `
  --temperature 0
```

参数说明：

| 参数 | 是否必需 | 说明 |
| --- | --- | --- |
| `--audio` | 是 | 输入 `.mp3`、`.wav` 或 `.mp4` 文件路径；MP4 会先提取音频 |
| `--output` | 是 | 输出 JSON 文件路径，父目录不存在时会自动创建 |
| `--prompt` | 否 | 音乐分析问题，默认使用普通音乐描述提示词 |
| `--max-new-tokens` | 否 | 输出长度上限，完整 JSON 建议 `4096` 或更高 |
| `--temperature` | 否 | JSON 输出建议设为 `0`，降低重复键和截断风险 |
| `--top-p` | 否 | 采样范围，默认 `1.0` |
| `--top-k` | 否 | 采样候选数量，默认 `50` |

成功后会输出类似信息：

```text
JSON written to: ...\output\tonghua_music_analysis.json
Model: ...
Target: cuda:0
Elapsed: ...s
```

检查 JSON 文件：

```powershell
Get-Content ".\output\tonghua_music_analysis.json" -Raw | ConvertFrom-Json
```

如果命令行提示 `Model output is not valid complete JSON`，说明模型输出被截断或包含额外
文本。请依次尝试：

1. 将 `--max-new-tokens` 提高到 `6144` 或 `8192`；
2. 保持 `--temperature 0`；
3. 缩短提示词；
4. 使用分阶段提示词分别生成全局画像、曲式时间线、歌词意象和故事映射。

### 9.1 推荐：分阶段文本分析后封装 JSON

验证表明，MOSS-Music 直接生成复杂嵌套 JSON 时容易出现重复键、无限重复字段和不闭合
括号。因此 `--staged-json` 不再要求模型生成 JSON，而是让模型输出四段受限的短文本，
由 Python 程序确定性封装成合法 JSON。这样得到的是稳定的中间分析文件，后续可交给 Qwen3
等文本模型做第二次结构化。

程序会依次执行四个较小任务：

1. 全局音乐画像、人声和乐器；
2. 曲式时间线和情绪弧线；
3. 歌词意象和故事创作素材；
4. 故事段落映射和不确定性。

每个阶段完成后，程序将文本保存到对应字段，再合并成一个 JSON 文件。输出中的字段形式为：

```json
{
  "profile_analysis": {"format": "plain_text", "content": "..."},
  "timeline_analysis": {"format": "plain_text", "content": "..."},
  "lyrics_analysis": {"format": "plain_text", "content": "..."},
  "mapping_analysis": {"format": "plain_text", "content": "..."}
}
```

如果模型调用失败，会保存对应原始响应，文件名类似：

```text
output\Ado_好きでいて_music_analysis.json.timeline.raw.txt
```

运行命令：

```powershell
python .\app.py `
  --audio "C:\Users\17853\Downloads\Ado - 好きでいて_723073984.mp3" `
  --output ".\output\Ado_好きでいて_music_analysis.json" `
  --staged-json `
  --max-new-tokens 2048
```

分阶段模式会固定使用 `Temperature=0`，并限制每段输出长度。它会重复调用同一个已缓存的
模型，因此总耗时可能比单次请求更长，但比一次生成 8000 多 token 的复杂嵌套 JSON 稳定。
如果后续需要完整 `music-story/0.2` 字段，应将这个中间 JSON 交给专门的文本结构化模型，
不要再次要求 MOSS-Music 直接补齐复杂嵌套结构。

## 10. 使用页面推理

1. 打开 `http://127.0.0.1:7860`。
2. 上传 `.mp3`、`.wav` 或其他可由 FFmpeg/TorchCodec 解码的音频。
3. 输入问题，例如：

```text
请从风格与速度、调性与和声、乐器编配、结构安排以及整体情绪几个方面描述这段音乐。
```

4. 点击提交按钮。
5. 第一次请求等待模型加载和音频处理完成。

测试音频位于：

```text
test\tonghua.mp3
test\keximeiruguo - 02.mp3
test\zhishaohaiyouni-intro+verse1.mp3
```

## 11. 停止服务

在运行 `app.py` 的终端按：

```text
Ctrl+C
```

出现 `KeyboardInterrupt`、`Keyboard interruption in main thread` 或 Gradio/Uvicorn 收尾堆栈，通常表示服务被手动停止，不一定是新的错误。

## 12. 常见错误

### 11.1 `ModuleNotFoundError: No module named 'gradio'`

使用了系统 Python，而不是项目 `.venv`。改用：

```powershell
.\.venv\Scripts\python.exe app.py
```

### 11.2 `No module named 'transformers'`

安装 Transformers：

```powershell
uv pip install --python .venv\Scripts\python.exe "transformers==4.57.1"
```

### 11.3 `huggingface-hub>=0.34.0,<1.0 is required`

当前 Hub 版本过高，执行：

```powershell
uv pip install --python .venv\Scripts\python.exe "huggingface-hub==0.36.0"
```

### 11.4 `TorchCodec is required for load_with_torchcodec`

安装 TorchCodec：

```powershell
uv pip install --python .venv\Scripts\python.exe "torchcodec==0.9.*"
```

### 11.5 `Could not load libtorchcodec`

依次检查：

1. 使用 shared FFmpeg，而不是只安装命令行可执行文件的版本。
2. FFmpeg 版本为 4-8，不要使用 FFmpeg 9。
3. 重启 VS Code 终端，使新的 PATH 生效。
4. 在当前 PowerShell 中设置 `$env:FFMPEG_PATH` 并执行 `os.add_dll_directory` 验证。
5. Torch 使用项目指定版本 `2.9.1+cu128`。

### 11.6 `operator torchvision::nms does not exist`

Torch 和 TorchVision 版本不匹配。安装对应版本：

```powershell
uv pip install --python .venv\Scripts\python.exe `
  --index-url https://download.pytorch.org/whl/cu128 `
  "torchvision==0.24.1+cu128"
```

### 11.7 `HTTPConnectionPool(host='127.0.0.1', port=30100)`

程序正在使用 SGLang 后端，但 SGLang 服务没有启动，或端口不一致。Windows HF 模式下执行：

```powershell
$env:MOSS_MUSIC_BACKEND = "hf"
```

当前代码默认的 SGLang 地址为：

```text
http://127.0.0.1:30000
```

### 11.8 `KeyboardInterrupt` 出现在 `snapshot_download` 或 `xet_get`

这表示 Transformers 正在从 Hugging Face 下载缺失模型文件，之后被手动中断。设置本地模型路径：

```powershell
$env:MOSS_MUSIC_MODEL_ID = (Resolve-Path `
  ".\weights\models\OpenMOSS--MOSS-Music-8B-Instruct\snapshots\master"
).Path
```

### 11.9 `sglang : 无法将“sglang”项识别为命令`

当前 Windows `.venv` 没有可用的 SGLang CLI。仓库内 SGLang 安装通常会因为 `sgl-kernel` 缺少 Windows wheel 而失败。请使用 HF 模式，或者在 WSL2/Linux 中运行 SGLang。

## 13. WSL2/Linux 下使用 SGLang

如果必须使用 SGLang，建议在 WSL2/Linux 中执行，而不是 Windows PowerShell：

```bash
cd /mnt/d/01_Project/01_Music2Story/MOSS-Music

pip install -e "sglang/python[all]"
pip install nvidia-cudnn-cu12==9.16.0.29

sglang serve \
  --model-path "/mnt/d/01_Project/01_Music2Story/MOSS-Music/weights/models/OpenMOSS--MOSS-Music-8B-Instruct/snapshots/master" \
  --trust-remote-code \
  --port 30000
```

SGLang 服务启动后，另开终端运行 Gradio：

```bash
cd /mnt/d/01_Project/01_Music2Story/MOSS-Music
export MOSS_MUSIC_BACKEND=sglang
python app.py
```

Windows 原生 HF 应用不要同时依赖 SGLang 服务；两种模式请选择一种。

## 14. 最短可用流程

在已经完成依赖安装、模型文件和 FFmpeg 配置的前提下，每次启动只需：

```powershell
cd D:\01_Project\01_Music2Story\MOSS-Music
.\.venv\Scripts\Activate.ps1

$env:MOSS_MUSIC_BACKEND = "hf"
$env:MOSS_MUSIC_MODEL_ID = (Resolve-Path `
  ".\weights\models\OpenMOSS--MOSS-Music-8B-Instruct\snapshots\master"
).Path

python app.py
```

然后访问：

```text
http://127.0.0.1:7860
```

## 15. 本项目的代码修改说明

下面列出本次 Windows HF 运行链路中涉及的代码修改，便于后续排查或恢复。

### 14.1 `src/hf_inference.py`

模型加载从直接传递4bit参数调整为 `BitsAndBytesConfig`：

```python
from transformers import BitsAndBytesConfig
```

在 `MossMusicHFInference.__init__` 中创建量化配置，并传入：

```python
quantization_config=quantization_config
```

同时使用：

```python
dtype=torch.bfloat16
device_map="auto"
```

这样做解决了 Transformers 关于 `load_in_4bit` / `load_in_8bit` 顶层参数即将移除的警告，
并保留了 4bit 量化和自动设备分配能力。

该文件还负责：

- 读取 `MOSS_MUSIC_MODEL_ID`；
- 创建模型和处理器；
- 将输入移动到模型设备；
- 调用模型生成文本；
- 将生成 token 解码为最终回答。

### 14.2 `src/audio_io.py`

音频读取使用项目依赖的 `torchaudio`：

```python
waveform, original_sample_rate = torchaudio.load(path)
```

然后执行：

- 多声道转单声道；
- 重采样到模型要求的采样率（通常为 16 kHz）；
- 转换为 CPU 上的 NumPy 数组供处理器使用。

在当前 TorchAudio 版本下，MP3 解码会通过 TorchCodec 使用 FFmpeg shared DLL，因此必须
同时满足 Torch、TorchAudio、TorchCodec 和 FFmpeg 的版本要求。

### 14.3 `app.py`

应用增加并使用了两个运行时选择：

```powershell
$env:MOSS_MUSIC_BACKEND = "hf"
$env:MOSS_MUSIC_MODEL_ID = "本地模型目录"
```

其中：

- `MOSS_MUSIC_BACKEND=hf` 使请求走本地 Transformers 模型，而不是 SGLang HTTP 服务；
- `MOSS_MUSIC_MODEL_ID` 指向本地模型快照，避免运行时从 Hugging Face 重新下载；
- SGLang 默认地址为 `http://127.0.0.1:30000`，与项目使用指南一致；
- HF 模式不需要运行 30000 端口的 SGLang 服务。

### 14.4 为什么不直接修改 `infer.py`

`infer.py` 是独立的命令行示例，适合快速测试单个音频；Gradio 页面实际调用的是：

```text
app.py -> src/hf_inference.py -> src/audio_io.py
```

因此量化、模型路径和页面推理相关修改应优先检查这条调用链。

### 14.5 代码修改后的验证顺序

建议按以下顺序验证，不要一开始就直接点击页面推理：

```powershell
# 1. 检查语法
.\.venv\Scripts\python.exe -m py_compile src\audio_io.py src\hf_inference.py app.py

# 2. 检查音频解码
.\.venv\Scripts\python.exe -c "import os; os.add_dll_directory(os.environ['FFMPEG_PATH']); from src.audio_io import load_audio; audio=load_audio('test/tonghua.mp3', 16000); print(audio.shape, audio.dtype)"

# 3. 设置 HF 本地模型
$env:MOSS_MUSIC_BACKEND = "hf"
$env:MOSS_MUSIC_MODEL_ID = (Resolve-Path ".\weights\models\OpenMOSS--MOSS-Music-8B-Instruct\snapshots\master").Path

# 4. 启动页面
python app.py
```

如果第 1 步失败，是代码或语法问题；如果第 2 步失败，是 FFmpeg/TorchCodec/音频环境
问题；如果页面启动后第一次推理失败，再检查模型分片、CUDA、bitsandbytes 和显存。
