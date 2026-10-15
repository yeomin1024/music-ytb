#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VERSION: v1.1 — 2026-10-15 — 사용자 지시 "음악도구를 music-ytb에 옮기고 처음에 올렸던건 지워": yeomin1024-ytb → music-ytb 저장소로
#          옮김. 동작은 v1.0 과 같고 셀 경로(/kaggle/working/music-ytb)·버전 표기만 바꿈
r"""
YuE2 (m-a-p/YuE2-3B) 로 노래·연주곡을 Kaggle GPU 에서 만든다.  ※ 연구·교육용 도구. Kaggle 실제 GPU 실행은 아직 검증 전.

Kaggle 노트북 — Settings → Accelerator: **GPU T4 x2**, Internet: **On** — 셀 (전체는 README.md):

  !test -d /kaggle/working/music-ytb || git clone -q --depth 1 https://github.com/yeomin1024/music-ytb.git /kaggle/working/music-ytb
  !python /kaggle/working/music-ytb/tools/kaggle_music_yue2.py run --mode instrumental \
      --style "calm lo-fi piano, soft drums, warm bass, 80 BPM" --seeds 1,2

  - 세션마다 처음 한 번: /tmp/yue2/venv 에 파이썬 3.12 + yue2-infer 0.1.6(해시 고정) + torch 2.10 (몇 분),
    GPU 점검(BF16 행렬곱·어텐션), 모델 YuE2-3B·YuE2-Vae 약 7.8GB(리비전 고정) → /tmp/yue2/hf
  - 결과: /kaggle/working/music/<이름>/seed-<시드>/audio.flac · audio.mp3 · score.abc · kaggle_run.json
  - 시드가 여러 개면 GPU 마다 나눠 동시에 (T4 x2 → 2곡 동시에. 공식 기준 '후보 2개 중 고르기')

모드
  song          가사 파일(--lyrics-file, [Verse]·[Chorus] 같은 태그) + 스타일 → 노래
  instrumental  YuE2 가 악보(선율·화음)를 쓰고 → 노래 선율을 악기 성부로 옮긴 뒤(공식 yue2-instrumental 스킬의 변환 코드,
                커밋 고정) → 연주곡. 영상 배경음악용. 섹션 수(--sections)로 길이를 대략 정한다 (초 단위 지정은 모델이 보장하지 않음)

T4(Turing, 연산 능력 7.5)에서 공식 코드가 그대로는 멈추거나 메모리가 넘치는 곳 2군데 — 연산 능력 < 8.0 일 때만 고친다 (로그 [PATCH]):
  1) AR 단계 CUDA 그래프가 어텐션을 'flash' 로 고른다. 고르는 조건이 '함수가 있는가'뿐이라 T4 에서도 고르는데,
     PyTorch 의 FlashAttention 은 Ampere(8.0) 이상 전용 → 실행 중 "FlashAttention only supports Ampere GPUs or newer."
     → 공식 코드에 있는 대체 경로 'sdpa'(PyTorch 가 GPU 에 맞는 커널을 고름)로.
  2) T4 는 BF16 용 빠른 어텐션 커널(flash·mem-efficient·cuDNN)이 없어 수학 커널(fp32)로 가는데, 공식 NAR 코드는 CUDA 에서
     곡 전체 질의를 한 번에 계산(질의 × 키 행렬이 수 GB) → 공식 코드가 CPU 에서 쓰는 방법 그대로 질의를 --chunk(기본 256)줄씩.
  소리를 바꾸는 설정(BF16 가중치, 샘플링, 32 단계, VAE)은 그대로다. VRAM 20GiB 미만이면 공식 옵션 offload_ar 를 켠다
  (NAR 동안 AR 가중치를 CPU 로 — 메모리만 아끼고 결과는 같다).

라이선스: 코드 Apache-2.0, 가중치 CC BY-NC 4.0 + 창작자 추가 허가 — 개인 창작자는 결과물로 수익을 내도 된다(로열티 없음),
회사의 상업 이용은 별도 계약 (공식 README 'License'). 공식 권장 사양은 BF16 지원 GPU 24GB — T4 는 권장 밖이라 느리다.

다른 명령
  setup   설치·GPU 점검만 (run 이 알아서 부름)
  check   (venv 안에서) GPU 점검만 다시
"""
# CHANGELOG
#   v1.0 — 2026-10-15 — 새 도구. setup(uv 0.12.19 → 파이썬 3.12 venv, yue2-infer 0.1.6 휠 sha256 고정, 스킬 변환 코드 5개 커밋·sha256 고정,
#          GPU 점검) · run(시드를 GPU 마다 나눠 하위 프로세스, 모델 먼저 한 번 받기) · _worker(T4 패치 apply_t4_patches,
#          song/instrumental, 결과 검사·mp3·kaggle_run.json). 영향: 새 파일 tools/kaggle_music_yue2.py, tools/tests/test_kaggle_music_yue2.py
#          (처음에는 yeomin1024-ytb 커밋 2223bfb)
#   v1.1 — 2026-10-15 — 저장소 이동 yeomin1024-ytb → music-ytb (사용자 지시). 동작 변경 없음: 고정값·패치·로그 그대로,
#          docstring 의 셀 경로와 TOOL_VERSION 만. 영향: 모듈 머리말, TOOL_VERSION (kaggle_run.json 의 tool 값)
import argparse
import functools
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from datetime import datetime
from pathlib import Path

TOOL_VERSION = "v1.1"
ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------ 고정값 (바꾸면 VERSION 을 올리고 CHANGELOG 에 적는다)
UV_VERSION = "0.12.19"  # 2026-09-25 배포 — 설치 도구만, 결과에 영향 없음
PYTHON_VERSION = "3.12"  # 공식 검증 버전
PACKAGE_VERSION = "0.1.6"
WHEEL_URL = ("https://github.com/multimodal-art-projection/YuE/releases/download/"
             "yue2-v0.1.6/yue2_infer-0.1.6-py3-none-any.whl")
WHEEL_SHA256 = "129a015327a2eca530760ffd2163ccb8cb65f7e601715de1d9626a40c568e2e2"
SKILL_COMMIT = "1647252d68b70cbead046ffd3f7caf036dde47ee"  # YuE main 2026-10-06
SKILL_URL = ("https://raw.githubusercontent.com/multimodal-art-projection/YuE/{commit}/"
             "skills/yue2-music/instrumental/scripts/{name}")
SKILL_FILES = {  # 연주곡 변환에 쓰는 공식 코드 (표준 라이브러리만 씀)
    "abc_tools.py": "ea04b922dacebec7ad257a2f8d83bdb5dfecb7a23110c1a3121c5c41c313930e",
    "common.py": "2929619bd37a176c87c37fdcde9864613d4d5172ca2d89d724aa450a4d7053be",
    "compile_score.py": "c4eec7f68e3f09b93b45d568a8af640b3f10bb47d31ef6d3b6ce4ae23ac1dea3",
    "instrumentalize.py": "a769111656f418cf4298841dc4a7f6be1a50bfae1e66b1bccfaca3eacf948681",
    "instrumental.py": "bd294e4a57f1100722f7eeb14ed2457e1bc65b7a6cf516c4225fb1c3e7a441b0",
}
# 공식 연주곡 스킬(assets/release.json, 2026-09-22 검증)이 고정한 리비전
MODEL, MODEL_REV = "m-a-p/YuE2-3B", "14fc6c6f146441b1dd6363fcb2e01e82a6914cb7"
VAE, VAE_REV = "m-a-p/YuE2-Vae", "9a94e1d0ea9f8087e98f77fa88df4a4068104d2a"

WORK = Path(os.environ.get("YUE2_WORK", "/tmp/yue2"))  # /kaggle/working(20GB, 출력으로 저장됨) 대신 임시 디스크
VENV = WORK / "venv"
VENV_PY = VENV / "bin" / "python"
SKILL_DIR = WORK / "skill"
HF_HOME = WORK / "hf"
READY = WORK / "ready.json"
OUT = Path("/kaggle/working/music") if Path("/kaggle/working").is_dir() else ROOT / "output" / "music"
MIN_FREE_GB = 20  # venv(torch·CUDA 라이브러리) 약 6GB + 모델 7.8GB + 여유
MIN_CAP = (7, 5)  # torch 2.10(PyPI, CUDA 12.8 빌드)이 지원하는 가장 낮은 연산 능력 — P100(6.0)은 안 됨
PATCH_BELOW = (8, 0)  # FlashAttention·BF16 빠른 어텐션 커널이 생기는 연산 능력
DEFAULT_CHUNK = 256  # 공식 nar.attention 이 CPU/MPS 에서 쓰는 질의 묶음 크기
DEFAULT_SECTIONS = "Intro,Verse,Chorus,Outro"  # 공식 연주곡 스킬의 기본 계획 태그
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}")
SR = 48000

LOG = logging.getLogger("yue2")


# ------------------------------------------------------------------ 로그
def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if math.isfinite(v) else str(v)
    s = str(v)
    return f'"{s}"' if (" " in s or not s) else s


def log(stage, level=logging.INFO, **kv):
    """[STAGE] [TIMESTAMP] key=value ... (저장소 공통 로그 형식)"""
    LOG.log(level, "[%s] [%s] %s", stage, datetime.now().isoformat(timespec="seconds"),
            " ".join(f"{k}={_fmt(v)}" for k, v in kv.items()))


def die(stage, error, hint, **kv):
    log(stage, logging.ERROR, error=error, hint=hint, **kv)
    raise SystemExit(1)


def setup_logging(level="INFO", file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if file:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(file, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, level), format="%(message)s", handlers=handlers, force=True)


# ------------------------------------------------------------------ 입력 (GPU 없이 시험할 수 있는 부분)
def parse_seeds(text):
    """'1,2,3' → [1, 2, 3]. 공식 SongRequest 범위 0 ≤ seed < 2**63, 중복 금지."""
    seeds = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if not re.fullmatch(r"\d+", part):
            raise ValueError(f"시드는 0 이상의 정수: {part!r}")
        seed = int(part)
        if seed >= 2**63:
            raise ValueError(f"시드가 너무 큼: {seed}")
        if seed in seeds:
            raise ValueError(f"시드 중복: {seed}")
        seeds.append(seed)
    if not seeds:
        raise ValueError("시드가 비었음")
    return seeds


def section_lyrics(sections):
    """'Intro,Verse,Chorus,Outro' → '[Intro]\\n\\n[Verse]\\n\\n[Chorus]\\n\\n[Outro]\\n' (공식 연주곡 계획 가사와 같은 모양)."""
    names = [s.strip() for s in str(sections).split(",") if s.strip()]
    if not names:
        raise ValueError("섹션이 비었음")
    for n in names:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9 -]{0,30}", n):
            raise ValueError(f"섹션 이름은 영문(예: Intro, Verse, Chorus, Bridge, Outro): {n!r}")
    return "\n\n".join(f"[{n}]" for n in names) + "\n"


def instrumental_style(style):
    """공식 instrumental.py prepare() 와 같은 규칙: 'Instrumental, ' 머리 + 'no vocals …' 조건 + 마침표."""
    style = (style or "").strip().rstrip(".,") or "Expressive instrumental music"
    if not re.match(r"^instrumental\b", style, re.I):
        style = "Instrumental, " + style
    for condition in ("no vocals", "no singing", "no choir", "no spoken words"):
        if condition not in style.lower():
            style += ", " + condition
    return style + "."


def assign_seeds(seeds, gpu_indices):
    """시드를 GPU 에 돌아가며 나눈다 → {gpu: [seeds]} (시드보다 GPU 가 많으면 남는 GPU 는 안 씀)."""
    if not gpu_indices:
        raise ValueError("GPU 가 없음")
    plan = {}
    for i, seed in enumerate(seeds):
        plan.setdefault(gpu_indices[i % len(gpu_indices)], []).append(seed)
    return plan


def fresh_dir(path):
    """이미 있으면 -2, -3 … 을 붙인다 (이전 결과를 덮어쓰지 않음)."""
    path = Path(path)
    if not path.exists():
        return path
    for n in range(2, 1000):
        cand = path.with_name(f"{path.name}-{n}")
        if not cand.exists():
            return cand
    raise FileExistsError(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------ T4 패치 (모듈을 인자로 받아 GPU 없이 시험 가능)
def apply_t4_patches(cuda_graph, nar, chunk=DEFAULT_CHUNK):
    """연산 능력 < 8.0 GPU 용. 소리를 바꾸는 값은 건드리지 않고 '어떤 커널로 계산하나'만 바꾼다 → 적용한 패치 이름 목록.

    1) cuda_graph.GraphAR(attention_backend='auto') 는 aten._flash_attention_forward 가 '있으면' flash 를 고른다
       (GPU 세대 확인 없음). T4 에서는 그 커널이 TORCH_CHECK(is_sm80_or_newer) 로 멈춘다 → 'sdpa' 로 고정.
    2) nar.synthesize(query_chunk_size=None) 는 CUDA 에서 질의 전체를 한 번에 → T4 의 fp32 수학 커널에서 메모리 폭발.
       호출자가 값을 주지 않았을 때만 chunk 를 넣는다 (공식 CPU 경로와 같은 방식).
    """
    if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
        raise ValueError("chunk 는 1 이상의 정수")
    applied = []
    init = cuda_graph.GraphAR.__init__
    if not getattr(init, "_kaggle_t4", False):
        @functools.wraps(init)
        def graph_init(self, *args, **kwargs):
            kwargs["attention_backend"] = "sdpa"
            return init(self, *args, **kwargs)
        graph_init._kaggle_t4 = True
        cuda_graph.GraphAR.__init__ = graph_init
    applied.append("graph_attention=sdpa")
    synth = nar.synthesize
    if getattr(synth, "_kaggle_t4", False):
        synth = synth.__wrapped__
    @functools.wraps(synth)
    def synthesize(*args, **kwargs):
        if kwargs.get("query_chunk_size") is None:
            kwargs["query_chunk_size"] = chunk
        return synth(*args, **kwargs)
    synthesize._kaggle_t4 = True
    nar.synthesize = synthesize
    applied.append(f"nar_query_chunk={chunk}")
    return applied


# ------------------------------------------------------------------ 환경 (Kaggle 기본 파이썬)
def query_gpus():
    """nvidia-smi → [{index, name, mem_mib, driver, cap}] (cap = (major, minor) 또는 None)."""
    cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
           "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return [], f"{type(e).__name__}: {e}"
    rows = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 5:
            continue
        idx, name, mem, driver, cap = parts
        try:
            cap_t = tuple(int(x) for x in cap.split("."))[:2]
        except ValueError:
            cap_t = None
        try:
            mem_mib = int(float(mem))
        except ValueError:
            mem_mib = 0
        rows.append(dict(index=int(idx), name=name, mem_mib=mem_mib, driver=driver, cap=cap_t))
    return rows, None


def ram_gib():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return float("nan")


def disk_free_gb(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def run_cmd(cmd, stage, hint, env=None):
    log(stage, run=" ".join(str(c) for c in cmd[:6]) + (" …" if len(cmd) > 6 else ""))
    try:
        rc = subprocess.run([str(c) for c in cmd], env=env).returncode
    except OSError as e:
        die(stage, error=f"{type(e).__name__}: {e}", hint=hint)
    if rc != 0:
        die(stage, error=f"exit={rc}", hint=hint)


def download(url, dest, sha256, stage):
    dest = Path(dest)
    if dest.is_file() and sha256_file(dest) == sha256:
        log(stage, file=dest.name, status="cached", sha256=sha256[:12])
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as e:  # noqa: BLE001 — 네트워크 오류는 종류가 많다
        die(stage, error=f"{type(e).__name__}: {e}", hint="Kaggle Settings → Internet: On 확인 뒤 다시", url=url)
    got = sha256_file(tmp)
    if got != sha256:
        tmp.unlink(missing_ok=True)
        die(stage, error="sha256 다름 — 받은 파일이 고정한 것과 다름", hint="이 도구의 고정값(WHEEL_SHA256·SKILL_FILES)을 새 버전으로 바꿔야 함",
            url=url, want=sha256[:12], got=got[:12])
    tmp.replace(dest)
    log(stage, file=dest.name, status="ok", bytes=dest.stat().st_size, sha256=sha256[:12])
    return dest


def worker_env(gpu=None):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)  # Kaggle 기본 패키지가 venv 로 섞이지 않게
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", HF_HOME=str(HF_HOME), HF_HUB_DISABLE_PROGRESS_BARS="1")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def setup(force=False):
    """설치·점검. 이미 같은 고정값으로 끝났으면 건너뜀 → GPU 목록."""
    t0 = time.time()
    rows, err = query_gpus()
    log("ENV", python=sys.version.split()[0], gpus=len(rows), ram_gib=round(ram_gib(), 1),
        disk_free_gb=round(disk_free_gb(WORK), 1), work=WORK, out=OUT)
    for g in rows:
        log("ENV", gpu=g["index"], name=g["name"], vram_mib=g["mem_mib"], driver=g["driver"],
            cap=".".join(map(str, g["cap"])) if g["cap"] else "?")
    if not rows:
        die("ENV", error=f"GPU 없음 ({err})", hint="Kaggle 오른쪽 Settings → Accelerator: GPU T4 x2 · Internet: On 뒤 다시")
    old = [g for g in rows if g["cap"] and g["cap"] < MIN_CAP]
    if old:
        die("ENV", error=f"{old[0]['name']} (연산 능력 {'.'.join(map(str, old[0]['cap']))}) — torch 2.10 이 지원하지 않음",
            hint="Settings → Accelerator 를 GPU T4 x2 로 바꾸고 다시 (P100 은 안 됨)")
    if disk_free_gb(WORK) < MIN_FREE_GB:
        log("ENV", logging.WARNING, warning="디스크 여유가 적음", need_gb=MIN_FREE_GB,
            hint="설치·모델 받기에서 멈추면 YUE2_WORK 환경 변수로 더 큰 디스크를 지정")
    want = {"tool": TOOL_VERSION, "wheel_sha256": WHEEL_SHA256, "skill_commit": SKILL_COMMIT, "python": PYTHON_VERSION}
    if READY.is_file() and VENV_PY.is_file() and not force:
        try:
            have = json.loads(READY.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            have = {}
        if all(have.get(k) == v for k, v in want.items()):
            log("SETUP", status="skip", reason="이미 설치·점검됨", venv=VENV)
            return rows
    WORK.mkdir(parents=True, exist_ok=True)
    if VENV.exists():
        shutil.rmtree(VENV)
    hint_net = "Kaggle Settings → Internet: On 확인 뒤 다시"
    run_cmd([sys.executable, "-m", "pip", "install", "-q", f"uv=={UV_VERSION}"], "SETUP", hint_net)
    run_cmd([sys.executable, "-m", "uv", "venv", "--python", PYTHON_VERSION, VENV], "SETUP",
            "uv 가 파이썬 3.12 를 못 받음 → Internet On 확인")
    wheel = download(WHEEL_URL, WORK / Path(WHEEL_URL).name, WHEEL_SHA256, "SETUP")
    run_cmd([sys.executable, "-m", "uv", "pip", "install", "--python", VENV_PY, wheel], "SETUP",
            "torch 2.10·transformers 설치 실패 → 위 오류 줄 확인 (디스크·네트워크)")
    for name, digest in SKILL_FILES.items():
        download(SKILL_URL.format(commit=SKILL_COMMIT, name=name), SKILL_DIR / name, digest, "SETUP")
    run_cmd([VENV_PY, Path(__file__).resolve(), "check"], "CHECK",
            "이 GPU 에서 공식 BF16 경로가 안 됨 → 위 [CHECK] 줄 확인 (대안: yue2.cpp GGUF 포팅)", env=worker_env())
    READY.write_text(json.dumps(want | {"created": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False),
                     encoding="utf-8")
    log("SETUP", status="done", minutes=round((time.time() - t0) / 60, 1), venv=VENV)
    return rows


# ------------------------------------------------------------------ venv 안 — GPU 점검
def check():
    import torch
    import torch.nn.functional as F
    log("CHECK", torch=torch.__version__, cuda=torch.version.cuda, devices=torch.cuda.device_count())
    if not torch.cuda.is_available():
        die("CHECK", error="torch 가 GPU 를 못 봄", hint="nvidia-smi 확인 · Accelerator: GPU T4 x2")
    bad = 0
    for i in range(torch.cuda.device_count()):
        dev = torch.device("cuda", i)
        with torch.cuda.device(dev):
            cap = torch.cuda.get_device_capability(dev)
            bf16 = torch.cuda.is_bf16_supported()  # 공식 파이프라인이 검사하는 그 함수 (T4 는 에뮬레이션으로 True)
            g = torch.Generator(device=dev).manual_seed(0)
            a = torch.randn(512, 512, device=dev, generator=g)
            b = torch.randn(512, 512, device=dev, generator=g)
            ref = a @ b
            got = (a.bfloat16() @ b.bfloat16()).float()
            rel = ((got - ref).norm() / ref.norm()).item()
            q = torch.randn(1, 8, 300, 64, device=dev, dtype=torch.bfloat16, generator=g)
            kv = torch.randn(1, 2, 300, 64, device=dev, dtype=torch.bfloat16, generator=g)
            mask = torch.ones(300, 300, dtype=torch.bool, device=dev).tril()
            o = F.scaled_dot_product_attention(q, kv, kv, attn_mask=mask, enable_gqa=True)  # 모델과 같은 GQA·마스크 호출
            sdpa_ok = bool(torch.isfinite(o).all().item())
            torch.cuda.synchronize(dev)
        ok = bf16 and sdpa_ok and rel < 0.05
        bad += not ok
        log("CHECK", logging.INFO if ok else logging.ERROR, gpu=i, name=torch.cuda.get_device_name(i),
            cap=f"{cap[0]}.{cap[1]}", bf16_supported=bf16, bf16_matmul_rel_err=rel, sdpa_gqa_ok=sdpa_ok,
            patch="t4" if cap < PATCH_BELOW else "none", ok=ok)
    if bad:
        die("CHECK", error=f"GPU {bad}개 점검 실패", hint="위 [CHECK] 줄 · 대안: yue2.cpp(GGUF) 포팅")
    return 0


# ------------------------------------------------------------------ venv 안 — 생성
def audio_metrics(path):
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    finite = bool(np.isfinite(data).all())
    peak = float(np.abs(data).max()) if data.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64)))) if data.size else 0.0
    db = lambda x: 20 * math.log10(x) if x > 0 else float("-inf")  # noqa: E731
    return dict(seconds=round(len(data) / sr, 2), sample_rate=sr, channels=data.shape[1], finite=finite,
                peak_dbfs=round(db(peak), 2), rms_dbfs=round(db(rms), 2))


def build_request(args, seed, pipe=None, out=None):
    """→ (공식 SongRequest 인자 dict, 기록용 extra). instrumental 은 YuE2 계획 → 공식 변환 → 공식 검사."""
    if args.mode == "song":
        lyrics = Path(args.lyrics_file).read_text(encoding="utf-8")
        return dict(id=args.name, style=args.style.strip(), lyrics=lyrics, cot=args.cot, seed=seed), {}
    sys.path.insert(0, str(SKILL_DIR))
    from instrumentalize import convert_score  # noqa: E402 — 커밋·sha256 고정한 공식 코드
    from instrumental import lyric_tags, validate_request, validate_score  # noqa: E402
    from yue2.protocol import SongRequest

    plan_req = SongRequest(id=args.name, style=args.style.strip(), lyrics=section_lyrics(args.sections),
                           cot=args.plan_mode, seed=seed)
    t0 = time.time()
    plan = pipe.plan(request=plan_req)
    plan.save(out / "planning")
    log("PLAN", seed=seed, sections=args.sections, plan_mode=args.plan_mode, truncated=plan.truncated,
        abc_chars=len(plan.abc or ""), seconds=round(time.time() - t0, 1))
    if plan.truncated or not plan.abc:
        raise ValueError("YuE2 악보가 비었거나 잘림 — 다른 시드나 더 적은 섹션으로")
    converted, transfer = convert_score(plan.abc, overlap="vocal", keep_chords=True)
    (out / "conversion").mkdir(parents=True, exist_ok=True)
    (out / "conversion" / "score.abc").write_text(converted, encoding="utf-8")
    (out / "conversion" / "transfer.json").write_text(json.dumps(transfer, ensure_ascii=False, indent=1), encoding="utf-8")
    score = validate_score(converted)
    request = dict(id=args.name, style=instrumental_style(args.style), lyrics=lyric_tags(converted), abc=converted,
                   cot="full" if score.voices["Vocal"].chords else "melody", seed=seed)
    validate_request(request)  # 공식 검사: 가사 없음(섹션 태그만), cot 와 악보 일치
    log("CONVERT", seed=seed, ins_notes=len(score.voices["Ins"].notes), vocal_notes=len(score.voices["Vocal"].notes),
        cot=request["cot"], style=request["style"][:80])
    return request, {"plan_style": plan_req.style, "transfer": transfer}


def generate_one(pipe, args, seed, meta):
    import torch
    out = fresh_dir(Path(args.out) / args.name / f"seed-{seed}")
    out.mkdir(parents=True)
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    request, extra = build_request(args, seed, pipe, out)
    log("GEN", seed=seed, mode=args.mode, cot=request["cot"], lyrics_chars=len(request["lyrics"]), out=out)
    song = pipe(**request)
    song.save_artifacts(out)
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    metrics = audio_metrics(out / "audio.flac")
    truncated = song.truncated
    log("GEN", seed=seed, status="done", minutes=round((time.time() - t0) / 60, 1), peak_vram_gib=round(peak_gib, 2),
        truncated=any(truncated.values()), **metrics)
    timing = song.timing
    log("TIMING", seed=seed, nar_s=round(timing.get("nar_seconds", 0), 1), vae_s=round(timing.get("vae_seconds", 0), 1),
        e2e_s=round(timing.get("e2e_seconds", 0), 1))
    # 결과 검사 — 실패해도 파일은 남기고 경고 (사람이 들어 보고 판단)
    if not metrics["finite"]:
        log("CHECK", logging.WARNING, seed=seed, warning="NaN/Inf 가 있는 오디오")
    if metrics["rms_dbfs"] < -50:
        log("CHECK", logging.WARNING, seed=seed, warning="거의 무음", rms_dbfs=metrics["rms_dbfs"])
    if any(truncated.values()):
        log("CHECK", logging.WARNING, seed=seed, warning="생성 한도에 닿아 잘림", hint="가사·섹션을 줄이거나 다른 시드")
    mp3 = None
    if shutil.which("ffmpeg"):
        mp3 = out / "audio.mp3"
        r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-i", str(out / "audio.flac"),
                            "-b:a", "192k", str(mp3)])
        if r.returncode:
            log("SAVE", logging.WARNING, warning="mp3 변환 실패 — flac 은 있음", seed=seed)
            mp3 = None
    record = dict(tool=TOOL_VERSION, created=datetime.now().isoformat(timespec="seconds"), mode=args.mode,
                  request={k: v for k, v in request.items() if k != "abc"}, extra=extra, meta=meta,
                  peak_vram_gib=round(peak_gib, 3), truncated=truncated, metrics=metrics, timing=timing,
                  note="연구·교육용. 가중치 CC BY-NC 4.0 + 창작자 추가 허가 (공식 MODEL_LICENSE)")
    (out / "kaggle_run.json").write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log("SAVE", seed=seed, flac=out / "audio.flac", mp3=mp3 or "-", record=out / "kaggle_run.json")
    return out


def worker(args):
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    import torch
    if not torch.cuda.is_available():
        die("LOAD", error="GPU 없음", hint="Accelerator: GPU T4 x2")
    cap = torch.cuda.get_device_capability(0)
    import yue2
    import yue2.cuda_graph
    import yue2.nar
    from yue2 import YuE2Pipeline
    patches = apply_t4_patches(yue2.cuda_graph, yue2.nar, args.chunk) if cap < PATCH_BELOW else []
    vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    offload = {"on": True, "off": False, "auto": vram_gib < 20}[args.offload_ar]
    meta = dict(gpu=torch.cuda.get_device_name(0), cap=f"{cap[0]}.{cap[1]}", vram_gib=round(vram_gib, 2),
                torch=torch.__version__, yue2_infer=PACKAGE_VERSION, model=f"{MODEL}@{args.model_rev}",
                vae=f"{VAE}@{args.vae_rev}", backend=args.backend, offload_ar=offload, patches=patches)
    log("PATCH", gpu=meta["gpu"], cap=meta["cap"], applied=",".join(patches) or "none",
        reason="연산 능력 < 8.0: FlashAttention·BF16 빠른 커널 없음" if patches else "-")
    log("LOAD", model=MODEL, rev=args.model_rev[:12], vae=VAE, vae_rev=args.vae_rev[:12], backend=args.backend,
        offload_ar=offload, vram_gib=round(vram_gib, 1), hf_home=os.environ["HF_HOME"])
    t0 = time.time()
    try:
        pipe = YuE2Pipeline.from_pretrained(MODEL, vae=VAE, revision=args.model_rev, vae_revision=args.vae_rev,
                                            device="cuda", backend=args.backend, offload_ar=offload,
                                            progress=not args.quiet)
    except Exception as e:  # noqa: BLE001
        die("LOAD", error=f"{type(e).__name__}: {e}",
            hint="401·403·gated 면 Kaggle Secrets 에 HF_TOKEN 추가 · 연결 오류면 Internet On · 공간 부족이면 YUE2_WORK")
    log("LOAD", status="ok", seconds=round(time.time() - t0, 1))
    if args.fetch_only:
        pipe.close()
        return 0
    rc = 0
    with pipe:
        for seed in parse_seeds(args.seeds):
            try:
                generate_one(pipe, args, seed, meta)
            except torch.cuda.OutOfMemoryError as e:
                log("GEN", logging.ERROR, seed=seed, error=f"GPU 메모리 부족: {str(e).splitlines()[0]}",
                    hint="--chunk 128 · 가사/섹션 줄이기 · --backend torch-eager 순서로 시도")
                rc = 1
                break  # 메모리 상태가 불확실하니 다음 시드로 가지 않음
            except Exception as e:  # noqa: BLE001
                log("GEN", logging.ERROR, seed=seed, error=f"{type(e).__name__}: {e}",
                    where=traceback.extract_tb(e.__traceback__)[-1].name, hint="위 오류 · 다른 시드로 다시")
                rc = 1
    return rc


# ------------------------------------------------------------------ 하위 프로세스 (Kaggle 기본 파이썬)
def stream(proc, prefix, sink):
    for line in proc.stdout:  # text=True → '\r' 진행 줄도 한 줄씩 (공식 Progress 는 파이프에서 5초에 한 줄)
        sink(f"{prefix} | {line.rstrip()}")


def worker_cmd(args, gpu_seeds, fetch_only=False):
    cmd = [VENV_PY, Path(__file__).resolve(), "_worker", "--mode", args.mode, "--style", args.style,
           "--name", args.name, "--out", args.out, "--seeds", ",".join(map(str, gpu_seeds)),
           "--sections", args.sections, "--plan-mode", args.plan_mode, "--cot", args.cot,
           "--backend", args.backend, "--chunk", str(args.chunk), "--offload-ar", args.offload_ar,
           "--model-rev", args.model_rev, "--vae-rev", args.vae_rev, "--log-level", args.log_level]
    if args.lyrics_file:
        cmd += ["--lyrics-file", str(Path(args.lyrics_file).resolve())]
    if fetch_only:
        cmd.append("--fetch-only")
    if args.quiet:
        cmd.append("--quiet")
    return [str(c) for c in cmd]


def run(args):
    t0 = time.time()
    if not NAME_RE.fullmatch(args.name):
        die("INPUT", error=f"이름 {args.name!r}", hint="영문·숫자·_.- 만 (예: bgm-lofi-01)")
    if not args.style.strip():
        die("INPUT", error="스타일이 비었음", hint='--style "calm piano, soft drums, 80 BPM"')
    try:
        seeds = parse_seeds(args.seeds)
        if args.mode == "instrumental":
            section_lyrics(args.sections)
    except ValueError as e:
        die("INPUT", error=str(e), hint="--seeds 1,2 · --sections Intro,Verse,Chorus,Outro")
    if args.mode == "song":
        if not args.lyrics_file or not Path(args.lyrics_file).is_file():
            die("INPUT", error="노래 모드는 가사 파일이 필요", hint="--lyrics-file lyrics.txt ([Verse]·[Chorus] 태그 + 가사)")
        if not Path(args.lyrics_file).read_text(encoding="utf-8").strip():
            die("INPUT", error="가사 파일이 비었음", hint="[Verse] 아래에 가사")
    rows = setup(force=args.reinstall)
    usable = [g["index"] for g in rows]
    if args.gpus != "all":
        want = [int(x) for x in args.gpus.split(",") if x.strip()]
        usable = [i for i in want if i in usable]
        if not usable:
            die("INPUT", error=f"--gpus {args.gpus} 가 없음", hint=f"있는 GPU: {[g['index'] for g in rows]}")
    plan = assign_seeds(seeds, usable)
    log("RUN", mode=args.mode, name=args.name, seeds=",".join(map(str, seeds)),
        gpus=";".join(f"{g}:{','.join(map(str, s))}" for g, s in plan.items()), out=Path(args.out) / args.name)
    if len(plan) > 1 and ram_gib() < 24:
        log("RUN", logging.WARNING, warning="RAM 이 적어 두 GPU 동시 로드가 빠듯함", ram_gib=round(ram_gib(), 1),
            hint="멈추면 --gpus 0")
    sink = LOG.info
    # 모델을 먼저 한 번 받는다 (두 프로세스가 같은 7.8GB 를 동시에 받지 않게)
    first = next(iter(plan))
    proc = subprocess.Popen(worker_cmd(args, plan[first], fetch_only=True), env=worker_env(first),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    stream(proc, "fetch", sink)
    if proc.wait():
        die("LOAD", error="모델 받기 실패", hint="위 fetch 줄의 [LOAD] 오류 확인")
    procs, threads = [], []
    for gpu, gpu_seeds in plan.items():
        p = subprocess.Popen(worker_cmd(args, gpu_seeds), env=worker_env(gpu),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        t = threading.Thread(target=stream, args=(p, f"gpu{gpu}", sink), daemon=True)
        t.start()
        procs.append((gpu, p))
        threads.append(t)
    codes = {gpu: p.wait() for gpu, p in procs}
    for t in threads:
        t.join()
    files = sorted(str(p) for p in (Path(args.out) / args.name).glob("seed-*/audio.flac"))
    log("DONE", minutes=round((time.time() - t0) / 60, 1), files=len(files),
        exit=",".join(f"gpu{g}={c}" for g, c in codes.items()),
        hint="-" if not any(codes.values()) else "실패한 GPU 의 [GEN] 오류 줄 확인")
    for f in files:
        log("DONE", file=f)
    return 1 if any(codes.values()) else 0


# ------------------------------------------------------------------ 명령줄
def add_gen_args(p):
    p.add_argument("--mode", choices=("instrumental", "song"), default="instrumental")
    p.add_argument("--style", default="", help='영어 권장. 장르·악기·분위기·BPM (예: "calm lo-fi piano, soft drums, 80 BPM")')
    p.add_argument("--lyrics-file", help="song: 가사 파일 ([Verse]·[Chorus] 태그)")
    p.add_argument("--sections", default=DEFAULT_SECTIONS, help=f"instrumental: 계획 섹션 (기본 {DEFAULT_SECTIONS} — 늘리면 길어짐)")
    p.add_argument("--plan-mode", choices=("full", "melody"), default="full", help="instrumental: 악보에 화음까지(full) / 선율만")
    p.add_argument("--cot", choices=("full", "melody", "off"), default="full", help="song: 계획 방식 (공식 기본 full)")
    p.add_argument("--seeds", default="1", help="쉼표로 여러 개 — GPU 마다 나눠 동시에")
    p.add_argument("--name", default=None, help="결과 폴더 이름 (기본: 모드-날짜시각)")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--backend", choices=("torch", "torch-eager"), default="torch",
                   help="torch = CUDA 그래프(빠름), torch-eager = 그래프 없이(느림, 문제 생길 때)")
    p.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help="T4 패치 2: NAR 질의 묶음 (메모리 부족이면 128)")
    p.add_argument("--offload-ar", choices=("auto", "on", "off"), default="auto", help="auto = VRAM 20GiB 미만이면 켬")
    p.add_argument("--model-rev", default=MODEL_REV)
    p.add_argument("--vae-rev", default=VAE_REV)
    p.add_argument("--quiet", action="store_true", help="공식 진행 표시 끄기")
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")


def main(argv=None):
    ap = argparse.ArgumentParser(description="YuE2 노래·연주곡 — Kaggle GPU (연구·교육용)")
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="설치(처음 한 번) → 생성")
    add_gen_args(r)
    r.add_argument("--gpus", default="all", help="쓸 GPU (예: 0 또는 0,1)")
    r.add_argument("--reinstall", action="store_true", help="venv 를 지우고 다시 설치")
    s = sub.add_parser("setup", help="설치·GPU 점검만")
    s.add_argument("--reinstall", action="store_true")
    s.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    c = sub.add_parser("check", help="(venv 안) GPU 점검")
    c.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    w = sub.add_parser("_worker", help=argparse.SUPPRESS)
    add_gen_args(w)
    w.add_argument("--fetch-only", action="store_true")
    args = ap.parse_args(argv)
    if args.command == "run":
        args.name = args.name or f"{'bgm' if args.mode == 'instrumental' else 'song'}-{datetime.now():%Y%m%d-%H%M%S}"
        setup_logging(args.log_level, Path(args.out) / "music.log")
        return run(args)
    setup_logging(args.log_level)
    if args.command == "setup":
        setup(force=args.reinstall)
        return 0
    if args.command == "check":
        return check()
    return worker(args)


if __name__ == "__main__":
    sys.exit(main())
