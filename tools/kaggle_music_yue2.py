#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VERSION: v1.4 — 2026-10-16 — 사용자 지시 "음악 하나랑 … 멘트와 … 사진 … 트랙을 분리한 후 각각 멜로디, 이펙트 등을 분석하고 변형해서
#          내 멘트와 이미지 종합해서 음악 하나 … 가사 여부는 내가 체크" + "멘트·사진은 내가 클로드한테 물어보고 결과 넣을테니까 … 선택 2
#          (가사 초안 → 고친 뒤 생성)는 그렇게 해": 분석에 Demucs 트랙 분리(6트랙)·트랙별 수치·멜로디 특징·곡 구성을 더하고,
#          analyze 명령이 'Claude 에게 보낼 글'을 만든다 (멘트 포함, 사진은 사용자가 Claude 대화에 첨부)
#          (v1.3: 업로드 버튼 패널 · v1.2: 참고 곡 → 비슷한 분위기 여러 곡 → 멜로디 검사 → 통과한 곡만 results/ 에)
r"""
YuE2 (m-a-p/YuE2-3B) 로 노래·연주곡을 Kaggle GPU 에서 만든다.  ※ 연구·교육용 도구. Kaggle 실제 GPU 실행은 아직 검증 전.

Kaggle 노트북 — Settings → Accelerator: **GPU T4 x2**, Internet: **On**. 셀 전체는 README.md.

  # 버튼으로 (곡 올리기 → 비슷한 곡 만들기) — tools/kaggle_music_panel.py
  import sys, importlib; sys.path.insert(0, "/kaggle/working/music-ytb/tools")
  import kaggle_music_panel; importlib.reload(kaggle_music_panel).panel()

  # 명령으로: 참고 곡 하나 → 비슷한 분위기의 연주곡 4곡 → 멜로디 검사 → 통과한 곡만 저장소 results/ 에
  !python /kaggle/working/music-ytb/tools/kaggle_music_yue2.py run --ref <곡 파일 | auto> --count 4 --push

  # 멘트·사진까지 (버튼 패널이 하는 일): ① 분석 + Claude 에게 보낼 글 → ② Claude 답(STYLE·SECTIONS·LYRICS) → ③ 생성
  !python …/kaggle_music_yue2.py analyze --ref <곡> --name my-01 --comment-file comment.txt [--lyrics]
      → /kaggle/working/music/my-01/claude_request.txt (복사해 Claude 에, 사진 첨부)
  !python …/kaggle_music_yue2.py run --ref <곡> --name my-01 --style "<STYLE>" --sections "<SECTIONS>" \
      [--mode song --lyrics-file lyrics.txt] --count 4 --push

  # 프롬프트만으로
  !python /kaggle/working/music-ytb/tools/kaggle_music_yue2.py run --mode instrumental \
      --style "calm lo-fi piano, soft drums, warm bass, 80 BPM" --seeds 1,2

  - 세션마다 처음 한 번: /tmp/yue2/venv 에 파이썬 3.12 + yue2-infer 0.1.6(해시 고정) + torch 2.10 + 분석 패키지 (몇 분),
    GPU 점검(BF16 행렬곱·어텐션), 모델 YuE2-3B·YuE2-Vae 약 7.8GB(리비전 고정) → /tmp/yue2/hf
  - 결과: /kaggle/working/music/<이름>/seed-<시드>/audio.flac · audio.mp3(320k) · score.abc · kaggle_run.json · similarity.json
  - 시드가 여러 개면 GPU 마다 나눠 동시에 (T4 x2 → 2곡 동시에)

참고 곡 → 비슷한 분위기 (--ref 파일 | auto = /kaggle/input 아래 오디오 파일 하나)
  원곡 소리는 YuE2 에 넣지 않는다. 분석 결과를 '글(스타일 프롬프트)'로만 넘기므로 멜로디는 모델이 새로 쓴다.
  1) 분석: librosa(템포·조성·음량) · AudioSet 태그 모델 AST(장르·분위기·악기, 확률을 로그에) ·
     SheetSage2(원곡 멜로디를 악보로 받아 적기 — 검사용, YuE2 팀의 공식 모델, 템포·조성도 여기 값을 먼저 씀) ·
     Demucs htdemucs_6s 트랙 분리(보컬·드럼·베이스·기타·피아노·그 밖) → 트랙별 비중·밝기·스테레오 폭·다이내믹·잔향/지속·거친 정도
     (music_analysis.stem_features — 근사치) · 멜로디 특징(음역·순차/도약·리듬 밀도 — 음표는 내보내지 않음) · 곡 구성.
     분리한 원곡 트랙은 메모리에서 수치만 재고 버린다 (파일로 남기지 않고, 결과 곡에 섞지 않음 — 섞으면 샘플링)
  2) 프롬프트 = [--style-extra] + 장르 + 분위기 + 악기 + 장·단조(SheetSage2 가 될 때만) + BPM. 곡 이름·가수 이름은 넣지 않는다.
     --style 을 주면 그것을 씀
  3) --count 곡 (시드 1..N)
  4) 멜로디 검사 (similarity.json): 만든 곡의 악보(YuE2 계획) ↔ 원곡을 받아 적은 멜로디. 조옮김·템포와 상관없이
     '음정 변화 + 리듬 비'가 연속으로 같은 가장 긴 구간을 잰다 (같은 음 반복만인 구간은 빼고).
     리듬까지 같은 구간 ≥ FLAG_RUN_RHYTHM, 또는 음정만 같은 구간 ≥ FLAG_RUN_PITCH 이면 '검토 필요' — 경험칙이며 법적 판단이 아니다.
     같은 프롬프트로 만든 곡끼리의 값(우연히 겹치는 정도)도 함께 적는다. 받아 적기가 틀리면 놓칠 수 있다(SheetSage2 오차).
  5) --push: '통과' 곡만 저장소 results/<이름>/ 에 — mp3·악보·기록·검사 결과 + analysis.json·prompt.txt·README.md (허용 목록 방식).
     원곡 파일·원곡 악보·원곡 파일 이름은 올리지 않는다. '검토 필요'는 --push-flagged, 멜로디 검사를 못 한 곡은 --push-unchecked 일 때만.
     토큰은 환경 변수 GITHUB_TOKEN (Kaggle Secrets) — 로그·파일·명령줄·주소에 남기지 않는다 (git 이 물을 때만 GIT_ASKPASS 로).

모드
  song          가사 파일(--lyrics-file, [Verse]·[Chorus] 같은 태그) + 스타일 → 노래
  instrumental  YuE2 가 악보(선율·화음)를 쓰고 → 노래 선율을 악기 성부로 옮긴 뒤(공식 yue2-instrumental 스킬의 변환 코드,
                커밋 고정) → 연주곡. 섹션 수(--sections)로 길이를 대략 정한다 (초 단위 지정은 모델이 보장하지 않음)

T4(Turing, 연산 능력 7.5)에서 공식 코드가 그대로는 멈추거나 메모리가 넘치는 곳 2군데 — 연산 능력 < 8.0 일 때만 고친다 (로그 [PATCH]):
  1) AR 단계 CUDA 그래프가 어텐션을 'flash' 로 고른다. 고르는 조건이 '함수가 있는가'뿐이라 T4 에서도 고르는데,
     PyTorch 의 FlashAttention 은 Ampere(8.0) 이상 전용 → 실행 중 "FlashAttention only supports Ampere GPUs or newer."
     → 공식 코드에 있는 대체 경로 'sdpa'(PyTorch 가 GPU 에 맞는 커널을 고름)로.
  2) T4 는 BF16 용 빠른 어텐션 커널(flash·mem-efficient·cuDNN)이 없어 수학 커널(fp32)로 가는데, 공식 NAR 코드는 CUDA 에서
     곡 전체 질의를 한 번에 계산(질의 × 키 행렬이 수 GB) → 공식 코드가 CPU 에서 쓰는 방법 그대로 질의를 --chunk(기본 256)줄씩.
  소리를 바꾸는 설정(BF16 가중치, 샘플링, 32 단계, VAE)은 그대로다. VRAM 20GiB 미만이면 공식 옵션 offload_ar 를 켠다
  (NAR 동안 AR 가중치를 CPU 로 — 메모리만 아끼고 결과는 같다). SheetSage2 는 T4 에서 fp32 로 돌린다.

라이선스: 코드 Apache-2.0, 가중치 CC BY-NC 4.0 + 창작자 추가 허가 — 개인 창작자는 결과물로 수익을 내도 된다(로열티 없음),
회사의 상업 이용은 별도 계약 (공식 README 'License'). 공식 권장 사양은 BF16 지원 GPU 24GB — T4 는 권장 밖이라 느리다.

다른 명령
  setup        설치·GPU 점검만 (run 이 알아서 부름)
  check        (venv 안에서) GPU 점검만 다시
  similarity   --name 이름 : 이미 만든 곡들의 멜로디 검사를 다시
  push         --name 이름 : 이미 만든 곡을 저장소에 (올리기가 실패했을 때)
"""
# CHANGELOG
#   v1.0 — 2026-10-15 — 새 도구. setup(uv 0.12.19 → 파이썬 3.12 venv, yue2-infer 0.1.6 휠 sha256 고정, 스킬 변환 코드 5개 커밋·sha256 고정,
#          GPU 점검) · run(시드를 GPU 마다 나눠 하위 프로세스, 모델 먼저 한 번 받기) · _worker(T4 패치 apply_t4_patches,
#          song/instrumental, 결과 검사·mp3·kaggle_run.json). 영향: 새 파일 tools/kaggle_music_yue2.py, tools/tests/test_kaggle_music_yue2.py
#          (처음에는 yeomin1024-ytb 커밋 2223bfb)
#   v1.1 — 2026-10-15 — 저장소 이동 yeomin1024-ytb → music-ytb (사용자 지시). 동작 변경 없음: 고정값·패치·로그 그대로,
#          docstring 의 셀 경로와 TOOL_VERSION 만. 영향: 모듈 머리말, TOOL_VERSION (kaggle_run.json 의 tool 값)
#   v1.2 — 2026-10-15 — 참고 곡 → 비슷한 분위기 여러 곡 → 멜로디 검사 → 저장소 올리기 (사용자 지시).
#          새것: run --ref/--count/--style-extra/--push(-flagged·-unchecked·-flac)/--reanalyze, _analyze(librosa·AST·SheetSage2 →
#          analysis.json·prompt.txt), 멜로디 비교(melody_line·melody_tokens·longest_run·compare_melodies·judge), run_similarity,
#          push_check·publish_files·push_results, 명령 similarity·push.
#          바뀐 것: 설치에 EXTRA_PACKAGES(librosa 0.11.0 + SheetSage2 의존 4개 — YuE2-Turbo 와 같은 고정) 추가, 스킬 파일에
#          transcribe_cover.py·assets/cover-release.json 추가(경로 skill/scripts·skill/assets 로 — 설치를 한 번 다시 함),
#          mp3 192k → 320k(저장소에 올리는 파일이라 음질 우선), --seeds 기본 = 1..--count(--ref 면 4, 아니면 1).
#          장·단조는 SheetSage2 값만 프롬프트에 (librosa 크로마 추정은 합성 A단조를 A장조로 읽어 기록만).
#          위험 관련 값 변화 없음 (생성 설정·T4 패치 그대로). 영향: setup, run, worker_cmd, generate_one(mp3), main
#   v1.3 — 2026-10-15 — 업로드 버튼 패널(새 파일 tools/kaggle_music_panel.py v1.0)이 run --ref 를 부름 (사용자 지시).
#          run: 같은 --name 의 analysis.json 이 다른 참고 곡(sha256 다름)이면 멈춤 — 예전엔 분석을 건너뛰어 다른 곡의 프롬프트·
#          원곡 악보로 만들고 검사할 수 있었음. analyze: 원곡을 바꾼 wav 를 <이름>/reference/ref.wav(/kaggle/working — Kaggle 출력에
#          저장됨) 대신 WORK/ref_audio/<이름>.wav(/tmp)에. 생성·검사·올리기 동작은 그대로. 영향: run, analyze
#   v1.4 — 2026-10-16 — 트랙 분리·트랙별 분석·Claude 에게 보낼 글 (사용자 지시). 새것: EXTRA_PACKAGES 에 demucs==4.1.0(MIT,
#          torchaudio 없이 동작), separate_stems(htdemucs_6s, 메모리에서만), _analyze 에 stems·melody_features·form,
#          ensure_analysis(run 과 같이 씀), analyze 명령(--comment-file·--lyrics → claude_request.txt). 순수 계산은 새 파일
#          tools/music_analysis.py v1.0. 생성·검사·올리기·T4 패치는 그대로. 설치 목록이 바뀌어 venv 를 한 번 다시 만든다.
#          영향: setup(설치 목록), _analyze, run(분석 부분 → ensure_analysis), main
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import music_analysis as ma  # noqa: E402 — 표준 라이브러리만 (numpy 는 함수 안에서)

TOOL_VERSION = "v1.4"
ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------ 고정값 (바꾸면 VERSION 을 올리고 CHANGELOG 에 적는다)
UV_VERSION = "0.12.19"  # 2026-09-25 배포 — 설치 도구만, 결과에 영향 없음
PYTHON_VERSION = "3.12"  # 공식 검증 버전
PACKAGE_VERSION = "0.1.6"
WHEEL_URL = ("https://github.com/multimodal-art-projection/YuE/releases/download/"
             "yue2-v0.1.6/yue2_infer-0.1.6-py3-none-any.whl")
WHEEL_SHA256 = "129a015327a2eca530760ffd2163ccb8cb65f7e601715de1d9626a40c568e2e2"
EXTRA_PACKAGES = (
    "librosa==0.11.0",  # 참고 곡 분석 (템포·조성·음량)
    # SheetSage2 가 쓰는 것 — YuE2-Turbo(server extra)가 같은 YuE2 환경에서 SheetSage2 를 돌릴 때 고정한 버전
    "mir_eval==0.8.2", "pretty_midi==0.2.10", "mido==1.3.3", "scipy==1.15.3",
    "demucs==4.1.0",  # 트랙 분리 (MIT, 2026-07 배포 — torchaudio 없이 동작, 가중치는 HF adefossez/… 또는 dl.fbaipublicfiles.com)
)
DEMUCS_MODEL = "htdemucs_6s"  # drums·bass·other·vocals·guitar·piano
SKILL_COMMIT = "1647252d68b70cbead046ffd3f7caf036dde47ee"  # YuE main 2026-10-06
SKILL_URL = ("https://raw.githubusercontent.com/multimodal-art-projection/YuE/{commit}/"
             "skills/yue2-music/instrumental/{path}")
SKILL_FILES = {  # 공식 yue2-instrumental 스킬 (표준 라이브러리만 — transcribe_cover 만 torch·transformers)
    "scripts/abc_tools.py": "ea04b922dacebec7ad257a2f8d83bdb5dfecb7a23110c1a3121c5c41c313930e",
    "scripts/common.py": "2929619bd37a176c87c37fdcde9864613d4d5172ca2d89d724aa450a4d7053be",
    "scripts/compile_score.py": "c4eec7f68e3f09b93b45d568a8af640b3f10bb47d31ef6d3b6ce4ae23ac1dea3",
    "scripts/instrumentalize.py": "a769111656f418cf4298841dc4a7f6be1a50bfae1e66b1bccfaca3eacf948681",
    "scripts/instrumental.py": "bd294e4a57f1100722f7eeb14ed2457e1bc65b7a6cf516c4225fb1c3e7a441b0",
    "scripts/transcribe_cover.py": "d88174a8d7cd4d8b9c86f7c6096f3d59796e2dcdb80a8e2878330273a9b52b34",  # 원곡 → 멜로디 악보
    "assets/cover-release.json": "d5b31d144ae25a148802c7b44eab1612d3609bdb944c71a7a96915cc5df6c5a5",  # SheetSage2·MERT2 리비전
}
# 공식 연주곡 스킬(assets/release.json, 2026-09-22 검증)이 고정한 리비전
MODEL, MODEL_REV = "m-a-p/YuE2-3B", "14fc6c6f146441b1dd6363fcb2e01e82a6914cb7"
VAE, VAE_REV = "m-a-p/YuE2-Vae", "9a94e1d0ea9f8087e98f77fa88df4a4068104d2a"
AST_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"  # AudioSet 527 태그 (장르·분위기·악기)

WORK = Path(os.environ.get("YUE2_WORK", "/tmp/yue2"))  # /kaggle/working(20GB, 출력으로 저장됨) 대신 임시 디스크
VENV = WORK / "venv"
VENV_PY = VENV / "bin" / "python"
SKILL_ROOT = WORK / "skill"
SKILL_DIR = SKILL_ROOT / "scripts"  # transcribe_cover.py 가 ../assets/cover-release.json 을 읽는다
HF_HOME = WORK / "hf"
READY = WORK / "ready.json"
OUT = Path("/kaggle/working/music") if Path("/kaggle/working").is_dir() else ROOT / "output" / "music"
KAGGLE_INPUT = Path("/kaggle/input")
AUDIO_EXT = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac")
MIN_FREE_GB = 25  # venv(torch·CUDA 라이브러리) 약 6GB + YuE2 7.8GB + SheetSage2·MERT2·AST 약 4GB + 여유
MIN_CAP = (7, 5)  # torch 2.10(PyPI, CUDA 12.8 빌드)이 지원하는 가장 낮은 연산 능력 — P100(6.0)은 안 됨
PATCH_BELOW = (8, 0)  # FlashAttention·BF16 빠른 어텐션 커널이 생기는 연산 능력
DEFAULT_CHUNK = 256  # 공식 nar.attention 이 CPU/MPS 에서 쓰는 질의 묶음 크기
DEFAULT_SECTIONS = "Intro,Verse,Chorus,Outro"  # 공식 연주곡 스킬의 기본 계획 태그
REF_COUNT = 4  # --ref 일 때 기본 곡 수
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}")
SR = 48000
MP3_BITRATE = "320k"

# 저장소 올리기
REPO_SLUG = "yeomin1024/music-ytb"
RESULTS_DIR = "results"
MAX_PUSH_MB = 95  # GitHub 파일 한도 100MB
GIT_NAME, GIT_EMAIL = "kaggle_music_yue2 (Kaggle)", "kaggle-music-yue2@users.noreply.github.com"
PUSH_TRIES = 4
SEED_FILES = ("audio.mp3", "score.abc", "kaggle_run.json", "similarity.json")  # 곡마다 올리는 것 (허용 목록)
NAME_FILES = ("analysis.json", "prompt.txt")  # 이름마다 올리는 것 — reference/ 폴더(원곡·원곡 악보)는 목록에 없다

# 멜로디 검사 기준 — 경험칙이며 법적 판단이 아니다. 근거는 tools/tests 의 무작위 멜로디 실험 (README '멜로디 검사')
FLAG_RUN_RHYTHM = 8   # 음정 변화 + 리듬 비가 연속 8개(음 10개) 같으면 '검토 필요'
FLAG_RUN_PITCH = 12   # 리듬은 달라도 음정 변화가 연속 12개(음 13개) 같으면 '검토 필요'
MIN_MOVING = 4        # 같은 음 반복만인 구간은 세지 않는다 — 구간 안에 움직이는 음정(≠0)이 이만큼은 있어야
VERDICT_KO = {"pass": "통과", "review": "검토 필요", "unchecked": "멜로디 검사 못 함", "no_ref": "참고 곡 없음"}
PUSH_DEFAULT = ("pass", "no_ref")

# AudioSet 태그 → 프롬프트 말. 확률(10초 창 평균)이 기준 이상인 것만, 높은 순으로. 기준은 보정 전 값 — 로그의 확률을 보고 조정
TAG_GENRES = {
    "Pop music": "pop", "Rock music": "rock", "Punk rock": "punk rock", "Heavy metal": "metal",
    "Progressive rock": "progressive rock", "Grunge": "grunge", "Rock and roll": "rock and roll",
    "Hip hop music": "hip hop", "Rhythm and blues": "R&B", "Soul music": "soul", "Funk": "funk", "Disco": "disco",
    "Jazz": "jazz", "Blues": "blues", "Country": "country", "Folk music": "folk", "Reggae": "reggae", "Ska": "ska",
    "Classical music": "classical", "Electronic music": "electronic", "Electronic dance music": "EDM",
    "House music": "house", "Techno": "techno", "Trance music": "trance", "Dubstep": "dubstep",
    "Drum and bass": "drum and bass", "Ambient music": "ambient", "Electronica": "electronica",
    "New-age music": "new age", "Video game music": "video game music", "Soundtrack music": "cinematic soundtrack",
}
TAG_MOODS = {
    "Happy music": "happy and bright", "Exciting music": "energetic and exciting", "Sad music": "sad and emotional",
    "Tender music": "tender and warm", "Angry music": "intense and aggressive", "Scary music": "dark and tense",
    "Funny music": "playful",
}
TAG_INSTRUMENTS = {
    "Piano": "piano", "Electric piano": "electric piano", "Organ": "organ", "Synthesizer": "synthesizer",
    "Electric guitar": "electric guitar", "Acoustic guitar": "acoustic guitar", "Bass guitar": "electric bass",
    "Drum kit": "live drums", "Drum machine": "drum machine", "Violin, fiddle": "strings", "Cello": "strings",
    "String section": "strings", "Bowed string instrument": "strings", "Orchestra": "strings",
    "Brass instrument": "brass", "Trumpet": "brass", "Saxophone": "saxophone", "Flute": "flute", "Harp": "harp",
    "Glockenspiel": "glockenspiel", "Marimba, xylophone": "marimba", "Accordion": "accordion", "Ukulele": "ukulele",
    "Mandolin": "mandolin", "Banjo": "banjo", "Harmonica": "harmonica",
}
TAG_RULES = (("genres", TAG_GENRES, 0.15, 2), ("moods", TAG_MOODS, 0.10, 1), ("instruments", TAG_INSTRUMENTS, 0.10, 6))

PITCH_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
KK_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)  # Krumhansl–Kessler
KK_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

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


def resolve_seeds(seeds_text, count, has_ref):
    """--seeds 를 주면 그대로, 아니면 1..count (count 기본: 참고 곡이면 REF_COUNT, 아니면 1)."""
    if seeds_text:
        return parse_seeds(seeds_text)
    n = count if count is not None else (REF_COUNT if has_ref else 1)
    if not isinstance(n, int) or n < 1 or n > 64:
        raise ValueError(f"--count 는 1~64: {n}")
    return list(range(1, n + 1))


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


def find_reference(spec, input_root=KAGGLE_INPUT):
    """--ref 파일 경로 또는 'auto'(input_root 아래 오디오 파일이 딱 하나일 때) → Path."""
    if spec != "auto":
        p = Path(spec)
        if not p.is_file():
            raise FileNotFoundError(f"참고 곡 파일이 없음: {spec}")
        return p
    found = sorted(p for p in Path(input_root).rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT) \
        if Path(input_root).is_dir() else []
    if not found:
        raise FileNotFoundError(f"{input_root} 아래 오디오 파일이 없음 — 오른쪽 + Add Input → Upload 로 곡 하나를 올리기")
    if len(found) > 1:
        raise ValueError("오디오 파일이 여러 개 — --ref 로 하나를 고르기: " + " · ".join(str(p) for p in found[:10]))
    return found[0]


# ------------------------------------------------------------------ 참고 곡 분석 — 순수 함수 (시험 가능)
def _pearson(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else 0.0


def estimate_key(chroma):
    """12칸 크로마 평균(C, C#, …, B) → Krumhansl–Kessler 조성 {tonic, mode, r, margin(2등과 차이)}."""
    if len(chroma) != 12:
        raise ValueError("크로마는 12칸")
    scores = []
    for tonic in range(12):
        for mode, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            scores.append((_pearson(list(chroma), [prof[(i - tonic) % 12] for i in range(12)]), tonic, mode))
    scores.sort(reverse=True)
    r, tonic, mode = scores[0]
    return dict(tonic=PITCH_NAMES[tonic], mode=mode, r=round(r, 3), margin=round(r - scores[1][0], 3))


def abc_key_mode(key):
    """ABC 조표 'Am'·'F#m'·'Bb' → (으뜸음, 'minor'|'major')."""
    key = (key or "").strip()
    if not key:
        return None, None
    return (key[:-1], "minor") if key.endswith("m") else (key, "major")


def select_tags(probs, rules=TAG_RULES):
    """AudioSet 확률 {이름: p} → {genres: [...], moods: [...], instruments: [...], *_evidence: [(이름, p)]}.
    같은 프롬프트 말(예: Cello·String section → strings)은 한 번만."""
    out = {}
    for kind, table, minimum, limit in rules:
        words, evidence = [], []
        for label, p in sorted(((lb, probs[lb]) for lb in table if lb in probs), key=lambda x: -x[1]):
            if p < minimum or len(words) >= limit:
                break
            if table[label] in words:
                continue
            words.append(table[label])
            evidence.append((label, round(float(p), 3)))
        out[kind] = words
        out[kind + "_evidence"] = evidence
    return out


def build_prompt(tags, bpm, mode, extra=""):
    """[extra] + 장르 + 분위기 + 악기 + 장·단조 + BPM → 영어 스타일 프롬프트 (곡·가수 이름은 넣지 않는다)."""
    parts = []
    extra = (extra or "").strip().rstrip(".,")
    if extra:
        parts.append(extra)
    for kind in ("genres", "moods", "instruments"):
        parts += [w for w in tags.get(kind, []) if w.lower() not in extra.lower()]
    if mode in ("major", "minor"):
        parts.append(f"{mode} key")
    if bpm and bpm > 0:
        parts.append(f"{int(round(bpm))} BPM")
    return ", ".join(parts)


# ------------------------------------------------------------------ 멜로디 검사 — 순수 함수 (시험 가능)
def melody_line(score):
    """abc_tools Score 의 Vocal·Ins 음표 → 같은 시각에는 가장 높은 음 하나(주선율 근사) → [(onset 4분음표, MIDI 음)] 시간순."""
    top = {}
    for name in ("Vocal", "Ins"):
        voice = score.voices.get(name)
        if voice is None:
            continue
        for onset, pitch, _duration in voice.notes:
            if onset not in top or pitch > top[onset]:
                top[onset] = pitch
    return [(t, top[t]) for t in sorted(top)]


def melody_tokens(line):
    """→ (음정 변화 목록, 리듬 비 목록). i 번째 = 음 i→i+1 의 반음 차이, 리듬 비 = 간격(i→i+1) / 간격(i-1→i) (i=0 은 None).
    조옮김과 템포(반·두 배 박자 표기 포함)에 상관없다."""
    pitch = [line[i + 1][1] - line[i][1] for i in range(len(line) - 1)]
    ioi = [line[i + 1][0] - line[i][0] for i in range(len(line) - 1)]
    rhythm = [None] + [ioi[i] / ioi[i - 1] for i in range(1, len(ioi))]
    return pitch, rhythm


def longest_run(a, b, moving=None, min_moving=MIN_MOVING):
    """a·b 에서 연속으로 같은 가장 긴 구간 → (길이, a 끝 다음 위치, b 끝 다음 위치).
    moving(토큰)→bool 을 주면, 구간 안에 움직이는 토큰이 min_moving 개 이상인 구간만 센다."""
    best = (0, 0, 0)
    prev_len = [0] * (len(b) + 1)
    prev_mov = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur_len = [0] * (len(b) + 1)
        cur_mov = [0] * (len(b) + 1)
        ai = a[i - 1]
        add = 1 if moving is None or moving(ai) else 0
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                length = prev_len[j - 1] + 1
                mov = prev_mov[j - 1] + add
                cur_len[j], cur_mov[j] = length, mov
                if length > best[0] and (moving is None or mov >= min_moving):
                    best = (length, i, j)
        prev_len, prev_mov = cur_len, cur_mov
    return best


def _ngrams(seq, n):
    return {tuple(seq[i:i + n]) for i in range(len(seq) - n + 1) if any(seq[i:i + n])}


def compare_melodies(ref, gen, ref_bpm, gen_bpm):
    """두 멜로디 줄 → 겹침 지표. 음이 3개 미만이면 None."""
    if len(ref) < 3 or len(gen) < 3:
        return None
    rp, rr = melody_tokens(ref)
    gp, gr = melody_tokens(gen)
    rt, gt = list(zip(rp[1:], rr[1:])), list(zip(gp[1:], gr[1:]))
    run_r, r_end, g_end = longest_run(rt, gt, moving=lambda t: t[0] != 0)
    run_p, rp_end, gp_end = longest_run(rp, gp, moving=lambda t: t != 0)

    def at(line, note, bpm):
        return round(float(line[note][0]) * 60 / bpm, 1) if bpm else None

    shared = _ngrams(gp, 6) & _ngrams(rp, 6)
    gen_grams = _ngrams(gp, 6)
    return dict(
        run_rhythm=run_r, run_pitch=run_p,
        # 리듬 토큰 k 는 음 k+1→k+2, 음정 토큰 k 는 음 k→k+1 → 구간 첫 음 위치
        rhythm_ref_at_s=at(ref, r_end - run_r + 1, ref_bpm) if run_r else None,
        rhythm_gen_at_s=at(gen, g_end - run_r + 1, gen_bpm) if run_r else None,
        pitch_ref_at_s=at(ref, rp_end - run_p, ref_bpm) if run_p else None,
        pitch_gen_at_s=at(gen, gp_end - run_p, gen_bpm) if run_p else None,
        shared_pitch_6grams=round(len(shared) / len(gen_grams), 3) if gen_grams else 0.0,
        ref_notes=len(ref), gen_notes=len(gen),
    )


def judge(metrics, has_ref=True):
    """지표 → 'pass' | 'review' | 'unchecked' | 'no_ref' (경험칙 — 법적 판단이 아님)."""
    if not has_ref:
        return "no_ref"
    if metrics is None:
        return "unchecked"
    if metrics["run_rhythm"] >= FLAG_RUN_RHYTHM or metrics["run_pitch"] >= FLAG_RUN_PITCH:
        return "review"
    return "pass"


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
    env.pop("GITHUB_TOKEN", None)  # 생성·분석 프로세스는 토큰이 필요 없다
    env.update(PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", HF_HOME=str(HF_HOME), HF_HUB_DISABLE_PROGRESS_BARS="1")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def setup(force=False):
    """설치·점검. 이미 같은 고정값으로 끝났으면 건너뜀 → GPU 목록."""
    t0 = time.time()
    rows, err = query_gpus()
    log("ENV", python=sys.version.split()[0], gpus=len(rows), ram_gib=round(ram_gib(), 1),
        disk_free_gb=round(disk_free_gb(WORK), 1), ffmpeg=bool(shutil.which("ffmpeg")), work=WORK, out=OUT)
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
    want = {"tool": TOOL_VERSION, "wheel_sha256": WHEEL_SHA256, "skill_commit": SKILL_COMMIT, "python": PYTHON_VERSION,
            "extra": list(EXTRA_PACKAGES), "skill_files": sorted(SKILL_FILES)}
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
    run_cmd([sys.executable, "-m", "uv", "pip", "install", "--python", VENV_PY, wheel, *EXTRA_PACKAGES], "SETUP",
            "torch 2.10·transformers·librosa 설치 실패 → 위 오류 줄 확인 (디스크·네트워크)")
    freeze = subprocess.run([sys.executable, "-m", "uv", "pip", "freeze", "--python", str(VENV_PY)],
                            capture_output=True, text=True)
    (WORK / "freeze.txt").write_text(freeze.stdout, encoding="utf-8")  # 실제 설치 버전 기록 (재현용)
    for path, digest in SKILL_FILES.items():
        download(SKILL_URL.format(commit=SKILL_COMMIT, path=path), SKILL_ROOT / path, digest, "SETUP")
    run_cmd([VENV_PY, Path(__file__).resolve(), "check"], "CHECK",
            "이 GPU 에서 공식 BF16 경로가 안 됨 → 위 [CHECK] 줄 확인 (대안: yue2.cpp GGUF 포팅)", env=worker_env())
    READY.write_text(json.dumps(want | {"created": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False),
                     encoding="utf-8")
    log("SETUP", status="done", minutes=round((time.time() - t0) / 60, 1), venv=VENV, freeze=WORK / "freeze.txt")
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
    try:
        import librosa
        log("CHECK", librosa=librosa.__version__)
    except ImportError as e:
        bad += 1
        log("CHECK", logging.ERROR, error=f"librosa 없음: {e}")
    if bad:
        die("CHECK", error=f"점검 실패 {bad}개", hint="위 [CHECK] 줄 · GPU 문제면 대안: yue2.cpp(GGUF) 포팅")
    return 0


# ------------------------------------------------------------------ venv 안 — 참고 곡 분석
def audio_features(wav):
    """librosa(torch 없이) → (공개용 수치, 조성, 템포, 22.05kHz 모노 신호, sr)."""
    import numpy as np
    import librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    seconds = len(y) / sr
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    key = estimate_key([float(x) for x in librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)])
    rms_db = 20 * np.log10(librosa.feature.rms(y=y)[0] + 1e-9)
    feats = dict(seconds=round(seconds, 1), tempo_bpm=round(tempo, 1), key=f"{key['tonic']} {key['mode']}", key_r=key["r"],
                 key_margin=key["margin"], loudness_db=round(float(rms_db.mean()), 1),
                 dynamic_range_db=round(float(np.percentile(rms_db, 95) - np.percentile(rms_db, 10)), 1),
                 brightness_hz=round(float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()), 0),
                 onsets_per_s=round(len(librosa.onset.onset_detect(y=y, sr=sr)) / max(seconds, 1e-9), 2))
    return feats, key, tempo, y, sr


def separate_stems(wav, device):
    """(venv) Demucs htdemucs_6s → ({트랙: (2, N) float32 배열}, 샘플레이트, 정보). 파일로 쓰지 않는다 — 수치만 재고 버림.
    정규화는 demucs.separate 와 같다 (곡 평균·표준편차로 나눈 뒤 되돌림)."""
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    model = get_model(DEMUCS_MODEL)
    model.eval()
    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    if sr != model.samplerate:
        import librosa
        data = librosa.resample(data.T, orig_sr=sr, target_sr=model.samplerate).T
        sr = model.samplerate
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    mix = torch.from_numpy(np.ascontiguousarray(data[:, :2].T))
    ref = mix.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.inference_mode():
        sources = apply_model(model, ((mix - mean) / std)[None], device=device, shifts=1, split=True, overlap=0.25,
                              progress=False)[0]
    sources = sources * std + mean
    arrays = {name: src.cpu().numpy() for name, src in zip(model.sources, sources)}
    info = dict(model=DEMUCS_MODEL, sources=list(model.sources), samplerate=sr)
    del model, sources
    if device != "cpu":
        torch.cuda.empty_cache()
    return arrays, sr, info


def ast_probs(y16, device):
    """16kHz 모노 → AudioSet 태그 확률(10초 창, 최대 36개 창 평균) → ({이름: p}, 정보)."""
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    fe = ASTFeatureExtractor.from_pretrained(AST_MODEL)
    model = ASTForAudioClassification.from_pretrained(AST_MODEL).to(device).eval()
    win = 16000 * 10
    starts = list(range(0, max(1, len(y16) - win + 1), win)) or [0]
    if len(starts) > 36:
        starts = [starts[round(i * (len(starts) - 1) / 35)] for i in range(36)]
    total, n = None, 0
    with torch.inference_mode():
        for k in range(0, len(starts), 8):
            batch = [y16[s:s + win] for s in starts[k:k + 8]]
            inputs = fe(batch, sampling_rate=16000, return_tensors="pt").to(device)
            p = torch.sigmoid(model(**inputs).logits).float().cpu()
            total = p.sum(0) if total is None else total + p.sum(0)
            n += p.shape[0]
    mean = (total / n).tolist()
    labels = model.config.id2label
    info = dict(model=AST_MODEL, revision=getattr(model.config, "_commit_hash", None), windows=n)
    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    return {labels[i]: mean[i] for i in range(len(mean))}, info


def analyze(args):
    """(venv) 참고 곡 → <이름>/analysis.json·prompt.txt (공개용, 파일 이름 없음) + <이름>/reference/transcription (로컬만: 받아 적은 악보).
    원곡을 바꾼 wav 는 WORK/ref_audio/ (/tmp)."""
    import torch
    import librosa
    name_dir = Path(args.out) / args.name
    ref_dir = name_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    src = Path(args.ref)
    t0 = time.time()
    wav = WORK / "ref_audio" / f"{args.name}.wav"  # 원곡 사본은 /tmp 에만 — Kaggle 출력(/kaggle/working)에 남기지 않음
    wav.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-ac", "2", "-ar", "44100",
                            str(wav)])
        if r.returncode:
            die("REF", error="ffmpeg 로 읽지 못함", hint="mp3·wav·flac 로 바꿔 다시 올리기")
    elif src.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg"):
        wav = src
    else:
        die("REF", error="ffmpeg 가 없어 이 형식을 못 읽음", hint="wav·flac·mp3 로 올리기")
    ref_sha = sha256_file(src)
    log("REF", file=src.name, sha256=ref_sha[:12], bytes=src.stat().st_size, note="파일 이름은 이 로그(로컬)에만 — 저장소에는 sha256 만")

    # 1) librosa — 템포·조성·음량 (조성·템포는 SheetSage2 가 되면 그 값을 먼저 씀)
    feats, _key, tempo, y, sr = audio_features(wav)
    if feats["seconds"] < 20:
        log("ANALYZE", logging.WARNING, warning="곡이 20초보다 짧음 — 분석이 부정확할 수 있음", seconds=feats["seconds"])
    log("ANALYZE", step="librosa", **feats)

    # 2) AudioSet 태그 — 장르·분위기·악기
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tags, ast_info, top = {"genres": [], "moods": [], "instruments": []}, {}, []
    try:
        y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
        probs, ast_info = ast_probs(y16, device)
        tags = select_tags(probs)
        top = [(lb, round(p, 3)) for lb, p in sorted(probs.items(), key=lambda x: -x[1])[:15]]
        missing = sorted(lb for _, table, _, _ in TAG_RULES for lb in table if lb not in probs)
        log("ANALYZE", step="tags", windows=ast_info.get("windows"), top=";".join(f"{lb}:{p}" for lb, p in top))
        for kind in ("genres", "moods", "instruments"):
            log("ANALYZE", step="tags", kind=kind, picked=",".join(tags[kind]) or "-",
                evidence=";".join(f"{lb}:{p}" for lb, p in tags[kind + "_evidence"]) or "-")
        if missing:
            log("ANALYZE", logging.WARNING, warning="태그 표의 이름 중 모델에 없는 것", names=",".join(missing))
    except Exception as e:  # noqa: BLE001
        log("ANALYZE", logging.WARNING, step="tags", error=f"{type(e).__name__}: {e}",
            hint="장르·악기를 --style-extra 로 직접 (예: \"J-pop band, piano, electric guitar, strings\")")

    # 3) SheetSage2 — 원곡 멜로디 받아 적기 (멜로디 검사용, 템포·조성도)
    cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    tdir = ref_dir / "transcription"
    sheet = dict(ok=False)
    mel, form = None, []
    if not (tdir / "score.abc").is_file():
        if tdir.exists():
            shutil.rmtree(tdir)
        cmd = [sys.executable, str(SKILL_DIR / "transcribe_cover.py"), str(wav), "--output", str(tdir), "--task", "melody-full",
               "--device", device, "--dtype", "bf16" if device == "cuda" and cap >= PATCH_BELOW else "fp32"]
        log("ANALYZE", step="sheetsage2", device=device, dtype=cmd[-1], task="melody-full")
        t1 = time.time()
        rc = subprocess.run(cmd).returncode
        log("ANALYZE", step="sheetsage2", exit=rc, seconds=round(time.time() - t1, 1))
    if (tdir / "score.abc").is_file():
        try:
            sys.path.insert(0, str(SKILL_DIR))
            from abc_tools import parse_abc
            score_text = (tdir / "score.abc").read_text(encoding="utf-8")
            score = parse_abc(score_text)
            tonic, mode = abc_key_mode(score.voices["Vocal"].key)
            sheet = dict(ok=True, bpm=score.bpm, key=f"{tonic} {mode}", melody_notes=len(melody_line(score)))
            mel = ma.melody_features(melody_line(score))  # 음표 자체는 공개 파일에 넣지 않는다 — 특징 숫자만
            form = ma.form_sections(score_text)
        except Exception as e:  # noqa: BLE001
            sheet = dict(ok=False, error=f"{type(e).__name__}: {e}")
    else:
        failure = tdir / "failure.json"
        sheet["error"] = json.loads(failure.read_text(encoding="utf-8")).get("error") if failure.is_file() else "score.abc 없음"
    log("ANALYZE", logging.INFO if sheet["ok"] else logging.WARNING, step="sheetsage2", **sheet,
        **({} if sheet["ok"] else {"hint": "멜로디 검사 못 함 → 이 곡들은 --push-unchecked 일 때만 올림 · 다시 받아 적기는 --reanalyze"}))

    if mel:
        log("ANALYZE", step="melody", **mel, form=",".join(form) or "-")

    # 3b) 트랙 분리 → 트랙별 수치 (분리한 원곡 트랙은 메모리에서만 — 파일·결과 곡에 남기지 않음)
    stems, stems_info, stems_error = {}, {}, None
    try:
        t2 = time.time()
        arrays, stem_sr, stems_info = separate_stems(wav, device)
        stems = ma.add_shares({name: ma.stem_features(arr, stem_sr) for name, arr in arrays.items()})
        del arrays
        for name, f in stems.items():
            f.pop("energy", None)
            f["words"] = ma.stem_words(f, name)
            log("ANALYZE", step="stems", stem=name, share=f["share"], centroid_hz=f.get("centroid_hz", "-"),
                width=f.get("width", "-"), crest_db=f.get("crest_db", "-"), tail=f.get("tail", "-"),
                flatness=f.get("flatness", "-"), words=";".join(f"{k}:{v}" for k, v in f["words"].items()))
        log("ANALYZE", step="stems", model=stems_info.get("model"), seconds=round(time.time() - t2, 1))
    except Exception as e:  # noqa: BLE001
        stems_error = f"{type(e).__name__}: {e}"
        log("ANALYZE", logging.WARNING, step="stems", error=stems_error,
            hint="트랙 분리 없이 진행 — Claude 에게 보낼 글에는 태그·멜로디 특징만")

    # 4) 프롬프트 — 장·단조는 SheetSage2 값만 넣는다 (librosa 크로마 추정은 합성 시험에서 A단조를 A장조로 읽음 → 기록만)
    if sheet["ok"]:
        bpm, mode, src_name, key_from = sheet["bpm"], sheet["key"].split()[1], "sheetsage2", "sheetsage2"
    else:
        bpm, mode, src_name, key_from = tempo, None, "librosa", "생략 (librosa 추정은 기록만)"
    style = build_prompt(tags, bpm, mode, args.style_extra)
    if not (tags["genres"] or tags["instruments"] or args.style_extra.strip()):
        log("PROMPT", logging.WARNING, warning="장르·악기 태그가 없음 — 프롬프트가 템포·조성뿐",
            hint="--style-extra 로 장르·악기를 더하기")
    public = dict(tool=TOOL_VERSION, created=datetime.now().isoformat(timespec="seconds"),
                  reference=dict(sha256=ref_sha, seconds=feats["seconds"]),
                  librosa=feats, tags={k: v for k, v in tags.items()}, tags_top15=top, ast=ast_info,
                  sheetsage2=dict(sheet),
                  stems=stems, stems_info=stems_info, stems_error=stems_error,
                  melody_features=mel, melody_words=ma.melody_words(mel), form=form,
                  prompt=dict(style=style, extra=args.style_extra, tempo_from=src_name, key_from=key_from),
                  note="원곡 파일·원곡 악보·파일 이름은 저장소에 올리지 않는다. 연구·교육용.")
    (name_dir / "analysis.json").write_text(json.dumps(public, ensure_ascii=False, indent=1), encoding="utf-8")
    (name_dir / "prompt.txt").write_text(style + "\n", encoding="utf-8")
    log("PROMPT", style=style, tempo_from=src_name, bpm=round(float(bpm), 1), mode=mode or "-", key_from=key_from,
        minutes=round((time.time() - t0) / 60, 1))
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
                            "-b:a", MP3_BITRATE, str(mp3)])
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


# ------------------------------------------------------------------ 멜로디 검사 실행 (Kaggle 기본 파이썬 — abc_tools 는 표준 라이브러리만)
def run_similarity(name_dir, parse_abc=None):
    """<이름>/seed-*/score.abc ↔ <이름>/reference/transcription/score.abc → seed 마다 similarity.json → {seed 폴더 이름: 기록}."""
    if parse_abc is None:
        if not (SKILL_DIR / "abc_tools.py").is_file():
            die("SIMILARITY", error="abc_tools.py 없음", hint="setup 을 먼저 (run 이 알아서 함)")
        sys.path.insert(0, str(SKILL_DIR))
        from abc_tools import parse_abc  # noqa: F811
    name_dir = Path(name_dir)
    has_ref = (name_dir / "reference").is_dir()
    ref = None
    ref_abc = name_dir / "reference" / "transcription" / "score.abc"
    if ref_abc.is_file():
        try:
            s = parse_abc(ref_abc.read_text(encoding="utf-8"))
            ref = (melody_line(s), s.bpm)
        except Exception as e:  # noqa: BLE001
            log("SIMILARITY", logging.WARNING, error=f"원곡 악보를 읽지 못함: {type(e).__name__}: {e}")
    seeds = sorted((d for d in name_dir.glob("seed-*") if (d / "kaggle_run.json").is_file()), key=lambda d: d.name)
    gens = {}
    for d in seeds:
        if (d / "score.abc").is_file():
            try:
                s = parse_abc((d / "score.abc").read_text(encoding="utf-8"))
                gens[d.name] = (melody_line(s), s.bpm)
            except Exception as e:  # noqa: BLE001
                log("SIMILARITY", logging.WARNING, seed=d.name, error=f"악보를 읽지 못함: {type(e).__name__}: {e}")
    # 같은 프롬프트로 만든 곡끼리 — 우연히 겹치는 정도 (판정에는 안 쓰고 함께 적기만)
    baseline = dict(run_rhythm=0, run_pitch=0, pairs=0)
    names = sorted(gens)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = compare_melodies(gens[a][0], gens[b][0], gens[a][1], gens[b][1])
            if m:
                baseline["pairs"] += 1
                baseline["run_rhythm"] = max(baseline["run_rhythm"], m["run_rhythm"])
                baseline["run_pitch"] = max(baseline["run_pitch"], m["run_pitch"])
    results = {}
    for d in seeds:
        m = compare_melodies(ref[0], gens[d.name][0], ref[1], gens[d.name][1]) if ref and d.name in gens else None
        verdict = judge(m, has_ref)
        record = dict(tool=TOOL_VERSION, verdict=verdict, verdict_ko=VERDICT_KO[verdict], melody=m,
                      baseline_between_generated=baseline,
                      thresholds=dict(run_rhythm=FLAG_RUN_RHYTHM, run_pitch=FLAG_RUN_PITCH, min_moving=MIN_MOVING),
                      method="만든 곡 악보(YuE2 계획) ↔ 원곡을 SheetSage2 로 받아 적은 멜로디. 같은 시각 가장 높은 음, "
                             "조옮김·템포 무관(음정 변화·리듬 비), 연속으로 같은 가장 긴 구간",
                      note="경험칙이며 법적 판단이 아니다. 받아 적기 오차로 놓칠 수 있다.")
        (d / "similarity.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        log("SIMILARITY", logging.INFO if verdict in PUSH_DEFAULT else logging.WARNING, seed=d.name,
            verdict=VERDICT_KO[verdict], run_rhythm=m["run_rhythm"] if m else "-", run_pitch=m["run_pitch"] if m else "-",
            ref_at_s=m["rhythm_ref_at_s"] if m else "-", gen_at_s=m["rhythm_gen_at_s"] if m else "-",
            baseline=f"{baseline['run_rhythm']}/{baseline['run_pitch']}")
        results[d.name] = record
    return results


# ------------------------------------------------------------------ 저장소 올리기 (Kaggle 기본 파이썬)
def git_env():
    """→ (환경, 토큰 있음). 토큰은 GIT_ASKPASS 스크립트가 환경 변수에서 읽어 git 에만 넘긴다 (스크립트 파일에는 토큰이 없다)."""
    askpass = WORK / "git_askpass.sh"
    askpass.parent.mkdir(parents=True, exist_ok=True)
    askpass.write_text('#!/bin/sh\ncase "$1" in\n  Username*) echo x-access-token ;;\n  *) echo "$GITHUB_TOKEN" ;;\nesac\n',
                       encoding="utf-8")
    askpass.chmod(0o700)
    env = os.environ.copy()
    env.update(GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT="0")
    return env, bool(os.environ.get("GITHUB_TOKEN"))


def git(*args, env=None, check=True, repo=None):
    r = subprocess.run(["git", "-C", str(repo or ROOT), "-c", f"user.name={GIT_NAME}", "-c", f"user.email={GIT_EMAIL}",
                        *args], env=env, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"git {args[0]} 실패 (exit {r.returncode}): {r.stderr.strip()[-300:]}")
    return r


def push_check():
    """오래 걸리는 생성 전에: 토큰·저장소·쓰기 권한 확인 (안 되면 멈춤)."""
    env, has_token = git_env()
    if not has_token:
        die("PUSH", error="GITHUB_TOKEN 없음",
            hint="Kaggle Add-ons → Secrets 에 GITHUB_TOKEN(music-ytb Contents: Read and write) → 셀 첫 줄에서 os.environ 에 넣기")
    try:
        url = git("remote", "get-url", "origin").stdout.strip()
    except RuntimeError as e:
        die("PUSH", error=f"{ROOT} 가 git 사본이 아님: {e}", hint="셀의 git clone 줄로 받은 사본의 도구를 실행")
    if REPO_SLUG not in url:
        die("PUSH", error=f"origin 이 {REPO_SLUG} 가 아님: {url}", hint="셀의 git clone 줄 확인")
    git("fetch", "-q", "origin", "main", env=env, check=False)
    r = git("push", "--dry-run", "origin", "refs/remotes/origin/main:refs/heads/main", env=env, check=False)
    if r.returncode:
        die("PUSH", error="쓰기 권한 확인 실패: " + (r.stderr.strip().splitlines() or ["?"])[-1][:200],
            hint="토큰의 Repository access 에 music-ytb · Permissions → Contents: Read and write")
    log("PUSH", check="ok", repo=REPO_SLUG)


def publish_files(name_dir, results, allow, include_flac=False):
    """올릴 파일 [(원본, 저장소 안 상대 경로)] — 허용 목록 방식: reference/(원곡·원곡 악보)는 어떤 경우에도 들어가지 않는다."""
    name_dir = Path(name_dir)
    rel = Path(RESULTS_DIR) / name_dir.name
    files = [(name_dir / f, rel / f) for f in NAME_FILES if (name_dir / f).is_file()]
    for seed, rec in sorted(results.items()):
        if rec["verdict"] not in allow:
            continue
        for f in SEED_FILES + (("audio.flac",) if include_flac else ()):
            src = name_dir / seed / f
            if src.is_file():
                files.append((src, rel / seed / f))
    assert not any("reference" in src.relative_to(name_dir).parts for src, _ in files)
    return files


def results_readme(name, results, allow, analysis):
    """results/<이름>/README.md 내용."""
    style = (analysis or {}).get("prompt", {}).get("style", "")
    lines = [f"# {name}", "",
             f"`tools/kaggle_music_yue2.py` {TOOL_VERSION} 로 만든 곡 ({datetime.now():%Y-%m-%d}). YuE2(m-a-p/YuE2-3B) 연주·작곡.",
             "참고 곡이 있으면 그 분석을 글(프롬프트)로만 넘겼다 — 원곡 파일·원곡 악보·파일 이름은 이 저장소에 없다.",
             "멜로디 검사는 경험칙이며 법적 판단이 아니다. 연구·교육용.", ""]
    if style:
        lines += ["## 프롬프트", "", "```", style, "```", ""]
    lines += ["## 곡", "", "| 곡 | 길이 | 멜로디 검사 | 원곡과 같은 구간 (리듬까지 / 음정만) | 파일 |", "|---|---|---|---|---|"]
    baseline = None
    for seed, rec in sorted(results.items()):
        m = rec.get("melody") or {}
        baseline = rec.get("baseline_between_generated", baseline)
        run = f"{m['run_rhythm']} / {m['run_pitch']}" if m else "-"
        seconds = rec.get("seconds")
        length = f"{int(seconds // 60)}:{int(seconds % 60):02d}" if seconds else "-"
        if rec["verdict"] in allow:
            link = f"[mp3]({seed}/audio.mp3) · [악보]({seed}/score.abc)"
            status = rec["verdict_ko"]
        else:
            link = "—"
            status = f"{rec['verdict_ko']} — 올리지 않음"
        lines.append(f"| {seed} | {length} | {status} | {run} | {link} |")
    lines.append("")
    if baseline and baseline.get("pairs"):
        lines.append(f"같은 프롬프트로 만든 곡끼리 겹친 최대 (우연히 겹치는 정도): 리듬까지 {baseline['run_rhythm']} / 음정만 {baseline['run_pitch']}.")
    lines += [f"기준: 원곡과 리듬까지 같은 음정 변화가 연속 {FLAG_RUN_RHYTHM}개 이상, 또는 음정만 연속 {FLAG_RUN_PITCH}개 이상이면 '검토 필요'"
              f" (같은 음 반복만인 구간은 빼고, 움직이는 음정 {MIN_MOVING}개 이상).", ""]
    if analysis:
        lib = analysis.get("librosa", {})
        tags = analysis.get("tags", {})
        sheet = analysis.get("sheetsage2", {})
        evidence = "; ".join(f"{lb} {p}" for kind in ("genres", "moods", "instruments")
                             for lb, p in tags.get(kind + "_evidence", []))
        lines += ["## 참고 곡 분석 (숫자만)", "",
                  f"- 길이 {lib.get('seconds')}초 · 템포 {lib.get('tempo_bpm')} BPM(librosa)"
                  + (f" / {sheet.get('bpm')} BPM(SheetSage2)" if sheet.get("ok") else "")
                  + f" · 조성 {lib.get('key')}(librosa)" + (f" / {sheet.get('key')}(SheetSage2)" if sheet.get("ok") else ""),
                  f"- 태그(확률): {evidence}" if evidence else "- 태그 없음", ""]
    return "\n".join(lines)


def push_results(name_dir, results, allow, include_flac=False):
    """허용된 곡만 저장소 results/<이름>/ 에 복사 → 커밋 → pull --rebase → push (재시도). → 0 성공."""
    name_dir = Path(name_dir)
    for seed, rec in results.items():  # README 의 길이 칸
        run_json = name_dir / seed / "kaggle_run.json"
        if run_json.is_file():
            rec["seconds"] = json.loads(run_json.read_text(encoding="utf-8")).get("metrics", {}).get("seconds")
    files = publish_files(name_dir, results, allow, include_flac)
    songs = sorted({str(rel.parent.name) for _, rel in files if rel.name == "audio.mp3"})
    skipped = sorted(s for s, r in results.items() if r["verdict"] not in allow)
    log("PUSH", allow=",".join(sorted(allow)), songs=",".join(songs) or "-", skipped=",".join(skipped) or "-",
        files=len(files))
    if not songs:
        log("PUSH", logging.WARNING, warning="올릴 곡이 없음", hint="검사 결과를 보고 --push-flagged / --push-unchecked")
        return 1
    env, _ = git_env()
    for src, rel in files:
        mb = src.stat().st_size / 1e6
        if mb > MAX_PUSH_MB:
            log("PUSH", logging.WARNING, warning="GitHub 한도에 가까워 뺌", file=rel, mb=round(mb, 1))
            continue
        (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, ROOT / rel)
    analysis = None
    if (name_dir / "analysis.json").is_file():
        analysis = json.loads((name_dir / "analysis.json").read_text(encoding="utf-8"))
    dest = Path(RESULTS_DIR) / name_dir.name
    (ROOT / dest / "README.md").write_text(results_readme(name_dir.name, results, allow, analysis), encoding="utf-8")
    try:
        git("add", "--", str(dest))
        if git("diff", "--cached", "--quiet", check=False).returncode == 0:
            log("PUSH", status="바뀐 것 없음")
            return 0
        git("commit", "-q", "-m", f"음악: {name_dir.name} — {len(songs)}곡 올림 (kaggle_music_yue2 {TOOL_VERSION})"
            + (f", 안 올림 {len(skipped)}곡" if skipped else ""))
    except RuntimeError as e:
        log("PUSH", logging.ERROR, error=str(e), hint="push --name 으로 다시")
        return 1
    for attempt in range(1, PUSH_TRIES + 1):
        pull = git("pull", "-q", "--rebase", "origin", "main", env=env, check=False)
        push = git("push", "-q", "origin", "HEAD:main", env=env, check=False)
        if pull.returncode == 0 and push.returncode == 0:
            head = git("rev-parse", "--short", "HEAD").stdout.strip()
            log("PUSH", status="ok", commit=head, path=dest, songs=len(songs))
            return 0
        err = (push.stderr or pull.stderr).strip().splitlines()
        log("PUSH", logging.WARNING, attempt=attempt, error=(err[-1] if err else "?")[:200])
        time.sleep(2 ** attempt)
    log("PUSH", logging.ERROR, error="올리기 실패", hint=f"결과는 {name_dir} 에 있음 → push --name {name_dir.name} 로 다시")
    return 1


def push_allow(args):
    return set(PUSH_DEFAULT) | ({"review"} if args.push_flagged else set()) | ({"unchecked"} if args.push_unchecked else set())


# ------------------------------------------------------------------ 하위 프로세스 (Kaggle 기본 파이썬)
def stream(proc, prefix, sink):
    for line in proc.stdout:  # text=True → '\r' 진행 줄도 한 줄씩 (공식 Progress 는 파이프에서 5초에 한 줄)
        sink(f"{prefix} | {line.rstrip()}")


def run_streamed(cmd, env, prefix):
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    stream(proc, prefix, LOG.info)
    return proc.wait()


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


def ensure_analysis(name_dir, ref, args, gpu):
    """참고 곡 분석이 없으면 (venv) _analyze 를 돌린다. 같은 이름에 다른 곡(sha256 다름)이면 멈춤 → analysis.json 내용."""
    ref_sha = sha256_file(ref)
    analysis_json = name_dir / "analysis.json"
    if analysis_json.is_file() and not args.reanalyze:
        old_sha = json.loads(analysis_json.read_text(encoding="utf-8")).get("reference", {}).get("sha256")
        if old_sha != ref_sha:
            die("INPUT", error=f"이름 {name_dir.name} 은 다른 참고 곡으로 이미 분석함",
                hint="이름을 바꾸거나(--name) --reanalyze", old=str(old_sha)[:12], new=ref_sha[:12])
        log("ANALYZE", status="skip", reason="같은 곡을 이미 분석함 (--reanalyze 로 다시)", path=analysis_json)
    else:
        if args.reanalyze and (name_dir / "reference").exists():
            shutil.rmtree(name_dir / "reference")
        cmd = [str(VENV_PY), str(Path(__file__).resolve()), "_analyze", "--ref", str(ref), "--name", name_dir.name,
               "--out", str(name_dir.parent), "--style-extra", args.style_extra, "--log-level", args.log_level]
        if run_streamed(cmd, worker_env(gpu), "analyze"):
            die("ANALYZE", error="참고 곡 분석 실패", hint="위 analyze 줄의 오류 확인")
    return json.loads(analysis_json.read_text(encoding="utf-8"))


CLAUDE_BEGIN = "----- Claude 에게 보낼 글 (여기부터 복사 — 사진을 함께 첨부) -----"
CLAUDE_END = "----- (여기까지) -----"


def analyze_command(args):
    """참고 곡 분석(트랙 분리 포함) → <이름>/claude_request.txt (멘트 포함) → 출력. 생성은 하지 않는다."""
    if not NAME_RE.fullmatch(args.name):
        die("INPUT", error=f"이름 {args.name!r}", hint="영문·숫자·_.- 만 (예: my-song-01)")
    try:
        ref = find_reference(args.ref)
    except (FileNotFoundError, ValueError) as e:
        die("INPUT", error=str(e), hint="--ref <곡 파일> 또는 --ref auto")
    comment = ""
    if args.comment_file:
        try:
            comment = Path(args.comment_file).read_text(encoding="utf-8")
        except OSError as e:
            die("INPUT", error=f"멘트 파일을 못 읽음: {e}", hint="--comment-file 경로 확인")
    rows = setup(force=args.reinstall)
    name_dir = Path(args.out) / args.name
    analysis = ensure_analysis(name_dir, ref, args, rows[0]["index"])
    text = ma.claude_request(analysis, comment, args.lyrics)
    request = name_dir / "claude_request.txt"
    request.write_text(text, encoding="utf-8")
    log("CLAUDE", file=request, lyrics=args.lyrics, comment_chars=len(comment.strip()), chars=len(text),
        stems=len(analysis.get("stems") or {}), melody=bool(analysis.get("melody_features")))
    print(CLAUDE_BEGIN)
    print(text, end="")
    print(CLAUDE_END)
    return 0


def run(args):
    t0 = time.time()
    if not NAME_RE.fullmatch(args.name):
        die("INPUT", error=f"이름 {args.name!r}", hint="영문·숫자·_.- 만 (예: bgm-lofi-01)")
    ref = None
    if args.ref:
        try:
            ref = find_reference(args.ref)
        except (FileNotFoundError, ValueError) as e:
            die("INPUT", error=str(e), hint="--ref /kaggle/input/<데이터셋>/<파일> 또는 --ref auto")
    if not args.style.strip() and not ref:
        die("INPUT", error="스타일이 비었음", hint='--style "calm piano, soft drums, 80 BPM" 또는 --ref 로 참고 곡')
    try:
        seeds = resolve_seeds(args.seeds, args.count, bool(ref))
        if args.mode == "instrumental":
            section_lyrics(args.sections)
    except ValueError as e:
        die("INPUT", error=str(e), hint="--seeds 1,2 또는 --count 4 · --sections Intro,Verse,Chorus,Outro")
    if args.mode == "song":
        if not args.lyrics_file or not Path(args.lyrics_file).is_file():
            die("INPUT", error="노래 모드는 가사 파일이 필요", hint="--lyrics-file lyrics.txt ([Verse]·[Chorus] 태그 + 가사)")
        if not Path(args.lyrics_file).read_text(encoding="utf-8").strip():
            die("INPUT", error="가사 파일이 비었음", hint="[Verse] 아래에 가사")
    rows = setup(force=args.reinstall)
    if args.push:
        push_check()
    usable = [g["index"] for g in rows]
    if args.gpus != "all":
        want = [int(x) for x in args.gpus.split(",") if x.strip()]
        usable = [i for i in want if i in usable]
        if not usable:
            die("INPUT", error=f"--gpus {args.gpus} 가 없음", hint=f"있는 GPU: {[g['index'] for g in rows]}")
    name_dir = Path(args.out) / args.name

    # 참고 곡 → 분석 → 프롬프트
    style_from = "--style"
    if ref:
        analysis = ensure_analysis(name_dir, ref, args, usable[0])
        if args.style.strip():
            style_from = "--style (참고 곡 분석은 멜로디 검사에만)"
        else:
            args.style, style_from = analysis["prompt"]["style"], "참고 곡 분석"
    log("PROMPT", style=args.style, source=style_from)

    plan = assign_seeds(seeds, usable)
    log("RUN", mode=args.mode, name=args.name, seeds=",".join(map(str, seeds)),
        gpus=";".join(f"{g}:{','.join(map(str, s))}" for g, s in plan.items()), out=name_dir)
    if len(plan) > 1 and ram_gib() < 24:
        log("RUN", logging.WARNING, warning="RAM 이 적어 두 GPU 동시 로드가 빠듯함", ram_gib=round(ram_gib(), 1),
            hint="멈추면 --gpus 0")
    # 모델을 먼저 한 번 받는다 (두 프로세스가 같은 7.8GB 를 동시에 받지 않게)
    first = next(iter(plan))
    if run_streamed(worker_cmd(args, plan[first], fetch_only=True), worker_env(first), "fetch"):
        die("LOAD", error="모델 받기 실패", hint="위 fetch 줄의 [LOAD] 오류 확인")
    procs, threads = [], []
    for gpu, gpu_seeds in plan.items():
        p = subprocess.Popen(worker_cmd(args, gpu_seeds), env=worker_env(gpu),
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        t = threading.Thread(target=stream, args=(p, f"gpu{gpu}", LOG.info), daemon=True)
        t.start()
        procs.append((gpu, p))
        threads.append(t)
    codes = {gpu: p.wait() for gpu, p in procs}
    for t in threads:
        t.join()

    # 멜로디 검사 → 올리기
    results = run_similarity(name_dir)
    files = sorted(str(p) for p in name_dir.glob("seed-*/audio.flac"))
    counts = {v: sum(1 for r in results.values() if r["verdict"] == v) for v in VERDICT_KO}
    log("DONE", minutes=round((time.time() - t0) / 60, 1), files=len(files),
        verdicts=",".join(f"{VERDICT_KO[v]}={n}" for v, n in counts.items() if n) or "-",
        exit=",".join(f"gpu{g}={c}" for g, c in codes.items()),
        hint="-" if not any(codes.values()) else "실패한 GPU 의 [GEN] 오류 줄 확인")
    for f in files:
        log("DONE", file=f)
    push_rc = push_results(name_dir, results, push_allow(args), args.push_flac) if args.push else 0
    return 1 if any(codes.values()) or push_rc else 0


# ------------------------------------------------------------------ 명령줄
def add_gen_args(p):
    p.add_argument("--mode", choices=("instrumental", "song"), default="instrumental")
    p.add_argument("--style", default="", help='영어 권장. 장르·악기·분위기·BPM (예: "calm lo-fi piano, soft drums, 80 BPM")')
    p.add_argument("--lyrics-file", help="song: 가사 파일 ([Verse]·[Chorus] 태그)")
    p.add_argument("--sections", default=DEFAULT_SECTIONS, help=f"instrumental: 계획 섹션 (기본 {DEFAULT_SECTIONS} — 늘리면 길어짐)")
    p.add_argument("--plan-mode", choices=("full", "melody"), default="full", help="instrumental: 악보에 화음까지(full) / 선율만")
    p.add_argument("--cot", choices=("full", "melody", "off"), default="full", help="song: 계획 방식 (공식 기본 full)")
    p.add_argument("--seeds", default=None, help="쉼표로 여러 개 — GPU 마다 나눠 동시에 (없으면 1..--count)")
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


def add_push_args(p):
    p.add_argument("--push-flagged", action="store_true", help="멜로디 검사 '검토 필요' 곡도 올림")
    p.add_argument("--push-unchecked", action="store_true", help="멜로디 검사를 못 한 곡도 올림")
    p.add_argument("--push-flac", action="store_true", help="flac 도 올림 (곡당 약 30MB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="YuE2 노래·연주곡 — Kaggle GPU (연구·교육용)")
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="설치(처음 한 번) → [참고 곡 분석] → 생성 → 멜로디 검사 → [저장소에 올리기]")
    add_gen_args(r)
    r.add_argument("--ref", help="참고 곡 파일, 또는 auto (= /kaggle/input 아래 오디오 파일 하나)")
    r.add_argument("--count", type=int, default=None, help=f"곡 수 (--seeds 가 없을 때, 기본: 참고 곡이면 {REF_COUNT} 아니면 1)")
    r.add_argument("--style-extra", default="", help='참고 곡 프롬프트 앞에 더할 말 (예: "J-pop band")')
    r.add_argument("--reanalyze", action="store_true", help="같은 이름으로 분석한 것이 있어도 다시")
    r.add_argument("--push", action="store_true", help=f"결과를 {REPO_SLUG} {RESULTS_DIR}/<이름>/ 에 (GITHUB_TOKEN 필요)")
    add_push_args(r)
    r.add_argument("--gpus", default="all", help="쓸 GPU (예: 0 또는 0,1)")
    r.add_argument("--reinstall", action="store_true", help="venv 를 지우고 다시 설치")
    s = sub.add_parser("setup", help="설치·GPU 점검만")
    s.add_argument("--reinstall", action="store_true")
    s.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    c = sub.add_parser("check", help="(venv 안) GPU 점검")
    c.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    for cmd_name, help_text in (("similarity", "이미 만든 곡들의 멜로디 검사를 다시"), ("push", "이미 만든 곡을 저장소에")):
        q = sub.add_parser(cmd_name, help=help_text)
        q.add_argument("--name", required=True)
        q.add_argument("--out", default=str(OUT))
        q.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
        if cmd_name == "push":
            add_push_args(q)
    z = sub.add_parser("analyze", help="참고 곡 분석(트랙 분리 포함) → Claude 에게 보낼 글 (생성은 안 함)")
    z.add_argument("--ref", required=True, help="참고 곡 파일, 또는 auto")
    z.add_argument("--name", default=None, help="결과 폴더 이름 (기본: ref-날짜시각)")
    z.add_argument("--out", default=str(OUT))
    z.add_argument("--comment-file", help="이 곡을 듣고 느낀 멘트 (글 파일)")
    z.add_argument("--lyrics", action="store_true", help="가사 넣기 — Claude 에게 가사도 부탁")
    z.add_argument("--style-extra", default="")
    z.add_argument("--reanalyze", action="store_true")
    z.add_argument("--reinstall", action="store_true")
    z.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    a = sub.add_parser("_analyze", help=argparse.SUPPRESS)
    a.add_argument("--ref", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--out", default=str(OUT))
    a.add_argument("--style-extra", default="")
    a.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    w = sub.add_parser("_worker", help=argparse.SUPPRESS)
    add_gen_args(w)
    w.add_argument("--fetch-only", action="store_true")
    args = ap.parse_args(argv)
    if args.command == "run":
        prefix = "bgm" if args.mode == "instrumental" else "song"
        args.name = args.name or f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}"
        setup_logging(args.log_level, Path(args.out) / "music.log")
        return run(args)
    if args.command == "analyze":
        args.name = args.name or f"ref-{datetime.now():%Y%m%d-%H%M%S}"
        setup_logging(args.log_level, Path(args.out) / "music.log")
        return analyze_command(args)
    if args.command in ("similarity", "push"):
        setup_logging(args.log_level, Path(args.out) / "music.log")
        name_dir = Path(args.out) / args.name
        if not name_dir.is_dir():
            die("INPUT", error=f"{name_dir} 없음", hint="run 에서 쓴 --name 그대로")
        results = run_similarity(name_dir)
        if args.command == "push":
            push_check()
            return push_results(name_dir, results, push_allow(args), args.push_flac)
        return 0
    setup_logging(args.log_level)
    if args.command == "setup":
        setup(force=args.reinstall)
        return 0
    if args.command == "check":
        return check()
    if args.command == "_analyze":
        os.environ.setdefault("HF_HOME", str(HF_HOME))
        return analyze(args)
    return worker(args)


if __name__ == "__main__":
    sys.exit(main())
