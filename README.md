# music-ytb

오픈소스 작곡 AI **YuE2**(HKUST·M-A-P, 2026-09 공개, `m-a-p/YuE2-3B`)로 노래·연주곡을 **Kaggle 무료 GPU(T4)** 에서 만드는 도구.
※ 연구·교육용. Kaggle 실제 GPU 실행은 아직 검증 전 — 첫 실행 로그의 `[PATCH]`·`[GEN]`·`[TIMING]` 줄로 확인한다.

| 파일 | 하는 일 |
|---|---|
| `tools/kaggle_music_yue2.py` (v1.1) | 설치(처음 한 번) → GPU 점검 → 생성 → `audio.flac`·`audio.mp3`·악보·기록 |
| `tools/tests/test_kaggle_music_yue2.py` | GPU 없이 도는 시험 15개: `python3 -m unittest tools.tests.test_kaggle_music_yue2 -v` |

## Kaggle 노트북

1. kaggle.com → Create → **New Notebook**
2. 오른쪽 **Settings** → **Accelerator: GPU T4 x2** · **Internet: On** (P100 은 torch 2.10 이 지원하지 않아 안 됨)
3. 아래 셀을 넣고 실행

```python
!test -d /kaggle/working/music-ytb || git clone -q --depth 1 https://github.com/yeomin1024/music-ytb.git /kaggle/working/music-ytb
!cd /kaggle/working/music-ytb && git config core.fileMode false && git fetch -q origin main && git reset -q --hard origin/main

# 연주곡(영상 배경음) 2곡 — T4 두 장이 한 곡씩 동시에
!python /kaggle/working/music-ytb/tools/kaggle_music_yue2.py run --mode instrumental \
    --style "calm lo-fi piano, soft drums, warm bass, 80 BPM" --seeds 1,2

from pathlib import Path
from IPython.display import Audio, display
for p in sorted(Path("/kaggle/working/music").glob("*/seed-*/audio.mp3")):
    print(p); display(Audio(str(p)))
```

- 세션마다 처음 한 번 설치(몇 분) + 모델 약 7.8GB 받기. 같은 세션에서 다시 돌리면 건너뛴다.
- 결과: `/kaggle/working/music/<이름>/seed-<시드>/` — `audio.flac`(48kHz 스테레오), `audio.mp3`, `score.abc`(악보), `kaggle_run.json`(설정·시간·메모리·패치 기록). 로그 전체는 `/kaggle/working/music/music.log`.
- 오른쪽 **Output** 에서 내려받거나, 노트북을 **Save Version** 하면 남는다.

### 가사가 있는 노래

```python
%%writefile /kaggle/working/lyrics.txt
[Verse]
(가사)

[Chorus]
(가사)
```

```python
!python /kaggle/working/music-ytb/tools/kaggle_music_yue2.py run --mode song \
    --style "Korean, ballad, warm male vocal, piano, 72 BPM" --lyrics-file /kaggle/working/lyrics.txt --seeds 1,2
```

한국어 가사를 얼마나 잘 부르는지는 공식 문서에 없다 (들어 보고 판단).

## 자주 바꾸는 옵션

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--mode` | `instrumental` | `instrumental` = YuE2 악보 → 노래 선율을 악기로 → 연주곡 · `song` = 가사 + 스타일 → 노래 |
| `--style` | — | 영어 권장. 장르·악기·분위기·BPM |
| `--sections` | `Intro,Verse,Chorus,Outro` | 연주곡 길이를 대략 정함 (예: `Intro,Verse,Chorus,Verse,Chorus,Outro`). 초 단위 지정은 모델이 보장하지 않음 |
| `--seeds` | `1` | 쉼표로 여러 개 → GPU 마다 나눠 동시에. 같은 시드·설정이면 같은 곡 |
| `--name` | `bgm-날짜시각` | 결과 폴더 이름 (영문·숫자·`_.-`) |
| `--chunk` | `256` | T4 메모리 절약 묶음. 메모리 부족이면 `128` |
| `--backend` | `torch` | 문제가 생기면 `torch-eager` (느림) |
| `--gpus` | `all` | `0` 이면 한 장만 |

## 문제가 생기면

| 로그 | 할 일 |
|---|---|
| `[ENV] … GPU 없음` · `torch 2.10 이 지원하지 않음` | Accelerator 를 **GPU T4 x2** 로 |
| `[CHECK] … ok=False` | 이 GPU 에서 공식 BF16 경로가 안 됨 → 대안: [yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp)(GGUF, 비공식) |
| `[LOAD] … 401/403/gated` | Hugging Face 토큰이 필요한 모델 → Add-ons → Secrets 에 `HF_TOKEN` 추가, 실행 줄 위에 `import os; from kaggle_secrets import UserSecretsClient; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")` |
| `[GEN] … GPU 메모리 부족` | `--chunk 128` → 가사·섹션 줄이기 → `--backend torch-eager` 순서로 |
| `[CHECK] … 거의 무음` · `잘림` | 다른 시드로, 또는 섹션 줄이기 |

저장소를 비공개로 바꾸면 클론에 토큰이 필요하다 — `yeomin1024-ytb` 의 Kaggle 셀처럼 `GITHUB_TOKEN` 시크릿(이 저장소 접근 권한 포함)을 쓴다.

## T4 에서 고친 것

공식 권장 사양은 BF16 지원 GPU 24GB. T4(16GB, 연산 능력 7.5)에서는 공식 코드가 그대로는 두 군데에서 멈추거나 메모리가 넘친다. 연산 능력 8.0 미만일 때만 고친다 (`apply_t4_patches`, 로그 `[PATCH]`). 소리를 바꾸는 설정(가중치·샘플링·32 단계·VAE)은 그대로.

1. AR 단계가 어텐션을 FlashAttention 으로 고른다(함수가 있는지만 봄). FlashAttention 은 Ampere 이상 전용 → `sdpa`(PyTorch 가 GPU 에 맞게 고름)
2. T4 는 BF16 빠른 어텐션이 없어 fp32 수학 계산으로 가는데, 곡 전체 질의를 한 번에 계산해 메모리가 넘침 → 공식 CPU 방식처럼 256줄씩

VRAM 20GiB 미만이면 공식 옵션 `offload_ar` 를 켠다 (메모리만 아끼고 결과는 같음). T4 는 권장 사양 밖이라 느리다 — 곡당 수십 분 걸릴 수 있다 (추정, `[TIMING]` 로그로 확인).

## 고정한 것 (바꾸면 VERSION·CHANGELOG)

- 설치: uv 0.12.19, 파이썬 3.12, `yue2-infer` 0.1.6 휠 sha256 `129a0153…`, torch 2.10.0 (CUDA 12.8)
- 모델: `m-a-p/YuE2-3B@14fc6c6f…`, `m-a-p/YuE2-Vae@9a94e1d0…` (공식 연주곡 스킬이 고정한 리비전)
- 연주곡 변환: 공식 `skills/yue2-music/instrumental/scripts` 5개 파일, 커밋 `1647252d…`·파일마다 sha256

## 라이선스

YuE2 코드 Apache-2.0, 가중치 CC BY-NC 4.0 + 창작자 추가 허가 — 개인 창작자는 만든 곡으로 수익을 내도 된다(로열티 없음), 회사의 상업 이용은 별도 계약. 자세히: [YuE2 README 'License'](https://github.com/multimodal-art-projection/YuE#license).
