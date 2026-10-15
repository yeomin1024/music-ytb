# music-ytb

오픈소스 작곡 AI **YuE2**(HKUST·M-A-P, 2026-09 공개, `m-a-p/YuE2-3B`)로 노래·연주곡을 **Kaggle 무료 GPU(T4)** 에서 만드는 도구.
좋아하는 곡 하나를 올리면 **분위기가 비슷하고 멜로디는 다른 곡 여러 개**를 만들고, 원곡과 멜로디를 비교해 **통과한 곡만 이 저장소 `results/`** 에 올린다.
※ 연구·교육용. Kaggle 실제 GPU 실행은 아직 검증 전 — 첫 실행 로그의 `[PATCH]`·`[ANALYZE]`·`[GEN]`·`[SIMILARITY]`·`[PUSH]` 줄로 확인한다.

| 파일 | 하는 일 |
|---|---|
| `tools/kaggle_music_yue2.py` (v1.2) | 설치(처음 한 번) → [참고 곡 분석] → 생성 → 멜로디 검사 → [저장소에 올리기] |
| `tools/tests/test_kaggle_music_yue2.py` | GPU·네트워크 없이 도는 시험 38개: `python3 -m unittest tools.tests.test_kaggle_music_yue2 -v` |
| `results/<이름>/` | 올라간 곡 — `seed-N/audio.mp3`(320k)·`score.abc`(AI 악보)·`kaggle_run.json`·`similarity.json`, `analysis.json`·`prompt.txt`·`README.md` |

## 처음 한 번 준비

1. **Kaggle 노트북**: kaggle.com → Create → New Notebook → 오른쪽 **Settings** → **Accelerator: GPU T4 x2** · **Internet: On** (P100 은 torch 2.10 이 지원하지 않아 안 됨)
2. **GitHub 토큰** (결과를 이 저장소에 올리는 열쇠 — 채팅·셀·파일에 붙여 넣지 않는다)
   - GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate new token
   - Repository access: **Only select repositories** → `yeomin1024/music-ytb` (yeomin1024-ytb 용 토큰이 있으면 그 토큰을 Edit 해서 music-ytb 를 더해도 된다)
   - Permissions → Repository permissions → **Contents: Read and write**
   - Kaggle 노트북 위 메뉴 **Add-ons → Secrets → Add Secret** → Label `GITHUB_TOKEN`, Value 토큰 → 이 노트북에 연결(체크)
3. **참고 곡 올리기**: 오른쪽 **+ Add Input → Upload** → 곡 파일(mp3·wav·flac·m4a) 하나 → 데이터셋은 **Private** 그대로. 다른 곡으로 바꿀 때는 그 입력을 빼고 새로 올린다 (`--ref auto` 는 오디오 파일이 하나일 때만)

> ⚠️ 참고 곡 파일은 **이 저장소(공개)에 올리지 않는다**. 도구도 원곡 파일·원곡을 받아 적은 악보·파일 이름은 올리지 않는다 (허용 목록 방식, 시험으로 확인).

## Kaggle 셀 — 참고 곡 하나 → 비슷한 분위기 4곡 → 검사 → 저장소

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")   # 결과 올리기용 (값은 출력하지 않음)
!test -d /kaggle/working/music-ytb || git clone -q --depth 1 https://github.com/yeomin1024/music-ytb.git /kaggle/working/music-ytb
!cd /kaggle/working/music-ytb && git config core.fileMode false && git fetch -q origin main && git reset -q --hard origin/main

!python /kaggle/working/music-ytb/tools/kaggle_music_yue2.py run --ref auto --count 4 --name jpop-like-01 \
    --style-extra "J-pop band" --push

from pathlib import Path
from IPython.display import Audio, display
for p in sorted(Path("/kaggle/working/music/jpop-like-01").glob("seed-*/audio.mp3")):
    print(p, Path(p.parent, "similarity.json").read_text(encoding="utf-8")[:120]); display(Audio(str(p)))
```

순서 (로그 단계):
1. `[SETUP]`·`[CHECK]` — 세션마다 처음 한 번 설치(몇 분)·GPU 점검. 같은 세션에서 다시 돌리면 건너뜀
2. `[PUSH] check=ok` — 생성 전에 토큰·쓰기 권한부터 확인 (안 되면 여기서 멈춤)
3. `[ANALYZE]` — 참고 곡 분석. 원곡 소리는 YuE2 에 넣지 않고 **글(프롬프트)로만** 넘긴다 → 멜로디는 모델이 새로 씀
   - librosa: 길이·템포·음량·밝기 (조성 추정은 기록만)
   - AST(AudioSet 태그): 장르·분위기·악기와 그 확률 — `[ANALYZE] step=tags … evidence=` 줄
   - SheetSage2(YuE2 팀의 받아 적기 모델): 원곡 멜로디 악보(검사용, 로컬에만) + 템포·장/단조
4. `[PROMPT]` — 예: `J-pop band, pop, energetic and exciting, electric guitar, electric bass, live drums, strings, minor key, 152 BPM`. 곡·가수 이름은 넣지 않음
5. `[GEN]` — 시드 1~4, T4 두 장이 두 곡씩 (곡당 수십 분 걸릴 수 있음 — 추정)
6. `[SIMILARITY]` — 곡마다 '통과' / '검토 필요' / '멜로디 검사 못 함'
7. `[PUSH] status=ok` — '통과' 곡만 `results/<이름>/` 에 커밋·푸시

프롬프트가 마음에 안 들면 `--style "..."` 로 직접 주면 된다 (분석은 검사에만 쓰임). 같은 `--name` 으로 다시 돌리면 분석은 건너뛰고(`--reanalyze` 로 다시) 곡만 더 만든다 (`seed-1-2` …).

## 멜로디 검사 (표절 방지)

- 비교: 만든 곡의 악보(YuE2 가 쓴 계획) ↔ 원곡을 SheetSage2 로 받아 적은 멜로디. 같은 시각에는 가장 높은 음(주선율 근사).
- 조옮김·템포(반·두 배 박자 표기 포함)와 상관없게 **음정 변화**와 **리듬 비(앞뒤 음 간격의 비)** 로 바꿔, 연속으로 같은 가장 긴 구간을 잰다. 같은 음 반복만인 구간은 세지 않는다(움직이는 음정 4개 이상).
- **'검토 필요'** = 리듬까지 같은 음정 변화가 연속 **8개 이상**(음 약 10개) 또는 음정만 같은 것이 연속 **12개 이상**. 위치(원곡 몇 초 ↔ 만든 곡 몇 초)를 `similarity.json` 에 적는다.
- 기준을 정한 실험 (2026-10-15, 시험 코드에 일부 포함): 우연히 겹치기 쉬운 단조로운 무작위 팝 멜로디(8분음표 위주·순차 진행 위주·구간 반복) 1,080쌍 — 리듬까지 같은 구간 99.9%가 5 이하(최대 8), 음정만 99.9%가 9 이하 → 이 기준으로 잘못 걸리는 비율 0~0.2%. 같은 멜로디·조옮김+두 배 박자 표기 사본·다른 곡에 끼운 10음 구간·리듬만 바꾼 사본은 모두 '검토 필요'.
- 한계: **경험칙이며 법적 판단이 아니다.** 10음보다 짧은 동기(훅)는 걸리지 않을 수 있고, SheetSage2 받아 적기 오차로 놓칠 수 있다. 원곡을 모델에 넣지 않으므로 위험은 낮지만 0 은 아니다 — 올리기 전에 직접 들어 보는 것을 권한다.
- 같은 프롬프트로 만든 곡끼리 겹친 최대값도 함께 적는다 (우연히 겹치는 정도 — 판정에는 안 씀).

| 판정 | 기본으로 올림? | 바꾸기 |
|---|---|---|
| 통과 | 올림 | — |
| 검토 필요 | 안 올림 | `--push-flagged` |
| 멜로디 검사 못 함 (SheetSage2 실패) | 안 올림 | `--push-unchecked`, 다시 받아 적기는 `--reanalyze` |
| 참고 곡 없음 (프롬프트만) | 올림 | — |

## 프롬프트만으로 (참고 곡 없이)

```python
!python /kaggle/working/music-ytb/tools/kaggle_music_yue2.py run --mode instrumental --name jpop-band-01 \
    --style "J-pop band, bright and energetic, catchy lead melody on piano, electric bass, two electric guitars panned hard left and right, live drums, lush strings, major key, 160 BPM" \
    --seeds 1,2 --push
```

가사가 있는 노래: `%%writefile /kaggle/working/lyrics.txt` 로 `[Verse]`·`[Chorus]` 태그와 가사를 쓰고 `--mode song --lyrics-file /kaggle/working/lyrics.txt`.

## 자주 바꾸는 옵션

| 옵션 | 기본 | 뜻 |
|---|---|---|
| `--ref` | — | 참고 곡 파일 경로, 또는 `auto`(= `/kaggle/input` 아래 오디오 파일 하나) |
| `--count` | 참고 곡이면 4, 아니면 1 | 곡 수 (시드 1..N). `--seeds 5,9` 로 직접 줄 수도 |
| `--style-extra` | — | 분석 프롬프트 앞에 더할 말 (AudioSet 에 없는 장르 — 예: `J-pop band`) |
| `--style` | — | 프롬프트를 직접 (참고 곡은 검사에만) |
| `--push` | 끔 | '통과' 곡을 `results/<이름>/` 에 (GITHUB_TOKEN 필요) |
| `--push-flac` | 끔 | flac 도 올림 (곡당 약 30MB) |
| `--mode` | `instrumental` | `song` = 가사 파일로 노래 |
| `--sections` | `Intro,Verse,Chorus,Outro` | 연주곡 길이를 대략 정함 (예: `Intro,Verse,Chorus,Verse,Chorus,Bridge,Chorus,Outro`) |
| `--chunk` | `256` | T4 메모리 절약 묶음. 메모리 부족이면 `128` |
| `--backend` | `torch` | 문제가 생기면 `torch-eager` (느림) |
| `--gpus` | `all` | `0` 이면 한 장만 |

다른 명령: `similarity --name 이름` (검사만 다시) · `push --name 이름` (올리기가 실패했을 때 다시).

## 문제가 생기면

| 로그 | 할 일 |
|---|---|
| `[ENV] … GPU 없음` · `torch 2.10 이 지원하지 않음` | Accelerator 를 **GPU T4 x2** 로 |
| `[INPUT] … 오디오 파일이 없음/여러 개` | + Add Input → Upload 로 곡 하나만, 또는 `--ref /kaggle/input/<데이터셋>/<파일>` |
| `[PUSH] … GITHUB_TOKEN 없음` · `쓰기 권한 확인 실패` | 준비 2번 (토큰 저장소 범위에 music-ytb, Contents: Read and write, 노트북에 Secret 연결) |
| `[CHECK] … ok=False` | 이 GPU 에서 공식 BF16 경로가 안 됨 → 대안: [yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp)(GGUF, 비공식) |
| `[LOAD] … 401/403/gated` | Hugging Face 토큰이 필요한 모델 → Secrets 에 `HF_TOKEN` 추가, 셀에서 `os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")` |
| `[ANALYZE] step=tags error=…` | 장르·악기 태그 없이 진행됨 → `--style-extra` 로 장르·악기를 직접 |
| `[ANALYZE] step=sheetsage2 ok=False` | 멜로디 검사 못 함 → `--reanalyze` 로 다시, 계속 안 되면 직접 들어 보고 `--push-unchecked` |
| `[GEN] … GPU 메모리 부족` | `--chunk 128` → 가사·섹션 줄이기 → `--backend torch-eager` 순서로 |
| `[PUSH] … 올리기 실패` | 결과는 `/kaggle/working/music/<이름>/` 에 있음 → `push --name <이름>` |

## T4 에서 고친 것

공식 권장 사양은 BF16 지원 GPU 24GB. T4(16GB, 연산 능력 7.5)에서는 공식 코드가 그대로는 두 군데에서 멈추거나 메모리가 넘친다. 연산 능력 8.0 미만일 때만 고친다 (`apply_t4_patches`, 로그 `[PATCH]`). 소리를 바꾸는 설정(가중치·샘플링·32 단계·VAE)은 그대로.

1. AR 단계가 어텐션을 FlashAttention 으로 고른다(함수가 있는지만 봄). FlashAttention 은 Ampere 이상 전용 → `sdpa`(PyTorch 가 GPU 에 맞게 고름)
2. T4 는 BF16 빠른 어텐션이 없어 fp32 수학 계산으로 가는데, 곡 전체 질의를 한 번에 계산해 메모리가 넘침 → 공식 CPU 방식처럼 256줄씩

VRAM 20GiB 미만이면 공식 옵션 `offload_ar` 를 켠다 (메모리만 아끼고 결과는 같음). SheetSage2 는 T4 에서 fp32.

## 고정한 것 (바꾸면 VERSION·CHANGELOG)

- 설치: uv 0.12.19, 파이썬 3.12, `yue2-infer` 0.1.6 휠 sha256 `129a0153…`, torch 2.10.0 (CUDA 12.8), librosa 0.11.0, SheetSage2 의존 mir_eval 0.8.2·pretty_midi 0.2.10·mido 1.3.3·scipy 1.15.3 (YuE2-Turbo 와 같은 고정). 실제 설치 목록은 `/tmp/yue2/freeze.txt`
- 모델: `m-a-p/YuE2-3B@14fc6c6f…`, `m-a-p/YuE2-Vae@9a94e1d0…`, SheetSage2·MERT2 는 공식 `cover-release.json` 리비전, 태그 `MIT/ast-finetuned-audioset-10-10-0.4593`(리비전은 `analysis.json` 에 기록)
- 공식 스킬 코드 7개(연주곡 변환·받아 적기·리비전 표): 커밋 `1647252d…`·파일마다 sha256

## 라이선스

YuE2 코드 Apache-2.0, 가중치 CC BY-NC 4.0 + 창작자 추가 허가 — 개인 창작자는 만든 곡으로 수익을 내도 된다(로열티 없음), 회사의 상업 이용은 별도 계약. 자세히: [YuE2 README 'License'](https://github.com/multimodal-art-projection/YuE#license). 참고 곡의 저작권은 원작자에게 있다 — 원곡 파일은 공개 저장소·공개 노트북에 올리지 않는다.
