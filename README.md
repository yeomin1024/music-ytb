# music-ytb

오픈소스 작곡 AI **YuE2**(HKUST·M-A-P, 2026-09 공개, `m-a-p/YuE2-3B`)로 노래·연주곡을 **Kaggle 무료 GPU(T4)** 에서 만드는 도구.
좋아하는 곡 하나 + 그 곡을 듣고 느낀 **멘트** + 떠오르는 **사진**으로 새 곡을 만든다: 곡을 **트랙별로 나눠 분석**하고(멜로디 특징·소리 질감), 그 숫자와 멘트·사진을 **Claude 대화**에서 프롬프트·가사로 바꾼 뒤, YuE2 가 **멜로디가 다른 새 곡**을 여러 개 만든다. 원곡과 멜로디를 비교해 **통과한 곡만 이 저장소 `results/`** 에 올린다.
※ 연구·교육용. Kaggle 실제 GPU 실행은 아직 검증 전 — 첫 실행 로그의 `[PATCH]`·`[ANALYZE]`·`[GEN]`·`[SIMILARITY]`·`[PUSH]` 줄로 확인한다.

| 파일 | 하는 일 |
|---|---|
| `tools/kaggle_music_panel.py` (v1.1) | 노트북 버튼 세 단계: **① 분석 + Claude 에게 보낼 글** → **② Claude 답 읽기** → **③ 곡 만들기** |
| `tools/kaggle_music_yue2.py` (v1.4) | 설치(처음 한 번) · `analyze`(트랙 분리·분석 → Claude 에게 보낼 글) · `run`(생성 → 멜로디 검사 → 저장소에) — 버튼이 이것을 부름 |
| `tools/music_analysis.py` (v1.0) | 트랙별 수치·멜로디 특징·곡 구성·Claude 에게 보낼 글·Claude 답 읽기 (순수 계산) |
| `tools/tests/` | GPU·네트워크 없이 도는 시험 66개: `python3 -m unittest tools.tests.test_kaggle_music_yue2 tools.tests.test_kaggle_music_panel tools.tests.test_music_analysis -v` |
| `results/<이름>/` | 올라간 곡 — `seed-N/audio.mp3`(320k)·`score.abc`(AI 악보)·`kaggle_run.json`·`similarity.json`, `analysis.json`·`prompt.txt`·`README.md` |

## 처음 한 번 준비

1. **Kaggle 노트북**: kaggle.com → Create → New Notebook → 오른쪽 **Settings** → **Accelerator: GPU T4 x2** · **Internet: On** (P100 은 torch 2.10 이 지원하지 않아 안 됨)
2. **GitHub 토큰** (결과를 이 저장소에 올리는 열쇠 — 채팅·셀·파일에 붙여 넣지 않는다)
   - GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate new token
   - Repository access: **Only select repositories** → `yeomin1024/music-ytb` (yeomin1024-ytb 용 토큰이 있으면 그 토큰을 Edit 해서 music-ytb 를 더해도 된다)
   - Permissions → Repository permissions → **Contents: Read and write**
   - Kaggle 노트북 위 메뉴 **Add-ons → Secrets → Add Secret** → Label `GITHUB_TOKEN`, Value 토큰 → 이 노트북에 연결(체크)
> ⚠️ 참고 곡 파일은 **이 저장소(공개)에 올리지 않는다**. 도구도 원곡 파일·원곡을 받아 적은 악보·파일 이름은 올리지 않는다 (허용 목록 방식, 시험으로 확인). 버튼으로 올린 원곡·멘트와 바꾼 wav 는 `/tmp/yue2/` 에만 두어 Kaggle 출력(`/kaggle/working`)에도 남지 않는다. 트랙 분리한 원곡 트랙은 메모리에서 수치만 재고 버린다 (파일로 남기지 않고, **결과 곡에 섞지 않는다** — 섞으면 샘플링).

## 노트북 셀 (이것 하나만 복사해서 실행)

```python
import os, sys, importlib
from kaggle_secrets import UserSecretsClient
os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")   # 결과 올리기용 (값은 출력하지 않음)
!test -d /kaggle/working/music-ytb || git clone -q --depth 1 https://github.com/yeomin1024/music-ytb.git /kaggle/working/music-ytb
!cd /kaggle/working/music-ytb && git config core.fileMode false && git fetch -q origin main && git reset -q --hard origin/main
sys.path.insert(0, "/kaggle/working/music-ytb/tools")
import kaggle_music_panel; importlib.reload(kaggle_music_panel).panel()
```

셀 아래에 세 단계가 나온다:

**① 분석**
- **[곡 올리기]** → 곡 파일 하나 (**mp3 권장** — 큰 wav 는 노트북 업로드 한도에 걸릴 수 있다)
- **멘트** 칸 — 이 곡을 듣고 느낀 점·떠오르는 장면
- **가사 넣기** 체크 — 켜면 Claude 에게 가사도 부탁하고 노래로, 끄면 연주곡
- **이름**(비우면 `ref-날짜시각`) → **[① 분석하고 Claude 에게 보낼 글 만들기]** (설치·분석에 몇 분)
- 끝나면 **'Claude 에게'** 칸에 글이 나온다 → 칸을 클릭하고 **전체 선택(Ctrl+A) → 복사** → **Claude 대화에 붙여 넣고, 떠오르는 사진을 첨부해서** 보낸다

**② Claude 답**
- Claude 답(코드 블록 포함 그대로)을 **'Claude 답'** 칸에 붙여 넣기 → **[② 답 읽기]**
- STYLE·SECTIONS·가사 칸이 채워진다 → **가사를 읽고 고친다** (그대로 둬도 됨)
- 형식이 틀리면 칸을 바꾸지 않고 무엇이 틀렸는지 알려 준다 → Claude 에게 "형식 그대로 다시 써 줘"
- Claude 를 건너뛰고 싶으면 ①이 STYLE 칸에 넣어 둔 **자동 프롬프트**로 바로 ③

**③ 곡 만들기**
- **곡 수**(기본 4) · **통과한 곡 저장소에 올리기**(기본 켬) → **[③ 곡 만들기]** (곡당 수십 분 — 추정)
- 끝나면 곡마다 길이·멜로디 검사 결과·플레이어

- 로그는 아래 칸과 `/kaggle/working/music/panel.log` 에 함께. 실행 중에는 버튼이 잠긴다. 멈추려면 Kaggle 의 셀 멈춤(■) — GPU 작업 프로세스까지 함께 멈춘다.
- 같은 이름으로 다시 하면 같은 곡은 분석을 건너뛴다. **다른 곡인데 이름이 같으면 멈춘다** → 이름을 바꾼다.

## Claude 에게 보낼 글 (① 이 만들어 줌)

도구가 아래 모양의 글을 숫자까지 채워서 만든다 — 그대로 복사해 Claude 에 붙이고 **사진을 첨부**하면 된다 (직접 쓸 필요 없음).

```
아래 [원곡 분석]은 내가 좋아하는 곡을 도구로 트랙별로 나눠 잰 숫자야 (너는 소리를 못 들으니 숫자로 설명할게).
이 분석에서 느껴지는 분위기·악기 구성·소리 질감에, [내 멘트]와 첨부한 사진에서 떠오르는 감정과 장면을 합쳐서
오픈소스 작곡 AI(YuE2)로 만들 **새 곡**의 설정을 써 줘.

지킬 것
1. 원곡을 따라 하지 않기 — 원곡 제목·가수·가사·멜로디를 쓰지 말고, 분위기·악기 구성·소리 질감만 참고.
2. STYLE — 영어 한 줄, 쉼표로 구분, 400자 이하. 순서: 장르, 분위기(멘트·사진 반영), 악기와 소리 질감, 장조/단조, 템포(숫자 BPM). …
3. SECTIONS — 곡 구성. Intro, Verse, Chorus, Bridge, Outro 중에서 골라 쉼표로 6~9개. 참고 — 원곡 구성: …
4. LYRICS — (가사 넣기) 내 멘트와 사진에서 떠오르는 새 가사, 한국어. 섹션 태그 줄 … / (연주곡) "없음"
5. 답은 아래 형식 그대로 코드 블록 하나에. …
(STYLE: / SECTIONS: / LYRICS: 형식 예시)

[내 멘트]
(멘트 칸에 쓴 글)

[사진]
(이 메시지에 사진을 첨부했어 — 색감·장소·시간대·감정을 반영해 줘)

[원곡 분석 — 숫자는 도구가 잰 근사치]
- 길이·템포·조성, 장르·분위기·악기 태그(확률)
- 트랙별(보컬·드럼·베이스·기타·피아노·그 밖): 비중 % · 밝기 Hz · 스테레오 폭 · 다이내믹 dB · 잔향·지속 · 거친 정도 (숫자 + 말)
- 멜로디 특징 (음표는 알려 주지 않음): 음역 · 순차/도약 비율 · 리듬 밀도
- 원곡 구성
```

## 분석 내용과 한계

- **트랙 분리**: Demucs `htdemucs_6s`(MIT) — 보컬·드럼·베이스·기타·피아노·그 밖. 겹치는 소리는 완벽히 나뉘지 않는다.
- **트랙별 수치** (`music_analysis.stem_features`, 모두 근사치): 비중(그 트랙 에너지 비율) · 밝기(스펙트럼 무게중심 Hz) · 스테레오 폭(옆/전체 에너지) · 다이내믹(최고/평균 dB — 작을수록 압축) · 잔향·지속(소리 시작 0.2~0.4초 뒤 에너지 비 — 소리가 촘촘하면 '잴 수 없음') · 거친 정도(스펙트럼 평탄도 — 소음 성분, **디스토션 판별은 아님**). 말로 바꾸는 기준은 합성 신호로 방향만 확인한 보정 전 값이라 숫자를 함께 넘긴다. "리버브 몇 초·어떤 플러그인" 같은 정확한 이펙트 설정은 알 수 없다.
- **멜로디 특징**: SheetSage2 로 받아 적은 원곡 멜로디에서 음역·순차/도약·같은 음 반복·리듬 밀도만 — **음표는 Claude 에게도 저장소에도 넘기지 않는다** (원곡 멜로디를 변형하면 파생 멜로디라 표절 위험).
- **곡 구성**: 받아 적은 악보의 구간 이름 (구성 자체는 보호 대상이 아님).
- **장·단조**는 SheetSage2 가 될 때만 (librosa 추정은 합성 시험에서 A단조를 A장조로 읽어 기록만).
- 한국어 가사를 YuE2 가 얼마나 잘 부르는지는 공식 문서에 없다 — 들어 보고 판단.

## 명령으로 (버튼 대신)

```
# ① 분석 + Claude 에게 보낼 글 → /kaggle/working/music/my-01/claude_request.txt
!python /kaggle/working/music-ytb/tools/kaggle_music_yue2.py analyze --ref <곡 파일> --name my-01 --comment-file comment.txt [--lyrics]
# ③ 곡 만들기 (Claude 답의 STYLE·SECTIONS, 가사는 파일로)
!python /kaggle/working/music-ytb/tools/kaggle_music_yue2.py run --ref <곡 파일> --name my-01 --style "<STYLE>" \
    --sections "<SECTIONS>" [--mode song --lyrics-file lyrics.txt] --count 4 --push
# 멘트·Claude 없이 자동 프롬프트로 한 번에
!python /kaggle/working/music-ytb/tools/kaggle_music_yue2.py run --ref <곡 파일> --count 4 --name jpop-like-01 --style-extra "J-pop band" --push
```

`--ref auto` = `+ Add Input → Upload` 로 올린 `/kaggle/input` 아래 오디오 파일 하나.

순서 (로그 단계):
1. `[SETUP]`·`[CHECK]` — 세션마다 처음 한 번 설치(몇 분)·GPU 점검. 같은 세션에서 다시 돌리면 건너뜀
2. `[PUSH] check=ok` — 생성 전에 토큰·쓰기 권한부터 확인 (안 되면 여기서 멈춤)
3. `[ANALYZE]` — 참고 곡 분석. 원곡 소리는 YuE2 에 넣지 않고 **글(프롬프트)로만** 넘긴다 → 멜로디는 모델이 새로 씀
   - librosa: 길이·템포·음량·밝기 (조성 추정은 기록만)
   - AST(AudioSet 태그): 장르·분위기·악기와 그 확률 — `[ANALYZE] step=tags … evidence=` 줄
   - SheetSage2(YuE2 팀의 받아 적기 모델): 원곡 멜로디 악보(검사용, 로컬에만) + 템포·장/단조 → `step=melody`
   - Demucs: 트랙별 수치 → `step=stems stem=… words=…` 줄
4. `[CLAUDE]`(analyze) 또는 `[PROMPT]`(run) — Claude 에게 보낼 글 / 쓰인 프롬프트. 곡·가수 이름은 넣지 않음
5. `[GEN]` — 시드 1~4, T4 두 장이 두 곡씩 (곡당 수십 분 걸릴 수 있음 — 추정)
6. `[SIMILARITY]` — 곡마다 '통과' / '검토 필요' / '멜로디 검사 못 함'
7. `[PUSH] status=ok` — '통과' 곡만 `results/<이름>/` 에 커밋·푸시

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
| `[PANEL] 먼저 [곡 올리기]로…` (올렸는데도) | 파일이 커서 업로드가 실패한 것 → mp3 로 다시 |
| `[INPUT] … 다른 참고 곡으로 이미 분석함` | 이름을 바꾸거나 명령에 `--reanalyze` |
| `[INPUT] … 오디오 파일이 없음/여러 개` (명령 `--ref auto`) | + Add Input → Upload 로 곡 하나만, 또는 `--ref /kaggle/input/<데이터셋>/<파일>` |
| `[PUSH] … GITHUB_TOKEN 없음` · `쓰기 권한 확인 실패` | 준비 2번 (토큰 저장소 범위에 music-ytb, Contents: Read and write, 노트북에 Secret 연결) |
| `[CHECK] … ok=False` | 이 GPU 에서 공식 BF16 경로가 안 됨 → 대안: [yue2.cpp](https://github.com/ServeurpersoCom/yue2.cpp)(GGUF, 비공식) |
| `[LOAD] … 401/403/gated` | Hugging Face 토큰이 필요한 모델 → Secrets 에 `HF_TOKEN` 추가, 셀에서 `os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")` |
| `[ANALYZE] step=tags error=…` | 장르·악기 태그 없이 진행됨 → `--style-extra` 로 장르·악기를 직접 |
| `[ANALYZE] step=stems error=…` | 트랙 분리 없이 진행됨 — Claude 에게 보낼 글에는 태그·멜로디 특징만 |
| `[PANEL] 답 오류: …` | Claude 에게 "형식 그대로(STYLE/SECTIONS/LYRICS 코드 블록) 다시 써 줘" 또는 칸을 직접 고치기 |
| `[ANALYZE] step=sheetsage2 ok=False` | 멜로디 검사 못 함 → `--reanalyze` 로 다시, 계속 안 되면 직접 들어 보고 `--push-unchecked` |
| `[GEN] … GPU 메모리 부족` | `--chunk 128` → 가사·섹션 줄이기 → `--backend torch-eager` 순서로 |
| `[PUSH] … 올리기 실패` | 결과는 `/kaggle/working/music/<이름>/` 에 있음 → `push --name <이름>` |

## T4 에서 고친 것

공식 권장 사양은 BF16 지원 GPU 24GB. T4(16GB, 연산 능력 7.5)에서는 공식 코드가 그대로는 두 군데에서 멈추거나 메모리가 넘친다. 연산 능력 8.0 미만일 때만 고친다 (`apply_t4_patches`, 로그 `[PATCH]`). 소리를 바꾸는 설정(가중치·샘플링·32 단계·VAE)은 그대로.

1. AR 단계가 어텐션을 FlashAttention 으로 고른다(함수가 있는지만 봄). FlashAttention 은 Ampere 이상 전용 → `sdpa`(PyTorch 가 GPU 에 맞게 고름)
2. T4 는 BF16 빠른 어텐션이 없어 fp32 수학 계산으로 가는데, 곡 전체 질의를 한 번에 계산해 메모리가 넘침 → 공식 CPU 방식처럼 256줄씩

VRAM 20GiB 미만이면 공식 옵션 `offload_ar` 를 켠다 (메모리만 아끼고 결과는 같음). SheetSage2 는 T4 에서 fp32.

## 고정한 것 (바꾸면 VERSION·CHANGELOG)

- 설치: uv 0.12.19, 파이썬 3.12, `yue2-infer` 0.1.6 휠 sha256 `129a0153…`, torch 2.10.0 (CUDA 12.8), librosa 0.11.0, demucs 4.1.0(트랙 분리 `htdemucs_6s`), SheetSage2 의존 mir_eval 0.8.2·pretty_midi 0.2.10·mido 1.3.3·scipy 1.15.3 (YuE2-Turbo 와 같은 고정). 실제 설치 목록은 `/tmp/yue2/freeze.txt`
- 모델: `m-a-p/YuE2-3B@14fc6c6f…`, `m-a-p/YuE2-Vae@9a94e1d0…`, SheetSage2·MERT2 는 공식 `cover-release.json` 리비전, 태그 `MIT/ast-finetuned-audioset-10-10-0.4593`(리비전은 `analysis.json` 에 기록)
- 공식 스킬 코드 7개(연주곡 변환·받아 적기·리비전 표): 커밋 `1647252d…`·파일마다 sha256

## 라이선스

YuE2 코드 Apache-2.0, 가중치 CC BY-NC 4.0 + 창작자 추가 허가 — 개인 창작자는 만든 곡으로 수익을 내도 된다(로열티 없음), 회사의 상업 이용은 별도 계약. 자세히: [YuE2 README 'License'](https://github.com/multimodal-art-projection/YuE#license). 참고 곡의 저작권은 원작자에게 있다 — 원곡 파일은 공개 저장소·공개 노트북에 올리지 않는다.
