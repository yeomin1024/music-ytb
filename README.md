# music-ytb

오픈소스 작곡 AI **YuE2**(HKUST·M-A-P, 2026-09 공개, `m-a-p/YuE2-3B`)로 노래·연주곡을 **Kaggle 무료 GPU(T4)** 에서 만드는 도구.
좋아하는 곡 하나 + 그 곡을 듣고 느낀 **멘트** + 떠오르는 **사진**으로 새 곡을 만든다: 곡을 **트랙별로 나눠 분석**하고(멜로디 특징·소리 질감), 그 숫자와 멘트·사진을 **Claude 대화**에서 프롬프트·가사로 바꾼 뒤, YuE2 가 **멜로디가 다른 새 곡**을 여러 개 만든다. 원곡과 멜로디를 비교해 **통과한 곡만 이 저장소 `results/`** 에 올린다.
※ 연구·교육용. Kaggle 실제 GPU 실행은 아직 검증 전 — 첫 실행 로그의 `[PATCH]`·`[ANALYZE]`·`[GEN]`·`[SIMILARITY]`·`[PUSH]` 줄로 확인한다.

**반주 만들기**: 멜로디 **한 마디**를 찍으면 AI(Anticipatory Music Transformer)가 드럼·베이스·일렉기타·피아노 등의 반주를 만들어 **악기마다 따로 된 소리**(한 마디·반복)와 MIDI 로 준다 → 아래 [반주 만들기](#반주-만들기--멜로디-한-마디--악기마다-따로-된-소리).

| 파일 | 하는 일 |
|---|---|
| `tools/kaggle_music_panel.py` (v1.2) | 노트북 버튼 세 단계: **① 분석 + Claude 에게 보낼 글** → **② Claude 답 읽기** → **③ 곡 만들기** · `ai_samples()` AI 음색 견본 |
| `tools/kaggle_music_yue2.py` (v1.6) | 설치(처음 한 번) · `analyze`(트랙 분리·분석 → Claude 에게 보낼 글) · `run`(생성 → 멜로디 검사 → 저장소에) — 버튼이 이것을 부름 |
| `tools/music_analysis.py` (v1.0) | 트랙별 수치·멜로디 특징·곡 구성·Claude 에게 보낼 글·Claude 답 읽기 (순수 계산) |
| `tools/ai_samples.py` (v1.0) | AI 음색 견본 표(악기·톤별 혼자 연주 프롬프트)·작업 나누기·잘라 내기·'한 악기' 판정 (순수 계산) |
| `web/melody_pad.html` (v1.0) | **멜로디 찍기** 웹 페이지 원본 (claude.ai 페이지로 올림) — 한 마디 피아노 롤·건반·허밍 파일 → 멜로디 코드 |
| `tools/kaggle_accomp_panel.py` (v1.1) | 노트북 버튼: 멜로디 코드 → 악기 고르기 → **[반주 만들기]** → 악기별 플레이어 · `samples()` 음색 견본 플레이어 |
| `tools/kaggle_accomp.py` (v1.1) | 반주 설치·AI 생성(악기마다 차례로)·악기별 WAV·MIDI·저장소에 · `samples` 음색 견본 — 버튼이 이것을 부름 |
| `tools/accomp.py` (v1.1) | 멜로디 코드·AMT 토큰·악기 음역 거르기·16분 격자·MIDI 쓰기·사운드폰트 렌더·음색 견본 표 (순수 계산) |
| `tools/tests/` | GPU·네트워크 없이 도는 시험 116개: `python3 -m unittest tools.tests.test_kaggle_music_yue2 tools.tests.test_kaggle_music_panel tools.tests.test_music_analysis tools.tests.test_ai_samples tools.tests.test_accomp tools.tests.test_kaggle_accomp tools.tests.test_kaggle_accomp_panel -v` (numpy·mido·tinysoundfont 가 있으면 소리·MIDI 시험도, 사운드폰트 실제 렌더는 `ACCOMP_TEST_SF2=<sf2>`) |
| `results/<이름>/` | 올라간 곡 — `seed-N/audio.mp3`(320k)·`score.abc`(AI 악보)·`kaggle_run.json`·`similarity.json`, `analysis.json`·`prompt.txt`·`README.md` |
| `results/accomp/<이름>/` | 올라간 반주 — 악기마다 `<악기>_1bar.wav`·`<악기>.wav`·`<악기>.mid`, `mix.wav`·`all.mid`·`run.json`·`README.md` |

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

## 반주 만들기 — 멜로디 한 마디 → 악기마다 따로 된 소리

**① 멜로디 찍기** — 웹 페이지 <https://claude.ai/artifact/HinPmeUvyqN7KhBgrGYZ6D> (claude.ai 에 로그인한 본인 계정으로 열림, 원본 `web/melody_pad.html`)
- **칸 클릭**: 빈 칸 = 음 찍기(옆으로 끌면 그만큼 길게) · 음 클릭 = 지우기 · 음을 잡고 옆으로 끌기 = 길이 바꾸기. 위의 '음 길이'(16분·8분·4분…)가 클릭 한 번의 길이
- **건반으로 차례로**: 아래 건반(도레미…)을 누르면 커서 자리부터 하나씩 채워진다. 컴퓨터 자판도 됨 (`Z S X D C V …` 아래 옥타브, `Q 2 W 3 E …` 위 옥타브, `-` 쉼표, Space 재생)
- **허밍으로**: 휴대폰 녹음 앱으로 '라라라'를 또박또박 불러 녹음한 파일(m4a·mp3·wav)을 올리면 음 높이를 찾아 칸에 채운다 (녹음 전체를 한 마디로 맞추기 또는 템포 그대로, 음계 음으로 맞추기). 페이지가 마이크를 직접 쓸 수 없어 파일로 받는다. 찾은 음 높이 그래프가 함께 나오니 재생해 보고 칸에서 고친다
- **음계 표시**: 장조·단조를 고르면 음계 밖 줄이 흐려지고 으뜸음 줄에 점 — 어떤 음을 찍어야 어울리는지 보인다. 예시 멜로디 3개, 되돌리기, 반음·옥타브 옮기기
- **[코드 복사]** → `M1 bpm=110 steps=16 notes=76@0+2,74@2+1,…` (음 번호@시작 칸+길이 칸, 16칸 = 4/4 한 마디. 음 이름 `E5@0+2` 도 됨)

**② Kaggle 셀** (준비는 위 '처음 한 번 준비'와 같음 — GPU T4 x2·Internet On·GITHUB_TOKEN)

```python
import os, sys, importlib
from kaggle_secrets import UserSecretsClient
os.environ["GITHUB_TOKEN"] = UserSecretsClient().get_secret("GITHUB_TOKEN")   # 결과 올리기용 (값은 출력하지 않음)
!test -d /kaggle/working/music-ytb || git clone -q --depth 1 https://github.com/yeomin1024/music-ytb.git /kaggle/working/music-ytb
!cd /kaggle/working/music-ytb && git config core.fileMode false && git fetch -q origin main && git reset -q --hard origin/main
sys.path.insert(0, "/kaggle/working/music-ytb/tools")
import kaggle_accomp_panel; importlib.reload(kaggle_accomp_panel).panel()
```

**③ 패널**
- **멜로디 코드** 칸에 붙여 넣기 → 아래 요약 줄(음 몇 개·BPM·음역)로 확인. 코드 대신 **MIDI 올리기**도 됨 (첫 음이 있는 마디)
- **멜로디 악기**(피아노·플루트·바이올린·신스 리드·색소폰·트럼펫·일렉기타…) · **반주 악기** 체크 (기본 드럼·베이스·일렉기타(깨끗한)·피아노)
- **반복 마디**(기본 4 — 반주가 자리 잡게 한 마디를 반복해서 만든다) · **시드**(바꾸면 다른 반주) · AI 크기(기본 medium) · 이름 · 저장소에 올리기
- **[반주 만들기]** → 끝나면 악기마다 **한 마디**·**반복** 플레이어와 합친 소리

결과 (`/kaggle/working/accomp/<이름>/`, 올리기를 켜면 `results/accomp/<이름>/`):

| 파일 | 내용 |
|---|---|
| `<악기>_1bar.wav` | 그 악기만, 한 마디 (마지막 마디 — 이어 붙이면 반복된다) |
| `<악기>.wav` | 그 악기만, 반복 마디 전체 |
| `<악기>.mid` | 그 악기 MIDI — 작곡 프로그램(DAW)에 불러와 음색을 바꾸면 소리가 좋아진다 |
| `melody*` | 내 멜로디 (고른 멜로디 악기 소리) |
| `mix.wav`·`mix_1bar.wav`·`all.mid` | 모두 합친 소리·모든 악기 MIDI |
| `run.json` | 멜로디 코드·시드·악기별 음 수·음역·모델 리비전·설치 버전 |

악기 (`kaggle_accomp.py instruments`):

| 반주 | 멜로디로도 |
|---|---|
| 드럼 `drums` · 베이스 `bass` · 신스 패드 `synth_pad` | — |
| 피아노 `piano` · 일렉트릭 피아노 `epiano` · 일렉기타 깨끗한 `guitar_clean` / 드라이브 `guitar_drive` / 디스토션 `guitar_dist` · 통기타 `acoustic_guitar` · 스트링 `strings` · 브라스 `brass` | 됨 |
| — | 신스 리드 `synth_lead` · 플루트 `flute` · 바이올린 `violin` · 알토 색소폰 `sax` · 트럼펫 `trumpet` · 코러스 '아' `voice` |

### 음색 견본 듣기 (어떤 악기·톤인지 먼저 들어 보기)

반주 결과가 내는 소리(사운드폰트)를 악기·톤마다 **약 6초씩** 들어 본다. 모두 같은 짧은 악절 — **C–G–Am–F 두 마디, 100 BPM**. 멜로디 견본은 같은 멜로디, 코드 견본은 같은 코드 진행(통기타·클린은 스트럼, 나일론은 아르페지오, 오버드라이브·디스토션은 파워 코드, 뮤트·펑크는 짧게 끊어 치기). AI 없이 소리만 내므로 **GPU 필요 없음**, 1분 안쪽.

```python
import sys, importlib
!test -d /kaggle/working/music-ytb || git clone -q --depth 1 https://github.com/yeomin1024/music-ytb.git /kaggle/working/music-ytb
!cd /kaggle/working/music-ytb && git config core.fileMode false && git fetch -q origin main && git reset -q --hard origin/main
sys.path.insert(0, "/kaggle/working/music-ytb/tools")
import kaggle_accomp_panel; importlib.reload(kaggle_accomp_panel).samples()          # 전부 38개
# importlib.reload(kaggle_accomp_panel).samples("electric,drums")                     # 묶음만: acoustic·electric·piano·drums·bass·synth
```

셀 아래에 묶음별로 이름·사운드폰트 번호·플레이어가 나온다. 파일은 `/kaggle/working/accomp/samples/<묶음>/<견본>.wav`. 견본마다 크기를 맞춰서(최고점 -1dBFS) **톤끼리 크기는 비교하지 않는다**.
명령으로는 `!python /kaggle/working/music-ytb/tools/kaggle_accomp.py samples [--only electric] [--bpm 100]`.

| 묶음 | 견본 (괄호 = 반주 만들기에서 바로 고를 수 있는 악기 키) |
|---|---|
| 어쿠스틱 기타 | 스틸 통기타 — 멜로디 (`acoustic_guitar`) · 스틸 통기타 — 코드 진행 (스트럼) (`acoustic_guitar`) · 나일론 기타 — 멜로디 · 나일론 기타 — 코드 진행 (아르페지오) · 12현 기타 — 코드 진행 (스트럼) |
| 일렉트릭 기타 | 일렉 클린 — 멜로디 (`guitar_clean`) · 일렉 클린 — 코드 진행 (스트럼) (`guitar_clean`) · 일렉 클린 + 코러스 — 코드 진행 · 재즈 기타 — 멜로디 · 재즈 기타 — 코드 진행 · 뮤트 — 코드 진행 (8분 뮤트) · 펑크 기타 — 코드 진행 (16분 커팅) · 오버드라이브 — 멜로디 (`guitar_drive`) · 오버드라이브 — 코드 진행 (파워 코드) (`guitar_drive`) · 디스토션 — 멜로디 (`guitar_dist`) · 디스토션 — 코드 진행 (파워 코드) (`guitar_dist`) |
| 그랜드 피아노 | 그랜드 피아노 — 멜로디 (`piano`) · 그랜드 피아노 — 코드 진행 (`piano`) · 그랜드 피아노 — 멜로디 + 코드 (두 손) (`piano`) |
| 드럼 | 드럼 — 스탠더드 (`drums`) · 드럼 — 룸 · 드럼 — 파워 (록) · 드럼 — 일렉트로닉 · 드럼 — 808/909 · 드럼 — 재즈 · 드럼 — 브러시 |
| 베이스 | 핑거 베이스 (`bass`) · 피크 베이스 · 프렛리스 베이스 · 슬랩 베이스 · 신스 베이스 1 · 신스 베이스 101 |
| 신디사이저 | 스퀘어 리드 — 멜로디 (`synth_lead`) · 소우 리드 — 멜로디 · 폴리신스 — 코드 진행 (스탭) · 브라이트 소우 스택 — 코드 진행 (스탭) · 웜 패드 — 코드 진행 (`synth_pad`) · 헤일로 패드 — 코드 진행 |

괄호가 없는 소리(나일론·12현·재즈·뮤트·펑크 기타, 드럼 세트들, 슬랩 베이스, 소우 리드 등)는 사운드폰트에는 있지만 반주 만들기에서는 아직 못 고른다. 반주 AI 는 악기를 GM 번호로만 구분해서, 같은 번호의 다른 뱅크 소리(예: 코러스 클린 8/27)는 렌더할 때만 바꿀 수 있다.

명령으로 (버튼 대신):

```
!python /kaggle/working/music-ytb/tools/kaggle_accomp.py run --melody "M1 bpm=110 steps=16 notes=76@0+2,74@2+1,72@3+1" \
    --melody-instrument piano --instruments drums,bass,guitar_clean,piano --loops 4 --seed 1 --push
```

어떻게 만드나 (로그 단계):
1. `[SETUP]`·`[CHECK]` — 세션마다 처음 한 번: anticipation(커밋 고정)·tinysoundfont·mido 설치, 사운드폰트 내려받기(sha256 확인), AI 어휘 상수가 도구와 같은지 `vocab=ok`
2. `[MELODY]` — 멜로디 코드 → 한 마디를 반복 마디 수만큼 이어 붙임
3. `[LOAD]` — AI 모델 `stanford-crfm/music-medium-800k` (리비전은 `run.json` 에)
4. `[GEN] inst=… notes=…` — **악기마다 한 번씩 차례로** (드럼 → 베이스 → 피아노·기타 → 스트링·패드). 조건 = 내 멜로디 + 앞에서 만든 악기들, 다음 음은 **그 악기·그 음역만** 나오게 거른다 → 한 악기만 든 소리. 음이 너무 적으면 다른 시드로 다시 (최대 3번, `warning=… 음이 적음` 이면 시드를 바꿔 다시)
5. `[SAVE]` — 16분음표 격자에 맞춘 뒤 악기별 WAV·MIDI. 트랙끼리 크기 균형은 그대로 두고 가장 큰 트랙 최고점을 -1dBFS 로
6. `[PUSH]` — `results/accomp/<이름>/` 에 (내가 찍은 멜로디라 원곡 문제 없음)

AI 모델과 소리:
- **Anticipatory Music Transformer** (Stanford CRFM, 코드·가중치 Apache-2.0, Lakh MIDI 로 학습) — 멜로디를 '조건'으로 주면 그에 맞는 다른 악기 연주를 만드는 모델.
- 처음 고른 **GETMusic**(Microsoft, 트랙별 생성)은 쓸 수 없다: 가중치 내려받기 링크가 지워졌고(이슈 #203·#213·#215, 답 없음) 저장소도 2026-10-01 보관 처리됐다.
- 소리: tinysoundfont 0.3.7 + 사운드폰트 **GeneralUser GS v2.0.3** (개인·상업 음악 제작에 제한 없이 사용 가능 — 그 라이선스 문서)

한계:
- **Kaggle 실제 실행은 아직 검증 전** — AI 생성은 이 저장소 시험에서 가짜 모델로만 확인했다 (토큰 모양·악기 거르기·순서·재시도·내보내기·올리기). 첫 실행에서 `[CHECK] vocab=ok`, `[GEN] inst=drums notes=…` 줄을 확인한다.
- 사운드폰트 소리는 실제 연주보다 단순하다 (특히 일렉기타) → MIDI 를 DAW 에서 다른 음색으로.
- AI 가 셈여림(세기)을 만들지 않아 악기마다 같은 세기. 한 마디만으로는 어울리는 화음이 여럿 → 시드를 바꿔 여러 번 들어 보기.
- 한 번에 100초까지 (반복 마디 × 마디 길이 95초 이하).

| 로그 | 할 일 |
|---|---|
| `[INPUT] … 멜로디` · `[PANEL] 멜로디 코드: …` | 멜로디 찍기 페이지의 [코드 복사] 그대로 붙여 넣기 (겹치는 음·마디 밖 음은 안 됨) |
| `[CHECK] … AMT 어휘가 다름` | anticipation 버전 문제 — 이 저장소 이슈로 알려 주기 (도구가 커밋을 고정하므로 보통 안 남) |
| `[LOAD] … error=` | Settings → Internet: On 확인. GPU 메모리 부족이면 AI 크기 small |
| `[GEN] … error=… out of memory` | AI 크기 small 또는 반복 마디를 줄여 다시 |
| `[GEN] … 음이 적음` | 시드를 바꾸거나 반복 마디를 늘려 다시 |
| `[PUSH] … 올리기 실패` | 결과는 `/kaggle/working/accomp/<이름>/` 에 있음 |

## AI 음색 견본 — 소리까지 AI(YuE2)가 만든 악기별 몇 초

위 '음색 견본 듣기'는 사운드폰트 소리다. 이것은 **YuE2 가 소리 자체를 만든다** (녹음 같은 소리). 악기·톤마다 **10초 안팎**, **한 악기 소리만** 남긴다.

```python
import sys, importlib
!test -d /kaggle/working/music-ytb || git clone -q --depth 1 https://github.com/yeomin1024/music-ytb.git /kaggle/working/music-ytb
!cd /kaggle/working/music-ytb && git config core.fileMode false && git fetch -q origin main && git reset -q --hard origin/main
sys.path.insert(0, "/kaggle/working/music-ytb/tools")
import kaggle_music_panel; importlib.reload(kaggle_music_panel).ai_samples("strings,piano")   # 묶음 고르기 (비우면 전부)
# importlib.reload(kaggle_music_panel).ai_samples(keys="electric_clean_melody,electric_dist_chords")   # 견본 고르기
```

- 준비: GPU T4 x2 · Internet On (위 '처음 한 번 준비' 1번). 저장소에 올리지 않으니 GITHUB_TOKEN 은 필요 없다.
- 두 GPU 가 동시에 돌면 RAM 이 빠듯할 수 있다 (로그 `RAM 이 적어…` 경고) — 셀이 멈추면 명령에 `--gpus 0`.
- **오래 걸린다 (추정, 아직 Kaggle 에서 재지 않음)**: 처음 한 번 설치·모델 받기 몇 분 + 견본 하나에 생성 1~3번 × 몇 분. 두 GPU 가 나눠 만든다. 처음에는 묶음 하나·둘로 해 보고, 첫 `[GEN] … status=done minutes=` 줄로 한 번에 걸리는 시간을 본다.
- 셀 아래에 견본마다 **판정**(목표 트랙 비중)·**한 악기만** 플레이어·**AI 원래 소리** 플레이어. 파일은 `/kaggle/working/music/ai_samples/<이름>/clips/<묶음>/<견본>.mp3`(한 악기만)·`<견본>_ai_raw.mp3`(걸러 내기 전), 기록 `index.json`.

'한 악기만'을 지키는 방법 (YuE2 는 원래 노래 전체를 만드는 모델이라 혼자 연주를 보장하지 않는다):
1. **프롬프트**로 혼자 연주를 요청 — 예: `solo string section … orchestral strings only, solo performance, dry studio recording, no other instruments, no drums, no bass, no percussion, no vocals, 100 BPM, C major` (견본마다 `tools/ai_samples.py`). 단일 멜로디는 선율만 계획(`melody`), 코드 진행은 화음까지 계획(`full`)
2. YuE2 연주곡 (계획 섹션 하나 `Verse` — 짧게) → 앞의 조용한 부분을 빼고 10초 (끝 0.6초 줄임)
3. **Demucs**(htdemucs_6s)로 드럼·베이스·기타·피아노·보컬·그 밖으로 나눠 **목표 트랙 비중**을 잰다 → 로그 `[SAMPLES] sample=… seed=… share=0.91 others=피아노 6%`
4. 80% 미만이면 **다른 시드로 다시** (최대 3번, `tries`) → 가장 높은 것을 고른다
5. 마지막 파일은 **목표 트랙만 걸러 낸 소리**. 판정: 80% 이상 '거의 그 악기만' · 50~80% '다른 소리가 조금 섞여 걸러 냄' · 50% 미만 '모델이 혼자 연주를 잘 안 함'

| 묶음 | 견본 | 걸러 낼 트랙 |
|---|---|---|
| 어쿠스틱 기타 | 통기타 — 단일 멜로디 `acoustic_melody` · 통기타 — 코드 진행 `acoustic_chords` | 기타 |
| 일렉트릭 기타 | 일렉 클린톤 — 단일 멜로디 `electric_clean_melody` · 일렉 클린톤 — 코드 진행 `electric_clean_chords` · 일렉 크런치(오버드라이브) — 단일 멜로디 `electric_crunch_melody` · 일렉 크런치(오버드라이브) — 코드 진행 `electric_crunch_chords` · 일렉 디스토션 — 단일 멜로디 `electric_dist_melody` · 일렉 디스토션 — 코드 진행 (파워 코드) `electric_dist_chords` · 재즈 기타 — 코드 진행 `electric_jazz_chords` | 기타 |
| 그랜드 피아노 | 그랜드 피아노 — 단일 멜로디 `piano_melody` · 그랜드 피아노 — 코드 진행 `piano_chords` | 피아노 |
| 스트링 | 스트링 — 단일 멜로디 (현악 합주) `strings_melody` · 스트링 — 코드 진행 (현악 합주) `strings_chords` | 그 밖 |
| 드럼 | 드럼 — 어쿠스틱 드럼 세트 `drums_acoustic` · 드럼 — 전자 드럼 (드럼 머신) `drums_electronic` | 드럼 |
| 베이스 | 일렉 베이스 — 베이스 라인 `bass_electric` · 신스 베이스 — 베이스 라인 `bass_synth` | 베이스 |
| 신디사이저 | 신스 리드 — 단일 멜로디 `synth_lead` · 신스 패드 — 코드 진행 `synth_pad` | 그 밖 |

한계:
- Demucs 트랙이 6개뿐이라 **스트링·신스는 둘 다 '그 밖' 트랙** — 스트링 견본에 신스 패드가 섞여 나오면 나눌 수 없다 (그때 판정 비중은 높게 나온다 → AI 원래 소리와 비교해 듣기). 기타 톤끼리(클린·디스토션)도 같은 '기타' 트랙.
- 걸러 낸 소리에는 분리 흔적(울림이 줄거나 다른 악기 잔향)이 남을 수 있다. 비중이 높을수록 적다.
- 80%·50% 기준은 경험칙이다 (Kaggle 에서 실제 값을 보고 조정).
- 드럼 견본도 YuE2 는 선율 악보를 먼저 쓰므로 멜로디 악기가 섞이기 쉽다 → 드럼 트랙만 남기지만 비중이 낮으면 `tries` 를 늘린다.
- 라이선스: YuE2 가중치 CC BY-NC 4.0 + 창작자 추가 허가 (위 '라이선스').

명령으로: `!python /kaggle/working/music-ytb/tools/kaggle_music_yue2.py samples --only strings --seconds 10 --tries 3` · 빠진 것만 다시: `--keys <견본>` · 다른 결과: `--seed 10`.

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
- 반주: anticipation 커밋 `af373979…`(2024-03-18), tinysoundfont 0.3.7(의존성 없이), mido 1.3.3, 사운드폰트 GeneralUser GS v2.0.3 커밋 `684543d5…`·sha256 `9575028c…`, 모델 `stanford-crfm/music-{small,medium,large}-800k`(리비전은 `run.json` 에)

## 라이선스

YuE2 코드 Apache-2.0, 가중치 CC BY-NC 4.0 + 창작자 추가 허가 — 개인 창작자는 만든 곡으로 수익을 내도 된다(로열티 없음), 회사의 상업 이용은 별도 계약. 자세히: [YuE2 README 'License'](https://github.com/multimodal-art-projection/YuE#license). 참고 곡의 저작권은 원작자에게 있다 — 원곡 파일은 공개 저장소·공개 노트북에 올리지 않는다.
