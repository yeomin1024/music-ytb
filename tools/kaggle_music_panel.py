#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VERSION: v1.1 — 2026-10-16 — 사용자 지시 "음악 하나랑 그 음악을 듣고 느낀 멘트와 떠오르는 사진 … 트랙을 분리한 후 … 내 멘트와 이미지
#          종합해서 음악 하나 … 가사 여부는 내가 체크" + "멘트·사진은 내가 클로드한테 물어보고 결과 넣을테니까 … 선택 2(가사 초안을
#          보고 고친 뒤 생성)는 그렇게 해": 세 단계 — ① 곡·멘트·가사 여부 → 분석 + Claude 에게 보낼 글 ② Claude 답 붙여 넣기 → 프롬프트·
#          가사 초안 ③ 고친 뒤 곡 만들기 (v1.0: [곡 올리기] + [비슷한 곡 만들기] 한 번에)
r"""
Kaggle 노트북 셀(README.md '노트북 셀')에서 버튼으로: 곡 올리기 → 분석 → Claude 와 함께 프롬프트·가사 → 곡 만들기.

  import sys, importlib; sys.path.insert(0, "/kaggle/working/music-ytb/tools")
  import kaggle_music_panel; importlib.reload(kaggle_music_panel).panel()

  ① [곡 올리기](mp3 권장) · 멘트 칸 · [가사 넣기] 체크 · 이름 → [① 분석하고 Claude 에게 보낼 글 만들기]
     = kaggle_music_yue2.py analyze --ref <곡> --name <이름> --comment-file <멘트> [--lyrics]
     → 트랙 분리·트랙별 분석·멜로디 특징 → 'Claude 에게 보낼 글' 칸 (전체 선택 → 복사 → Claude 대화에 붙이고 사진 첨부)
  ② Claude 답을 'Claude 답' 칸에 붙여 넣기 → [② 답 읽기] → STYLE·SECTIONS·가사 칸이 채워짐 (가사는 고쳐도 됨)
     (Claude 를 건너뛰면 ①이 채운 자동 프롬프트 그대로 ③ 가능)
  ③ 곡 수·저장소에 올리기 → [③ 곡 만들기]
     = kaggle_music_yue2.py run --ref <곡> --name <이름> --style … (--sections … | --mode song --lyrics-file …) --count N [--push]
     → 곡마다 멜로디 검사 결과·플레이어

  올린 원곡·멘트 파일은 /tmp/yue2/upload/ 에만 (Kaggle 출력·저장소에 남기지 않음). 실행 중에는 버튼이 잠긴다.
  멈추려면 Kaggle 의 셀 멈춤(■) — 하위 프로세스까지 함께 멈춘다. 로그는 아래 칸과 /kaggle/working/music/panel.log 에 함께.

위젯이 없는 곳(터미널)에서는 버튼 대신 kaggle_music_yue2.py analyze / run 명령 (그 파일 머리말).
※ 연구·교육용 도구. 멜로디 검사는 경험칙이며 법적 판단이 아니다.
"""
# CHANGELOG
#   v1.0 — 2026-10-15 — 새 파일. uploaded_files(ipywidgets 7·8 둘 다)·safe_filename·save_upload·build_command·check_inputs·
#          run_command(프로세스 그룹째 멈춤)·results_rows·panel. 영향: 새 파일, tools/tests/test_kaggle_music_panel.py
#   v1.1 — 2026-10-16 — 세 단계 패널 (사용자 지시). 새것: 멘트 칸·가사 넣기 체크·Claude 에게 보낼 글 칸·Claude 답 칸·STYLE/SECTIONS/가사 칸,
#          check_upload·check_make·build_analyze_command·build_run_command·write_text. 빠진 것: check_inputs·build_command
#          (한 번에 만들기 — ①→③ 으로 같은 일). 멜로디 검사·올리기·멈춤은 그대로. 영향: panel 전체, 시험
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import kaggle_music_yue2 as km  # noqa: E402 — 표준 라이브러리만 불러오는 모듈 (GPU·torch 없음)
import music_analysis as ma  # noqa: E402

PANEL_VERSION = "v1.1"
UPLOAD_DIR = km.WORK / "upload"  # /tmp — Kaggle 출력으로 저장되지 않는다
LOG_FILE = km.OUT / "panel.log"
DEFAULT_COUNT = km.REF_COUNT
MAX_COUNT = 16
ACCEPT = "audio/*," + ",".join(km.AUDIO_EXT)
DEFAULT_SECTIONS = "Intro,Verse,Chorus,Verse,Chorus,Bridge,Chorus,Outro"


def uploaded_files(value):
    """FileUpload.value → [(파일 이름, bytes)]. ipywidgets 8 = 튜플[dict(name, content…)], 7 = dict{이름: {content, metadata}}."""
    if not value:
        return []
    if isinstance(value, dict):
        return [(name, bytes(item["content"])) for name, item in value.items()]
    return [(item["name"], bytes(item["content"])) for item in value]


def safe_filename(name):
    """경로·제어 문자를 뺀 파일 이름 (한글은 그대로, 확장자는 소문자, 80자까지)."""
    stem, ext = os.path.splitext(os.path.basename(str(name)))
    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._") or "reference"
    return stem[:80] + ext.lower()


def check_upload(files, name):
    """① 단계 입력 → 오류 문구 (문제 없으면 None)."""
    if not files:
        return "먼저 [곡 올리기]로 곡 파일 하나를 올려 주세요. (올렸는데도 이 문구가 나오면 파일이 커서 실패한 것 — mp3 로 다시)"
    if len(files) > 1:
        return "곡은 하나만 올려 주세요."
    fname, data = files[0]
    if Path(fname).suffix.lower() not in km.AUDIO_EXT:
        return f"오디오 파일이 아님: {fname} ({', '.join(km.AUDIO_EXT)})"
    if not data:
        return "올린 파일이 비었음 — 다시 올려 주세요."
    if name and not km.NAME_RE.fullmatch(name):
        return f"이름은 영문·숫자·_.- 만: {name!r} (예: my-song-01)"
    return None


def check_make(analyzed, style, sections, want_lyrics, lyrics, count, push, token_present):
    """③ 단계 입력 → 오류 문구 (문제 없으면 None)."""
    if not analyzed:
        return "먼저 ① 분석을 끝내 주세요."
    if not style.strip():
        return "STYLE 이 비었음 — ② Claude 답을 읽거나 직접 써 주세요."
    if "\n" in style.strip():
        return "STYLE 은 한 줄로."
    if want_lyrics:
        if not re.search(r"^\[[A-Za-z][A-Za-z0-9 -]*\]\s*$", lyrics, flags=re.M) or not re.search(r"^(?!\[).*\S", lyrics, flags=re.M):
            return "가사 넣기를 켰는데 가사 칸에 [Verse] 같은 태그 줄과 가사가 없음."
    else:
        try:
            km.section_lyrics(sections)
        except ValueError as e:
            return f"SECTIONS: {e}"
    if not isinstance(count, int) or not 1 <= count <= MAX_COUNT:
        return f"곡 수는 1~{MAX_COUNT}"
    if push and not token_present:
        return ("저장소에 올리려면 GITHUB_TOKEN 이 필요 — 셀 첫 줄(UserSecretsClient)과 Kaggle Secrets 연결을 확인하거나, "
                "'저장소에 올리기'를 끄고 실행")
    return None


def save_upload(fname, data, dest_dir=None):
    """올린 곡 → dest_dir(기본 UPLOAD_DIR)/<안전한 이름> (같은 이름이 있으면 덮어씀) → 경로."""
    dest_dir = Path(dest_dir or UPLOAD_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / safe_filename(fname)
    path.write_bytes(data)
    return path


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def build_analyze_command(ref, name, comment_file, lyrics, python=sys.executable):
    cmd = [python, str(TOOLS / "kaggle_music_yue2.py"), "analyze", "--ref", str(ref), "--name", name]
    if comment_file:
        cmd += ["--comment-file", str(comment_file)]
    if lyrics:
        cmd.append("--lyrics")
    return cmd


def build_run_command(ref, name, count, push, style, sections="", lyrics_file=None, python=sys.executable):
    cmd = [python, str(TOOLS / "kaggle_music_yue2.py"), "run", "--ref", str(ref), "--name", name, "--count", str(count),
           "--style", style.strip()]
    if lyrics_file:
        cmd += ["--mode", "song", "--lyrics-file", str(lyrics_file)]
    else:
        cmd += ["--mode", "instrumental", "--sections", sections]
    if push:
        cmd.append("--push")
    return cmd


def run_command(cmd, write=print, log_file=None):
    """명령을 돌리며 한 줄씩 write·로그 파일로 → 종료 코드. 셀을 멈추면(KeyboardInterrupt) 하위 프로세스까지 멈춘다."""
    log_file = Path(log_file or LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_file.open("a", encoding="utf-8") as lf:
        def emit(line):
            write(line)
            lf.write(line + "\n")
            lf.flush()
        emit(f"[PANEL] {datetime.now():%Y-%m-%d %H:%M:%S} {PANEL_VERSION} cmd={' '.join(shlex.quote(c) for c in cmd[1:])}")
        try:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    start_new_session=True)  # 프로세스 그룹 — 멈출 때 GPU 작업 프로세스까지
        except OSError as e:
            emit(f"[PANEL] level=error error={e} hint=저장소 경로·파이썬 확인 ({ROOT})")
            return 1
        try:
            for line in proc.stdout:
                emit(line.rstrip("\n"))
            rc = proc.wait()
        except KeyboardInterrupt:
            emit("[PANEL] 멈춤 요청 — 하위 프로세스를 멈추는 중")
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=60)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            rc = 130
        finally:
            proc.stdout.close()
        emit(f"[PANEL] done exit={rc} minutes={(time.time() - t0) / 60:.1f}"
             + ("" if rc == 0 else " hint=위 [ERROR]·level=error 줄 확인"))
    return rc


def results_rows(name_dir):
    """<이름>/seed-* → [(곡, 길이 초, 검사 결과, mp3 경로|None)]."""
    rows = []
    for d in sorted(Path(name_dir).glob("seed-*")):
        sim, run = d / "similarity.json", d / "kaggle_run.json"
        verdict = json.loads(sim.read_text(encoding="utf-8")).get("verdict_ko", "?") if sim.is_file() else "검사 전"
        seconds = json.loads(run.read_text(encoding="utf-8")).get("metrics", {}).get("seconds") if run.is_file() else None
        mp3 = d / "audio.mp3"
        rows.append((d.name, seconds, verdict, mp3 if mp3.is_file() else None))
    return rows


def panel():
    """Kaggle·Jupyter: ① 곡·멘트·가사 여부 → 분석 + Claude 에게 보낼 글 ② Claude 답 → 프롬프트·가사 ③ 곡 만들기."""
    try:
        import ipywidgets as W
        from IPython.display import Audio, display
    except ImportError:
        print("[PANEL] ipywidgets 없음 → 버튼 대신: kaggle_music_yue2.py analyze … / run … (그 파일 머리말)")
        return 1
    wide = W.Layout(width="98%")
    up = W.FileUpload(accept=ACCEPT, multiple=False, description="곡 올리기")
    status = W.HTML("<i>아직 올린 곡이 없음</i>")
    comment = W.Textarea(value="", placeholder="이 곡을 듣고 느낀 점·떠오르는 장면 (Claude 에게 보낼 글에 들어감)",
                         description="멘트", layout=W.Layout(width="98%", height="90px"))
    want_lyrics = W.Checkbox(value=False, description="가사 넣기")
    name = W.Text(value="", placeholder="비우면 ref-날짜시각", description="이름")
    analyze_btn = W.Button(description="① 분석하고 Claude 에게 보낼 글 만들기", button_style="primary", icon="search",
                           layout=W.Layout(width="320px"))
    request = W.Textarea(value="", placeholder="① 이 끝나면 여기에 'Claude 에게 보낼 글'이 나옴", description="Claude 에게",
                         layout=W.Layout(width="98%", height="240px"))
    answer = W.Textarea(value="", placeholder="Claude 답(코드 블록 포함 그대로)을 여기에 붙여 넣기", description="Claude 답",
                        layout=W.Layout(width="98%", height="160px"))
    parse_btn = W.Button(description="② 답 읽기", icon="check", layout=W.Layout(width="160px"))
    style = W.Textarea(value="", placeholder="STYLE (영어 한 줄)", description="STYLE", layout=W.Layout(width="98%", height="60px"))
    sections = W.Text(value=DEFAULT_SECTIONS, description="SECTIONS", layout=wide)
    lyrics = W.Textarea(value="", placeholder="가사 넣기를 켰을 때 — [Verse] 같은 태그 줄 + 가사 (고쳐도 됨)", description="가사",
                        layout=W.Layout(width="98%", height="240px"))
    count = W.BoundedIntText(value=DEFAULT_COUNT, min=1, max=MAX_COUNT, description="곡 수")
    push = W.Checkbox(value=True, description="통과한 곡 저장소에 올리기")
    make_btn = W.Button(description="③ 곡 만들기", button_style="success", icon="play", layout=W.Layout(width="200px"))
    out = W.Output(layout=W.Layout(border="1px solid #cfd6e2", max_height="560px", overflow="auto", padding="6px"))
    note = W.HTML(
        "<b>①</b> 곡 올리기(mp3 권장) · 멘트 · 가사 넣기 여부 → 분석(트랙 분리 포함, 몇 분) → 'Claude 에게' 칸의 글을 "
        "<b>전체 선택 → 복사 → Claude 대화에 붙이고 사진을 첨부</b>해 보내기 · "
        "<b>②</b> Claude 답을 그대로 붙여 넣고 [답 읽기] → STYLE·SECTIONS·가사 확인·고치기 · "
        "<b>③</b> 곡 만들기 (곡당 수십 분). 원곡 소리는 분석에만 쓰고 결과 곡에 넣지 않는다. 원곡과 멜로디를 비교해 <b>통과한 곡만</b> 저장소에.")
    state = {"ref": None, "name": None, "analyzed": False}
    buttons = (up, analyze_btn, parse_btn, make_btn)

    def lock(flag):
        for w in buttons:
            w.disabled = flag

    def on_upload(_change):
        files = uploaded_files(up.value)
        if files:
            fname, data = files[0]
            status.value = f"올린 곡: <b>{safe_filename(fname)}</b> ({len(data) / 1e6:.1f}MB)"

    def on_analyze(_):
        lock(True)
        try:
            with out:
                files = uploaded_files(up.value)
                run_name = name.value.strip() or f"ref-{datetime.now():%Y%m%d-%H%M%S}"
                error = check_upload(files, run_name)
                if error:
                    print(f"[PANEL] {error}")
                    return
                name.value = run_name  # ③ 이 같은 이름을 쓴다
                ref = save_upload(*files[0])
                comment_file = write_text(UPLOAD_DIR / f"{run_name}.comment.txt", comment.value)
                print(f"[PANEL] ① 분석 — 곡 {ref.name} ({ref.stat().st_size / 1e6:.1f}MB), 이름 {run_name}, 가사 {'넣기' if want_lyrics.value else '없음'}")
                rc = run_command(build_analyze_command(ref, run_name, comment_file, want_lyrics.value), write=print)
                name_dir = km.OUT / run_name
                if rc != 0 or not (name_dir / "claude_request.txt").is_file():
                    print("[PANEL] 분석 실패 — 위 오류 줄 확인")
                    return
                state.update(ref=ref, name=run_name, analyzed=True)
                request.value = (name_dir / "claude_request.txt").read_text(encoding="utf-8")
                if (name_dir / "prompt.txt").is_file() and not style.value.strip():
                    style.value = (name_dir / "prompt.txt").read_text(encoding="utf-8").strip()  # Claude 를 건너뛸 때 쓰는 자동 프롬프트
                print("[PANEL] ① 끝 — 'Claude 에게' 칸의 글을 복사해 Claude 에 붙이고 사진을 첨부해 보내 주세요. 답을 'Claude 답' 칸에.")
        finally:
            lock(False)

    def on_parse(_):
        with out:
            result, errors, warnings = ma.parse_claude_answer(answer.value, want_lyrics.value)
            for e in errors:
                print(f"[PANEL] 답 오류: {e}")
            for w in warnings:
                print(f"[PANEL] 주의: {w}")
            if errors:
                print("[PANEL] Claude 에게 '형식 그대로 다시 써 줘'라고 하거나 칸을 직접 고쳐 주세요.")
                return
            style.value = result["style"]
            sections.value = result["sections"]
            lyrics.value = result["lyrics"]
            print(f"[PANEL] ② 답 읽음 — STYLE {len(result['style'])}자, SECTIONS {result['sections']}"
                  + (f", 가사 {len([ln for ln in result['lyrics'].splitlines() if ln.strip()])}줄" if want_lyrics.value else "")
                  + ". 고친 뒤 ③")

    def on_make(_):
        lock(True)
        try:
            with out:
                error = check_make(state["analyzed"], style.value, sections.value, want_lyrics.value, lyrics.value,
                                   count.value, push.value, bool(os.environ.get("GITHUB_TOKEN")))
                if error:
                    print(f"[PANEL] {error}")
                    return
                run_name, name_dir = state["name"], km.OUT / state["name"]
                lyrics_file = write_text(name_dir / "lyrics.txt", lyrics.value.strip() + "\n") if want_lyrics.value else None
                cmd = build_run_command(state["ref"], run_name, count.value, push.value, " ".join(style.value.split()),
                                        sections.value.strip(), lyrics_file)
                rc = run_command(cmd, write=print)
                rows = results_rows(name_dir)
                if rows:
                    print(f"[PANEL] 결과 {len(rows)}곡 — /kaggle/working/music/{run_name}/" + ("" if rc == 0 else " (일부 실패)"))
                for seed, seconds, verdict, mp3 in rows:
                    length = f"{int(seconds // 60)}:{int(seconds % 60):02d}" if seconds else "-"
                    print(f"  {seed}  {length}  멜로디 검사: {verdict}")
                    if mp3:
                        display(Audio(filename=str(mp3)))
        finally:
            lock(False)

    up.observe(on_upload, names="value")
    analyze_btn.on_click(on_analyze)
    parse_btn.on_click(on_parse)
    make_btn.on_click(on_make)
    display(W.VBox([note, W.HBox([up, status]), comment, W.HBox([want_lyrics, name]), analyze_btn, request,
                    answer, parse_btn, style, sections, lyrics, W.HBox([count, push]), make_btn, out]))
    return 0


if __name__ == "__main__":
    print(__doc__)
