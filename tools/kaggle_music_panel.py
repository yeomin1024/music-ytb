#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VERSION: v1.0 — 2026-10-15 — 사용자 지시 "업로드 버튼 만들어서 하도록 수정해": Kaggle 노트북 셀에 [곡 올리기] 버튼 + 설정 칸 +
#          [▶ 비슷한 곡 만들기] 버튼 → kaggle_music_yue2.py run --ref <올린 곡> … (+ Add Input 대신)
r"""
Kaggle 노트북 셀(README.md '노트북 셀')에서 참고 곡을 버튼으로 올리고, 비슷한 분위기의 곡을 만든다.

  import sys, importlib; sys.path.insert(0, "/kaggle/working/music-ytb/tools")
  import kaggle_music_panel; importlib.reload(kaggle_music_panel).panel()

  - [곡 올리기]: 곡 파일 하나 (mp3 권장 — 큰 wav 는 노트북 업로드 한도에 걸릴 수 있음).
    올린 원곡은 /tmp/yue2/upload/ 에만 둔다 (Kaggle 출력 /kaggle/working 에 남기지 않음 · 저장소에는 절대 안 올림)
  - 설정: 이름(비우면 ref-날짜시각) · 장르 더하기(예: J-pop band) · 곡 수(기본 4) · 통과한 곡 저장소에 올리기(기본 켬)
  - [▶ 비슷한 곡 만들기]: python tools/kaggle_music_yue2.py run --ref <올린 곡> --count N --name 이름 [--style-extra …] [--push]
    실행 중에는 버튼이 잠긴다. 멈추려면 Kaggle 의 셀 멈춤(■) — 하위 프로세스까지 함께 멈춘다
  - 끝나면 곡마다 멜로디 검사 결과와 플레이어. 로그는 아래 칸과 /kaggle/working/music/panel.log 에 함께

위젯이 없는 곳(터미널)에서는 버튼 대신:
  python tools/kaggle_music_yue2.py run --ref <곡 파일> --count 4 --push
※ 연구·교육용 도구. 멜로디 검사는 경험칙이며 법적 판단이 아니다.
"""
# CHANGELOG
#   v1.0 — 2026-10-15 — 새 파일. uploaded_files(ipywidgets 7·8 둘 다)·safe_filename·save_upload·build_command·check_inputs·
#          run_command(프로세스 그룹째 멈춤)·results_rows·panel. 영향: 새 파일, tools/tests/test_kaggle_music_panel.py
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import json
from datetime import datetime
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import kaggle_music_yue2 as km  # noqa: E402 — 표준 라이브러리만 불러오는 모듈 (GPU·torch 없음)

PANEL_VERSION = "v1.0"
UPLOAD_DIR = km.WORK / "upload"  # /tmp — Kaggle 출력으로 저장되지 않는다
LOG_FILE = km.OUT / "panel.log"
DEFAULT_COUNT = km.REF_COUNT
MAX_COUNT = 16
ACCEPT = "audio/*," + ",".join(km.AUDIO_EXT)


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


def check_inputs(files, name, count, push, token_present):
    """→ 오류 문구 (문제 없으면 None)."""
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
        return f"이름은 영문·숫자·_.- 만: {name!r} (예: jpop-like-01)"
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


def build_command(ref, name, count, extra, push, python=sys.executable):
    """→ kaggle_music_yue2.py run 명령 (리스트)."""
    cmd = [python, str(TOOLS / "kaggle_music_yue2.py"), "run", "--ref", str(ref), "--count", str(count), "--name", name]
    if extra.strip():
        cmd += ["--style-extra", extra.strip()]
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
             + ("" if rc == 0 else " hint=위 [ERROR]·level=error 줄 확인 (같은 이름으로 다시 누르면 분석은 건너뜀)"))
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
    """Kaggle·Jupyter: [곡 올리기] + 설정 + [▶ 비슷한 곡 만들기]. ipywidgets 가 없으면 안내만."""
    try:
        import ipywidgets as W
        from IPython.display import Audio, display
    except ImportError:
        print("[PANEL] ipywidgets 없음 → 버튼 대신: python tools/kaggle_music_yue2.py run --ref <곡 파일> --count 4 --push")
        return 1
    up = W.FileUpload(accept=ACCEPT, multiple=False, description="곡 올리기")
    name = W.Text(value="", placeholder="비우면 ref-날짜시각", description="이름")
    extra = W.Text(value="", placeholder="예: J-pop band", description="장르 더하기")
    count = W.BoundedIntText(value=DEFAULT_COUNT, min=1, max=MAX_COUNT, description="곡 수")
    push = W.Checkbox(value=True, description="통과한 곡 저장소에 올리기")
    go = W.Button(description="비슷한 곡 만들기", button_style="success", icon="play", layout=W.Layout(width="220px"))
    status = W.HTML("<i>아직 올린 곡이 없음</i>")
    out = W.Output(layout=W.Layout(border="1px solid #cfd6e2", max_height="560px", overflow="auto", padding="6px"))
    note = W.HTML("<b>① 곡 올리기</b>(mp3 권장) → <b>② 설정</b> → <b>③ 비슷한 곡 만들기</b>. "
                  "원곡은 분석만 하고 글(프롬프트)로 넘기므로 멜로디는 새로 쓴다. 원곡과 멜로디를 비교해 <b>통과한 곡만</b> 저장소에 올린다. "
                  "원곡 파일은 저장소·Kaggle 출력에 남기지 않는다. 곡당 수십 분 걸릴 수 있다.")
    controls = (up, name, extra, count, push, go)

    def on_upload(_change):
        files = uploaded_files(up.value)
        if files:
            fname, data = files[0]
            status.value = f"올린 곡: <b>{safe_filename(fname)}</b> ({len(data) / 1e6:.1f}MB)"

    up.observe(on_upload, names="value")

    def on_go(_):
        for w in controls:
            w.disabled = True
        try:
            with out:
                files = uploaded_files(up.value)
                run_name = name.value.strip() or f"ref-{datetime.now():%Y%m%d-%H%M%S}"
                error = check_inputs(files, run_name, count.value, push.value, bool(os.environ.get("GITHUB_TOKEN")))
                if error:
                    print(f"[PANEL] {error}")
                    return
                ref = save_upload(*files[0])
                print(f"[PANEL] 원곡 저장 {ref} ({ref.stat().st_size / 1e6:.1f}MB) — 이름 {run_name}")
                rc = run_command(build_command(ref, run_name, count.value, extra.value, push.value), write=print)
                rows = results_rows(km.OUT / run_name)
                if rows:
                    print(f"[PANEL] 결과 {len(rows)}곡 — /kaggle/working/music/{run_name}/" + ("" if rc == 0 else " (일부 실패)"))
                for seed, seconds, verdict, mp3 in rows:
                    length = f"{int(seconds // 60)}:{int(seconds % 60):02d}" if seconds else "-"
                    print(f"  {seed}  {length}  멜로디 검사: {verdict}")
                    if mp3:
                        display(Audio(filename=str(mp3)))
                prompt = km.OUT / run_name / "prompt.txt"
                if prompt.is_file():
                    print(f"[PANEL] 프롬프트: {prompt.read_text(encoding='utf-8').strip()}")
        finally:
            for w in controls:
                w.disabled = False

    go.on_click(on_go)
    display(W.VBox([note, W.HBox([up, status]), W.HBox([name, extra]), W.HBox([count, push]), go, out]))
    return 0


if __name__ == "__main__":
    print(__doc__)
