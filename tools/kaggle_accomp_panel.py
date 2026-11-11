#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VERSION: v1.1 — 2026-11-11 — 사용자 지시 "어쿠스틱 기타(단일 멜로디, 코드 진행 각각), 일렉트릭 기타(…톤별 각각), 그랜드 피아노, 드럼,
#          베이스, 신디사이저 샘플 좀 몇 초 들려주는 코드": samples() — 음색 견본을 만들어 묶음별 플레이어로
# (v1.0) — 2026-10-24 — 사용자 지시 "내가 멜로디 한 마디를 찍고 악기를 선택하면 … 베이스, 드럼, 일렉기타, 피아노 등도 한 마디씩
#          각각 한 악기의 음원만 … 멜로디를 쉽게 찍을 수 있을지 모르겠어" + "B 방식(AI)으로 해봐": 노트북 버튼 패널
r"""
Kaggle 노트북 셀(README '반주 만들기')에서 버튼으로: 멜로디 코드 붙여 넣기 → 멜로디 악기·반주 악기 고르기 → [반주 만들기].

  import sys, importlib; sys.path.insert(0, "/kaggle/working/music-ytb/tools")
  import kaggle_accomp_panel; importlib.reload(kaggle_accomp_panel).panel()

  멜로디 코드 = 웹 페이지 '멜로디 찍기'(MELODY_PAGE_URL — 칸 클릭·건반·허밍 녹음 파일)의 [코드 복사]. MIDI 파일을 올려도 된다(첫 마디).
  [반주 만들기] = kaggle_accomp.py run --melody "<코드>" --melody-instrument … --instruments … --loops N --seed S [--push]
  → 끝나면 악기마다 '한 마디'·'반복' 플레이어 + 합친 소리. 결과 폴더 /kaggle/working/accomp/<이름>/ (악기별 wav·mid)

  올린 MIDI 는 /tmp/accomp/upload/ 에만. 실행 중에는 버튼이 잠긴다. 멈추려면 Kaggle 의 셀 멈춤(■) — 하위 프로세스까지 함께.
  로그는 아래 칸과 /kaggle/working/accomp/panel.log 에 함께.

음색 견본 (AI 없이 사운드폰트 소리 — GPU 필요 없음, 1분 안쪽):
  import kaggle_accomp_panel; importlib.reload(kaggle_accomp_panel).samples()          # 전부 (38개, 하나에 약 6초)
  importlib.reload(kaggle_accomp_panel).samples("electric,drums")                       # 묶음만: acoustic·electric·piano·drums·bass·synth
※ 연구·교육용 도구.
"""
# CHANGELOG
#   v1.0 — 2026-10-24 — 새 파일: melody_summary·check_inputs·build_command·latest_result·result_rows·panel.
#          실행·업로드 읽기는 kaggle_music_panel 의 run_command·uploaded_files·save_upload 를 같이 쓴다.
#          영향: 새 파일, tools/tests/test_kaggle_accomp_panel.py
#   v1.1 — 2026-11-11 — samples()·build_samples_command·sample_groups (사용자 지시 — 음색 견본 듣기). panel() 은 그대로.
import html
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import accomp as ac  # noqa: E402 — 표준 라이브러리만 불러오는 모듈 (GPU·torch 없음)
import kaggle_accomp as ka  # noqa: E402
import kaggle_music_panel as kp  # noqa: E402

PANEL_VERSION = "v1.1"
MELODY_PAGE_URL = "https://claude.ai/artifact/HinPmeUvyqN7KhBgrGYZ6D"  # 원본 web/melody_pad.html
UPLOAD_DIR = ka.WORK / "upload"  # /tmp — Kaggle 출력으로 저장되지 않는다
LOG_FILE = ka.OUT / "panel.log"
MIDI_EXT = (".mid", ".midi")
MAX_SECONDS = 95  # kaggle_accomp.run 과 같은 한도 (AMT 는 100초까지)


def one_line(code):
    return " ".join((code or "").split())


def melody_summary(code):
    """멜로디 코드 → 한 줄 요약 (틀리면 무엇이 틀렸는지)."""
    if not (code or "").strip():
        return "아직 멜로디 코드가 없음"
    try:
        m = ac.parse_melody_code(code)
    except ValueError as e:
        return f"코드 오류: {e}"
    pitches = [p for p, _, _ in m["notes"]]
    return (f"음 {len(pitches)}개 · {m['bpm']:g} BPM · {m['steps']}칸 · "
            f"{ac.note_label(min(pitches))}~{ac.note_label(max(pitches))}")


def check_inputs(code, midi_files, keys, loops, name, push, token_present):
    """입력 → 오류 문구 (문제 없으면 None). 코드가 있으면 코드, 없으면 올린 MIDI."""
    melody = None
    if (code or "").strip():
        try:
            melody = ac.parse_melody_code(code)
        except ValueError as e:
            return f"멜로디 코드: {e} — 멜로디 찍기 페이지의 [코드 복사] 그대로 붙여 넣기"
    elif midi_files:
        if len(midi_files) > 1:
            return "MIDI 는 하나만 올려 주세요."
        fname, data = midi_files[0]
        if Path(fname).suffix.lower() not in MIDI_EXT:
            return f"MIDI 파일이 아님: {fname} (.mid)"
        if not data:
            return "올린 MIDI 가 비었음 — 다시 올려 주세요."
    else:
        return f"멜로디 코드를 붙여 넣거나 MIDI 를 올려 주세요. 멜로디 찍기: {MELODY_PAGE_URL}"
    bad = [k for k in keys if k not in ac.INSTRUMENTS or not ac.INSTRUMENTS[k].accomp]
    if not keys or bad:
        return "반주 악기를 하나 이상 고르세요." if not bad else f"모르는 반주 악기: {', '.join(bad)}"
    if not isinstance(loops, int) or not 1 <= loops <= ka.MAX_LOOPS:
        return f"반복 마디 수는 1~{ka.MAX_LOOPS}"
    if melody and ac.LEAD_IN + loops * melody["steps"] * ac.step_seconds(melody["bpm"]) > MAX_SECONDS:
        return f"너무 김 ({MAX_SECONDS}초 넘음) — 반복 마디 수를 줄이세요."
    if not ka.km.NAME_RE.fullmatch(name):
        return f"이름은 영문·숫자·_.- 만: {name!r} (예: acc-01)"
    if push and not token_present:
        return ("저장소에 올리려면 GITHUB_TOKEN 이 필요 — 셀 첫 줄(UserSecretsClient)과 Kaggle Secrets 연결을 확인하거나, "
                "'저장소에 올리기'를 끄고 실행")
    return None


def build_command(melody_code=None, melody_mid=None, melody_inst="piano", keys=(), loops=4, seed=1, model="medium",
                  name="acc", push=False, python=sys.executable):
    cmd = [python, str(TOOLS / "kaggle_accomp.py"), "run"]
    cmd += ["--melody", one_line(melody_code)] if melody_code else ["--melody-mid", str(melody_mid)]
    cmd += ["--melody-instrument", melody_inst, "--instruments", ",".join(keys), "--loops", str(loops), "--seed", str(seed),
            "--model", model, "--name", name]
    if push:
        cmd.append("--push")
    return cmd


def latest_result(out_root, since):
    """kaggle_accomp.run 이 남긴 latest.txt → 이번 실행(since 이후)의 결과 폴더 (없으면 None)."""
    latest = Path(out_root) / "latest.txt"
    if not latest.is_file() or latest.stat().st_mtime < since:
        return None
    path = Path(latest.read_text(encoding="utf-8").strip())
    return path if (path / "run.json").is_file() else None


def result_rows(out_dir):
    """run.json → [(키, 이름, 음 수|None, 한 마디 wav|None, 반복 wav|None)] — 멜로디, 반주 악기들, 합친 소리 순."""
    out_dir = Path(out_dir)
    rec = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    mel = ac.INSTRUMENTS.get(rec.get("melody_instrument"))
    items = [("melody", f"{mel.name_ko if mel else '멜로디'} (멜로디)", rec.get("melody", {}).get("notes"))]
    items += [(k, p.get("instrument", k), p.get("notes")) for k, p in rec.get("parts", {}).items()]
    items.append(("mix", "합친 소리", None))
    rows = []
    for key, label, notes in items:
        one, full = out_dir / f"{key}_1bar.wav", out_dir / f"{key}.wav"
        rows.append((key, label, notes, one if one.is_file() else None, full if full.is_file() else None))
    return rows


def build_samples_command(only="", bpm=None, out_dir=None, python=sys.executable):
    cmd = [python, str(TOOLS / "kaggle_accomp.py"), "samples", "--out", str(out_dir or ka.OUT / "samples")]
    if only:
        cmd += ["--only", ",".join(g.strip() for g in str(only).split(",") if g.strip())]
    if bpm:
        cmd += ["--bpm", str(bpm)]
    return cmd


def sample_groups(index_file):
    """index.json → [(묶음 이름, [견본 dict])] (표 순서 그대로)."""
    groups = {}
    for item in json.loads(Path(index_file).read_text(encoding="utf-8"))["samples"]:
        groups.setdefault(item["group_ko"], []).append(item)
    return list(groups.items())


def samples(only="", bpm=None):
    """Kaggle·Jupyter: 음색 견본 만들기(kaggle_accomp.py samples) → 묶음별로 이름·사운드폰트 번호·플레이어."""
    try:
        from IPython.display import HTML, Audio, display
    except ImportError:
        print("[PANEL] IPython 없음 → kaggle_accomp.py samples (그 파일 머리말)")
        return 1
    out_dir = ka.OUT / "samples"
    t0 = time.time()
    rc = kp.run_command(build_samples_command(only, bpm, out_dir), write=print, log_file=LOG_FILE)
    index = out_dir / "index.json"
    if rc != 0 or not index.is_file() or index.stat().st_mtime < t0:
        print("[PANEL] 견본 없음 — 위 [ERROR]·level=error 줄 확인")
        return rc or 1
    display(HTML("<p>모두 같은 짧은 악절 (C–G–Am–F, 100 BPM 두 마디). 견본마다 크기를 맞췄다 — 톤끼리 크기는 비교하지 않기. "
                       "'반주 도구'라고 적힌 소리는 반주 만들기에서 바로 고를 수 있다.</p>"))
    for group_ko, items in sample_groups(index):
        display(HTML(f"<h3>{html.escape(group_ko)}</h3>"))
        for it in items:
            where = f"반주 도구 <code>{html.escape(it['uses'])}</code>" if it["uses"] else "반주 도구에는 아직 없음"
            display(HTML(f"<b>{html.escape(it['name'])}</b> <small>· 사운드폰트 뱅크/프리셋 {it['soundfont']} · {where}</small>"))
            display(Audio(filename=str(out_dir / it["file"])))
    return 0


def panel():
    """Kaggle·Jupyter: 멜로디 코드(또는 MIDI) → 악기 고르기 → [반주 만들기] → 악기별 플레이어."""
    try:
        import ipywidgets as W
        from IPython.display import Audio, display
    except ImportError:
        print("[PANEL] ipywidgets 없음 → 버튼 대신: kaggle_accomp.py run --melody \"…\" (그 파일 머리말)")
        return 1
    note = W.HTML(value=(
        f"<b>①</b> <a href='{MELODY_PAGE_URL}' target='_blank'>멜로디 찍기 페이지</a>에서 한 마디를 찍고 [코드 복사] → 아래 칸에 붙여 넣기 "
        "(또는 MIDI 파일 올리기) · <b>②</b> 멜로디 악기·반주 악기 고르기 · <b>③</b> [반주 만들기] — 악기마다 AI 가 차례로 만든다 "
        "(드럼 → 베이스 → 피아노·기타 → 스트링). 끝나면 악기마다 <b>한 마디</b>·<b>반복</b> 소리와 MIDI."))
    code = W.Textarea(value="", placeholder="M1 bpm=110 steps=16 notes=76@0+2,74@2+1,…  (멜로디 찍기 페이지의 [코드 복사])",
                      description="멜로디 코드", layout=W.Layout(width="98%", height="70px"))
    summary = W.HTML(value=f"<i>{melody_summary('')}</i>")
    mid_up = W.FileUpload(accept=",".join(MIDI_EXT), multiple=False, description="MIDI 올리기")
    mel_options = [(f"{i.name_ko} ({k})", k) for k, i in ac.INSTRUMENTS.items() if i.melody]
    mel_inst = W.Dropdown(options=mel_options, value="piano", description="멜로디 악기")
    defaults = set(ka.DEFAULT_INSTRUMENTS.split(","))
    boxes = {k: W.Checkbox(value=k in defaults, description=i.name_ko, indent=False, layout=W.Layout(width="150px"))
             for k, i in ac.INSTRUMENTS.items() if i.accomp}
    loops = W.BoundedIntText(value=4, min=1, max=ka.MAX_LOOPS, description="반복 마디")
    seed = W.IntText(value=1, description="시드")
    model = W.Dropdown(options=[("medium (기본)", "medium"), ("small (빠름)", "small"), ("large (느림·T4 메모리 확인)", "large")],
                       value="medium", description="AI 크기")
    name = W.Text(value="", placeholder="비우면 acc-날짜시각", description="이름")
    push = W.Checkbox(value=True, description="결과 저장소에 올리기")
    make_btn = W.Button(description="반주 만들기", button_style="success", icon="play", layout=W.Layout(width="200px"))
    out = W.Output(layout=W.Layout(border="1px solid #cfd6e2", max_height="640px", overflow="auto", padding="6px"))

    def on_code(_change):
        summary.value = f"<i>{melody_summary(code.value)}</i>"

    def on_make(_):
        make_btn.disabled = True
        try:
            with out:
                files = kp.uploaded_files(mid_up.value)
                keys = [k for k, cb in boxes.items() if cb.value]
                run_name = name.value.strip() or f"acc-{datetime.now():%Y%m%d-%H%M%S}"
                error = check_inputs(code.value, files, keys, loops.value, run_name, push.value,
                                     bool(os.environ.get("GITHUB_TOKEN")))
                if error:
                    print(f"[PANEL] {error}")
                    return
                if code.value.strip():
                    melody_code, melody_mid = code.value, None
                    if files:
                        print("[PANEL] 코드와 MIDI 가 둘 다 있음 → 코드를 씀")
                else:
                    melody_code, melody_mid = None, kp.save_upload(*files[0], dest_dir=UPLOAD_DIR)
                cmd = build_command(melody_code, melody_mid, mel_inst.value, keys, loops.value, seed.value, model.value,
                                    run_name, push.value)
                print(f"[PANEL] 반주 — 이름 {run_name}, 멜로디 {mel_inst.value}, 반주 {', '.join(keys)}, {loops.value}마디, 시드 {seed.value}")
                t0 = time.time()
                rc = kp.run_command(cmd, write=print, log_file=LOG_FILE)
                out_dir = latest_result(ka.OUT, t0)
                if not out_dir:
                    print("[PANEL] 결과 없음 — 위 [ERROR]·level=error 줄 확인")
                    return
                print(f"[PANEL] 결과 {out_dir}" + ("" if rc == 0 else " (저장소 올리기 실패 — 결과는 이 폴더에 있음)")
                      + " · 다른 반주를 들으려면 시드를 바꿔 다시")
                for key, label, notes, one, full in result_rows(out_dir):
                    print(f"  {label}" + (f" — 음 {notes}개" if notes is not None else ""))
                    for tag, path in (("한 마디", one), ("반복", full)):
                        if path:
                            print(f"    {tag}: {path.name}")
                            display(Audio(filename=str(path)))
        finally:
            make_btn.disabled = False

    code.observe(on_code, names="value")
    make_btn.on_click(on_make)
    inst_box = W.HBox(list(boxes.values()), layout=W.Layout(flex_flow="row wrap"))
    display(W.VBox([note, code, summary, mid_up, W.HBox([mel_inst, loops, seed]), W.HTML(value="<b>반주 악기</b>"), inst_box,
                    W.HBox([model, name, push]), make_btn, out]))
    return 0


if __name__ == "__main__":
    print(__doc__)
