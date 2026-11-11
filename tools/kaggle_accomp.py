#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VERSION: v1.1 — 2026-11-11 — 사용자 지시 "어쿠스틱 기타(단일 멜로디, 코드 진행 각각), 일렉트릭 기타(단일 멜로디, 코드 진행, 클린톤 등
#          톤별 각각), 그랜드 피아노, 드럼, 베이스, 신디사이저 샘플 좀 몇 초 들려주는 코드": samples 명령 (음색 견본 WAV)
# (v1.0) — 2026-10-24 — 사용자 지시 "내가 멜로디 한 마디를 찍고 악기를 선택하면 그걸 기반으로 베이스, 드럼, 일렉기타, 피아노 등도
#          한 마디씩 각각 한 악기의 음원만 나오게 할 수 있어? … B 방식(AI 모델)으로 해봐"
r"""
멜로디 한 마디 → AI 반주 → 악기마다 따로 된 소리(WAV)·MIDI.  ※ 연구·교육용. Kaggle 실제 실행은 아직 검증 전.

AI 모델: Anticipatory Music Transformer (Stanford CRFM, 코드·가중치 Apache-2.0, Lakh MIDI 로 학습) — 멜로디를 '조건'으로 주면
반주를 만든다. 처음 고른 GETMusic(Microsoft)은 가중치 내려받기 링크가 지워졌고(이슈 #203·#213·#215, 답 없음) 저장소도 2026-10-01
보관 처리돼 쓸 수 없다.

Kaggle 노트북 셀 (버튼 패널은 tools/kaggle_accomp_panel.py — README '반주 만들기'):
  !python /kaggle/working/music-ytb/tools/kaggle_accomp.py run --melody "M1 bpm=110 steps=16 notes=69@0+2,72@2+2,76@4+4" \
      --melody-instrument piano --instruments drums,bass,guitar_clean,piano --loops 4 --push

흐름 (로그 단계)
  [SETUP]  Kaggle 기본 파이썬에 anticipation(커밋 고정, 의존성 없이)·tinysoundfont 0.3.7·mido 1.3.3 + 사운드폰트 GeneralUser GS
           v2.0.3(sha256 고정, 개인·상업 음악 제작에 제한 없이 사용 가능 — 그 라이선스 문서) — 세션마다 한 번, 빠름
  [CHECK]  torch·transformers·GPU, AMT 어휘 상수가 accomp.py 와 같은지
  [MELODY] 멜로디 코드(웹 페이지 '멜로디 찍기'가 만듦) → 한 마디를 --loops 번 반복 (앞에 0.1초 — AMT 가 0.05초 전 조건을 버려서)
  [GEN]    악기마다 한 번씩: 조건 = 멜로디 + 앞에서 만든 악기들, 다음 음표는 '이 악기·이 음역'만 (anticipation.sample.instr_logits
           를 잠깐 바꿔 끼움). 순서는 드럼 → 베이스 → 피아노·기타 → 스트링·패드. 음이 너무 적으면 다른 시드로 다시(최대 3번)
  [SAVE]   16분음표 격자에 맞춘 뒤 악기마다 <악기>.wav(전체 반복)·<악기>_1bar.wav(마지막 마디 — 이어 붙여 반복용)·<악기>.mid,
           all.mid·mix.wav·mix_1bar.wav·run.json. 트랙 사이 크기 균형은 그대로 두고 가장 큰 트랙 최고점을 -1dBFS 로
  [PUSH]   --push: results/accomp/<이름>/ 에 (내 멜로디로 만든 것이라 원곡 문제 없음)

음색 견본 (AI 없이 사운드폰트 소리만 — 반주 결과도 이 소리로 난다. 버튼 셀은 kaggle_accomp_panel.samples()):
  !python /kaggle/working/music-ytb/tools/kaggle_accomp.py samples [--only acoustic,electric,piano,drums,bass,synth] [--bpm 100]
  [SAMPLES] 같은 짧은 악절(C–G–Am–F 두 마디)을 악기·톤마다 → /kaggle/working/accomp/samples/<묶음>/<견본>.wav + index.json
           (견본마다 최고점 -1dBFS 로 맞춤 — 톤끼리 크기 차이는 비교하지 않는다)

한계: 소리는 사운드폰트(실제 연주보다 단순 — 특히 일렉기타) → MIDI 를 작곡 프로그램에 불러와 음색을 바꾸면 좋아진다.
AMT 는 셈여림(세기)을 만들지 않는다 → 악기마다 같은 세기. 한 마디만으로는 화음이 여럿 가능 → 시드를 바꿔 여러 번 들어 보기.
"""
# CHANGELOG
#   v1.0 — 2026-10-24 — 새 도구: setup·check_env·read_melody(코드·글 파일·MIDI)·load_model·generate_parts(악기별 차례, 마스크,
#          재시도)·export·run(+ --push, kaggle_music_yue2.commit_push)·instruments 명령. 영향: 새 파일, tools/tests/test_kaggle_accomp.py
#   v1.1 — 2026-11-11 — samples 명령(사용자 지시): ensure_sound(tinysoundfont·numpy·사운드폰트만 — AI 설치 없이)·make_samples
#          (accomp.SAMPLES → 묶음별 WAV·index.json). load_model·generate_parts 오류를 [LOAD]·[GEN] 줄로 (v1.0 커밋 직전에 넣은 것).
#          run 은 그대로. 영향: 새 ensure_sound·make_samples·samples 명령, main
import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import accomp as ac  # noqa: E402
import kaggle_music_yue2 as km  # noqa: E402 — 로그·내려받기·저장소 올리기를 같이 쓴다

TOOL_VERSION = "v1.1"
ROOT = km.ROOT
ANTICIPATION_COMMIT = "af37397922665a0fb8d474d7988b0f3755a38d45"  # jthickstun/anticipation main 2024-03-18
PIP_NO_DEPS = (f"anticipation @ git+https://github.com/jthickstun/anticipation@{ANTICIPATION_COMMIT}",
               "tinysoundfont==0.3.7")  # tinysoundfont 는 실시간 재생용 pyaudio 를 요구하지만 파일로 쓸 때는 필요 없음
PIP_DEPS = ("mido==1.3.3", "tqdm")
SOUNDFONT_COMMIT = "684543d5e5efaef08d02be50dcda8d552478fa60"  # mrbumpy409/GeneralUser-GS — v2.0.3
SOUNDFONT_URL = f"https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/{SOUNDFONT_COMMIT}/GeneralUser-GS.sf2"
SOUNDFONT_SHA256 = "9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe"
MODELS = {"small": "stanford-crfm/music-small-800k", "medium": "stanford-crfm/music-medium-800k",
          "large": "stanford-crfm/music-large-800k"}
WORK = Path(os.environ.get("ACCOMP_WORK", "/tmp/accomp"))
SOUNDFONT = WORK / "GeneralUser-GS.sf2"
READY = WORK / "ready.json"
OUT = Path("/kaggle/working/accomp") if Path("/kaggle/working").is_dir() else ROOT / "output" / "accomp"
RESULTS_SUBDIR = Path(km.RESULTS_DIR) / "accomp"
DEFAULT_INSTRUMENTS = "drums,bass,guitar_clean,piano"
MIN_NOTES = {"drums": 4}  # 그 밖 악기는 2
TRIES = 3
MAX_LOOPS = 8
SR = 44100
log, die = km.log, km.die


# ------------------------------------------------------------------ 설치·점검 (Kaggle 기본 파이썬)
def check_env():
    """torch·transformers·GPU·AMT 어휘 → 정보 dict (없으면 멈춤)."""
    try:
        import torch
        import transformers
    except ImportError as e:
        die("CHECK", error=f"{e}", hint="Kaggle 기본 환경에 torch·transformers 가 없음 → !pip install torch transformers")
    try:
        import anticipation.config as C
        import anticipation.vocab as V
    except ImportError as e:
        die("CHECK", error=f"anticipation 없음: {e}", hint="setup 을 먼저 (run 이 알아서 함)")
    mine = dict(TIME_OFFSET=ac.TIME_OFFSET, DUR_OFFSET=ac.DUR_OFFSET, NOTE_OFFSET=ac.NOTE_OFFSET, REST=ac.REST,
                CONTROL_OFFSET=ac.CONTROL_OFFSET)
    theirs = dict(TIME_OFFSET=V.TIME_OFFSET, DUR_OFFSET=V.DUR_OFFSET, NOTE_OFFSET=V.NOTE_OFFSET, REST=V.REST,
                  CONTROL_OFFSET=V.CONTROL_OFFSET)
    if mine != theirs or (C.TIME_RESOLUTION, C.MAX_PITCH) != (ac.TIME_RESOLUTION, ac.MAX_PITCH):
        die("CHECK", error=f"AMT 어휘가 다름: {theirs} ≠ {mine}", hint="anticipation 커밋이 바뀜 → accomp.py 상수 확인")
    cuda = torch.cuda.is_available()
    info = dict(torch=torch.__version__, transformers=transformers.__version__, cuda=cuda,
                gpu=torch.cuda.get_device_name(0) if cuda else "-", anticipation=ANTICIPATION_COMMIT[:12])
    log("CHECK", **info, vocab="ok")
    if not cuda:
        log("CHECK", logging.WARNING, warning="GPU 없음 — CPU 로 하면 느림", hint="Settings → Accelerator: GPU T4 x2")
    return info


def setup(force=False):
    want = dict(pip_no_deps=list(PIP_NO_DEPS), pip=list(PIP_DEPS), soundfont=SOUNDFONT_SHA256)
    if READY.is_file() and SOUNDFONT.is_file() and not force:
        try:
            if json.loads(READY.read_text(encoding="utf-8")).get("want") == want:
                log("SETUP", status="skip", reason="이미 설치함")
                return check_env()
        except (OSError, json.JSONDecodeError):
            pass
    t0 = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    hint = "Kaggle Settings → Internet: On 확인 뒤 다시"
    km.run_cmd([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *PIP_NO_DEPS], "SETUP", hint)
    km.run_cmd([sys.executable, "-m", "pip", "install", "-q", *PIP_DEPS], "SETUP", hint)
    km.download(SOUNDFONT_URL, SOUNDFONT, SOUNDFONT_SHA256, "SETUP")
    info = check_env()
    READY.write_text(json.dumps(dict(want=want, tool=TOOL_VERSION, created=datetime.now().isoformat(timespec="seconds"))),
                     encoding="utf-8")
    log("SETUP", status="done", seconds=round(time.time() - t0, 1))
    return info


def ensure_sound():
    """견본용 — tinysoundfont·numpy·사운드폰트만 (AI·torch 없이). 이미 있으면 건너뜀."""
    import importlib
    hint = "Kaggle Settings → Internet: On 확인 뒤 다시"
    for module, spec, no_deps in (("numpy", "numpy", False), ("tinysoundfont", "tinysoundfont==0.3.7", True)):
        try:
            importlib.import_module(module)
        except ImportError:
            km.run_cmd([sys.executable, "-m", "pip", "install", "-q", *(["--no-deps"] if no_deps else []), spec], "SETUP", hint)
            importlib.invalidate_caches()
    km.download(SOUNDFONT_URL, SOUNDFONT, SOUNDFONT_SHA256, "SETUP")
    return SOUNDFONT


# ------------------------------------------------------------------ 입력
def melody_from_midi(path, steps=16):
    """MIDI 파일 → 멜로디 dict (드럼 아닌 첫 트랙, 첫 음이 있는 마디부터 steps 칸, 16분음표 격자)."""
    import mido
    mid = mido.MidiFile(str(path))
    tempo = next((m.tempo for tr in mid.tracks for m in tr if m.type == "set_tempo"), 500000)
    tick = mid.ticks_per_beat / ac.STEPS_PER_BEAT
    for track in mid.tracks:
        now, on, notes = 0, {}, []
        for m in track:
            now += m.time
            if m.type in ("note_on", "note_off") and getattr(m, "channel", 0) != 9:
                if m.type == "note_on" and m.velocity > 0:
                    on[m.note] = now
                elif m.note in on:
                    start = on.pop(m.note)
                    notes.append((m.note, int(round(start / tick)), max(1, int(round((now - start) / tick)))))
        if notes:
            first_bar = min(s for _, s, _ in notes) // steps * steps
            picked, last_end = [], -1
            for p, s, n in sorted(notes, key=lambda x: (x[1], -x[0])):
                s -= first_bar
                if 0 <= s < steps and s >= last_end:  # 한 번에 한 음 (높은 음 먼저)
                    picked.append((p, s, min(n, steps - s)))
                    last_end = s + min(n, steps - s)
            return dict(bpm=round(mido.tempo2bpm(tempo), 2), steps=steps, notes=picked)
    raise ValueError("MIDI 에 드럼 아닌 음표가 없음")


def read_melody(args):
    if args.melody_mid:
        melody = melody_from_midi(args.melody_mid, args.steps)
    else:
        text = Path(args.melody_file).read_text(encoding="utf-8") if args.melody_file else args.melody
        melody = ac.parse_melody_code(text)
    if args.bpm:
        melody["bpm"] = float(args.bpm)
    return melody


def ordered(keys):
    return [k for k in ac.GENERATION_ORDER if k in keys]


# ------------------------------------------------------------------ AI
def load_model(name):
    import torch
    from transformers import AutoModelForCausalLM
    repo = MODELS[name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(repo).to(device).eval()
    except (OSError, RuntimeError, ValueError) as e:  # 내려받기·연결·메모리
        die("LOAD", error=f"{repo}: {str(e).splitlines()[0][:300]}",
            hint="Settings → Internet: On 확인 · GPU 메모리 부족이면 --model small (패널: AI 크기)")
    params = sum(p.numel() for p in model.parameters())
    info = dict(model=repo, revision=getattr(model.config, "_commit_hash", None), device=device,
                params_m=round(params / 1e6, 1), seconds=round(time.time() - t0, 1))
    log("LOAD", **info)
    return model, info


def sort_events(events):
    triples = sorted(zip(events[0::3], events[1::3], events[2::3]))
    return [tok for triple in triples for tok in triple]


def generate_parts(model, melody, melody_inst, keys, loops, seed, top_p, sample_module=None, seed_fn=None):
    """악기마다 차례로 AMT 반주 → {악기: {notes:[(칸,길이,음)], seed, attempts, raw_notes, seconds}}.
    조건 = 멜로디 + 앞에서 만든 악기들 (격자에 맞춘 것), 다음 음표 = 이 악기·이 음역만."""
    if sample_module is None:
        import anticipation.sample as sample_module
    if seed_fn is None:
        import torch
        seed_fn = torch.manual_seed
    bpm, steps = melody["bpm"], melody["steps"]
    total_steps = loops * steps
    t_end = ac.LEAD_IN + total_steps * ac.step_seconds(bpm) + 0.01
    cond = ac.notes_to_events(ac.loop_notes(melody, loops), melody_inst.program)
    original = sample_module.instr_logits
    parts = {}
    try:
        for i, key in enumerate(ordered(keys)):
            inst = ac.INSTRUMENTS[key]
            sample_module.instr_logits = (lambda logits, full_history, _i=inst:
                                          ac.instrument_mask(logits, _i.program, _i.low, _i.high))
            best = None
            for attempt in range(1, TRIES + 1):
                s = seed + 100 * i + attempt - 1
                seed_fn(s)
                t0 = time.time()
                events = sample_module.generate(model, 0, t_end, controls=ac.to_controls(sort_events(cond)), top_p=top_p)
                raw = [n for n in ac.events_to_notes(events) if n[3] == inst.program]
                notes = ac.quantize(raw, bpm, total_steps, drums=inst.drums)
                rec = dict(notes=notes, seed=s, attempts=attempt, raw_notes=len(raw), seconds=round(time.time() - t0, 1))
                log("GEN", inst=key, attempt=attempt, seed=s, raw_notes=len(raw), notes=len(notes), seconds=rec["seconds"],
                    **{k: v for k, v in ac.describe(notes).items() if k != "notes"})
                if best is None or len(notes) > len(best["notes"]):
                    best = rec
                if len(notes) >= MIN_NOTES.get(key, 2):
                    break
            else:
                log("GEN", logging.WARNING, inst=key, warning=f"{TRIES}번 해도 음이 적음", notes=len(best["notes"]),
                    hint="다른 --seed 로 다시, 또는 --loops 를 늘리기")
            parts[key] = best
            placed = [(ac.LEAD_IN + a, b, p) for a, b, p in ac.steps_to_seconds(best["notes"], bpm)]
            cond = cond + ac.notes_to_events(placed, inst.program)
    finally:
        sample_module.instr_logits = original
    return parts


# ------------------------------------------------------------------ 내보내기
def export(out_dir, melody, melody_inst, parts, loops, sf2=None, sr=SR):
    """악기별 WAV(전체·마지막 한 마디)·MIDI, all.mid·mix.wav → 파일 목록."""
    bpm, steps = melody["bpm"], melody["steps"]
    total = loops * steps
    mel_steps = sorted((k * steps + s, n, p) for k in range(loops) for p, s, n in melody["notes"])
    tracks = [("melody", melody_inst, mel_steps)] + [(k, ac.INSTRUMENTS[k], parts[k]["notes"]) for k in parts]
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = {key: ac.render(notes, inst, bpm, total, sf2 or SOUNDFONT, sr=sr) for key, inst, notes in tracks}
    g = ac.common_gain(list(audio.values()))
    files = []
    for key, inst, notes in tracks:
        a = audio[key] * g
        ac.write_wav(out_dir / f"{key}.wav", a, sr)
        ac.write_wav(out_dir / f"{key}_1bar.wav", ac.bar_slice(a, bpm, steps, loops - 1, sr), sr)
        ac.write_midi(out_dir / f"{key}.mid", [(inst, notes)], bpm, steps)
        files += [f"{key}.wav", f"{key}_1bar.wav", f"{key}.mid"]
    ac.write_midi(out_dir / "all.mid", [(inst, notes) for _, inst, notes in tracks], bpm, steps)
    mixed, mg = ac.mix({k: audio[k] * g for k in audio}, {k: inst.gain for k, inst, _ in tracks})
    ac.write_wav(out_dir / "mix.wav", mixed, sr)
    ac.write_wav(out_dir / "mix_1bar.wav", ac.bar_slice(mixed, bpm, steps, loops - 1, sr), sr)
    files += ["all.mid", "mix.wav", "mix_1bar.wav"]
    log("SAVE", out=out_dir, tracks=",".join(k for k, _, _ in tracks), stem_gain=round(g, 3), mix_gain=round(mg, 3),
        files=len(files))
    return files


def results_readme(name, record):
    lines = [f"# {name}", "",
             f"`tools/kaggle_accomp.py` {TOOL_VERSION} — 멜로디 한 마디 → AI 반주 (Anticipatory Music Transformer `{record['model']['model']}`), "
             f"{record['created'][:10]}. 소리는 사운드폰트 GeneralUser GS v2.0.3 (개인·상업 음악 제작에 제한 없이 사용 가능). 연구·교육용.", "",
             "```", record["melody_code"], "```", "",
             f"- 템포 {record['bpm']} BPM · 반복 {record['loops']}마디 · 멜로디 악기 {record['melody_instrument']} · 시드 {record['seed']} · top_p {record['top_p']}", "",
             "| 악기 | 음 수 | 음역 | 한 마디 | 전체 | MIDI |", "|---|---|---|---|---|---|"]
    for key, part in [("melody", record["melody"])] + list(record["parts"].items()):
        name_ko = ac.INSTRUMENTS[record["melody_instrument"]].name_ko + " (멜로디)" if key == "melody" else ac.INSTRUMENTS[key].name_ko
        rng = f"{part.get('low', '-')}~{part.get('high', '-')}" if part.get("notes") else "-"
        lines.append(f"| {name_ko} | {part.get('notes', 0)} | {rng} | [wav]({key}_1bar.wav) | [wav]({key}.wav) | [mid]({key}.mid) |")
    lines += ["", "합친 소리: [mix_1bar.wav](mix_1bar.wav) · [mix.wav](mix.wav) · 모든 악기 MIDI [all.mid](all.mid)", ""]
    return "\n".join(lines)


def push(out_dir, record):
    dest = RESULTS_SUBDIR / out_dir.name
    (ROOT / dest).mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(out_dir.iterdir()):
        if f.suffix in (".wav", ".mid", ".json") and f.stat().st_size / 1e6 <= km.MAX_PUSH_MB:
            shutil.copy2(f, ROOT / dest / f.name)
            count += 1
    (ROOT / dest / "README.md").write_text(results_readme(out_dir.name, record), encoding="utf-8")
    log("PUSH", path=dest, files=count + 1)
    return km.commit_push(dest, f"반주: {out_dir.name} — {', '.join(record['parts'])} (kaggle_accomp {TOOL_VERSION})",
                          retry_hint=f"결과는 {out_dir} 에 있음", parts=len(record["parts"]))


def make_samples(out_dir, groups=None, bpm=ac.SAMPLE_BPM, sf2=None, sr=SR, synth_factory=None):
    """accomp.SAMPLES → out_dir/<묶음>/<견본>.wav (견본마다 최고점 -1dBFS) + index.json → index (list of dict)."""
    t0 = time.time()
    picked = [x for x in ac.SAMPLES if not groups or x.group in groups]
    out_dir = Path(out_dir)
    index = []
    for x in picked:
        notes = ac.sample_notes(x, bpm)
        audio = ac.render_notes(notes, sf2 or SOUNDFONT, x.bank, x.preset, x.drums, ac.SAMPLE_STEPS * ac.step_seconds(bpm),
                                sr=sr, tail=1.2, synth_factory=synth_factory)
        peak = float(abs(audio).max()) if len(audio) else 0.0
        if peak <= 1e-4:
            log("SAMPLES", logging.WARNING, sample=x.key, warning="소리 없음", hint="사운드폰트에 이 뱅크·프리셋이 없음 — 표 확인")
            continue
        path = out_dir / x.group / f"{x.key}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        ac.write_wav(path, audio * ac.common_gain([audio]), sr)
        rec = dict(key=x.key, group=x.group, group_ko=ac.SAMPLE_GROUPS[x.group], name=x.name_ko, part=x.part,
                   soundfont=f"{x.bank}/{x.preset}", uses=x.uses, notes=len(notes), seconds=round(len(audio) / sr, 2),
                   peak_db=round(20 * math.log10(peak), 1), file=str(path.relative_to(out_dir)))
        index.append(rec)
        log("SAMPLES", sample=x.key, name=x.name_ko, sf=rec["soundfont"], notes=rec["notes"], seconds=rec["seconds"],
            peak_db=rec["peak_db"], accomp=x.uses or "-")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(json.dumps(dict(tool=TOOL_VERSION, bpm=bpm, created=datetime.now().isoformat(timespec="seconds"),
                                                         samples=index), ensure_ascii=False, indent=1), encoding="utf-8")
    log("SAMPLES", out=out_dir, count=len(index), skipped=len(picked) - len(index), seconds=round(time.time() - t0, 1))
    return index


# ------------------------------------------------------------------ 실행
def run(args):
    t0 = time.time()
    if not km.NAME_RE.fullmatch(args.name):
        die("INPUT", error=f"이름 {args.name!r}", hint="영문·숫자·_.- 만 (예: acc-01)")
    try:
        melody = read_melody(args)
    except (ValueError, OSError) as e:
        die("INPUT", error=f"멜로디: {e}", hint="웹 페이지 '멜로디 찍기'의 코드를 그대로 (M1 bpm=… steps=16 notes=…)")
    keys = [k.strip() for k in args.instruments.split(",") if k.strip()]
    bad = [k for k in keys if k not in ac.INSTRUMENTS or not ac.INSTRUMENTS[k].accomp]
    if not keys or bad:
        die("INPUT", error=f"반주 악기: {', '.join(bad) or '없음'}",
            hint="쓸 수 있는 것: " + ", ".join(k for k, i in ac.INSTRUMENTS.items() if i.accomp))
    mel_inst = ac.INSTRUMENTS.get(args.melody_instrument)
    if not mel_inst or not mel_inst.melody:
        die("INPUT", error=f"멜로디 악기: {args.melody_instrument}",
            hint="쓸 수 있는 것: " + ", ".join(k for k, i in ac.INSTRUMENTS.items() if i.melody))
    if not 1 <= args.loops <= MAX_LOOPS:
        die("INPUT", error=f"--loops {args.loops}", hint=f"1~{MAX_LOOPS}")
    seconds = ac.LEAD_IN + args.loops * melody["steps"] * ac.step_seconds(melody["bpm"])
    if seconds > 95:
        die("INPUT", error=f"길이 {seconds:.0f}초 — AMT 는 100초까지", hint="--loops 를 줄이기")
    log("MELODY", code=ac.melody_code(melody), instrument=mel_inst.key, loops=args.loops, seconds=round(seconds, 1),
        **ac.describe([(s, n, p) for p, s, n in melody["notes"]]))
    if args.push:
        km.push_check()
    env = setup(force=args.reinstall)
    model, model_info = load_model(args.model)
    try:
        parts = generate_parts(model, melody, mel_inst, keys, args.loops, args.seed, args.top_p)
    except RuntimeError as e:  # torch.cuda.OutOfMemoryError 도 RuntimeError
        die("GEN", error=str(e).splitlines()[0][:300], hint="--model small 또는 --loops 를 줄여 다시")
    out_dir = km.fresh_dir(Path(args.out) / args.name)
    files = export(out_dir, melody, mel_inst, parts, args.loops)
    mel_steps = [(s, n, p) for p, s, n in melody["notes"]]
    record = dict(tool=TOOL_VERSION, created=datetime.now().isoformat(timespec="seconds"), melody_code=ac.melody_code(melody),
                  bpm=melody["bpm"], steps=melody["steps"], loops=args.loops, seed=args.seed, top_p=args.top_p,
                  melody_instrument=mel_inst.key, melody=ac.describe(mel_steps), model=model_info, env=env,
                  soundfont=dict(url=SOUNDFONT_URL, sha256=SOUNDFONT_SHA256),
                  parts={k: dict(ac.describe(p["notes"]), seed=p["seed"], attempts=p["attempts"], raw_notes=p["raw_notes"],
                                 seconds=p["seconds"], instrument=ac.INSTRUMENTS[k].name_ko, program=ac.INSTRUMENTS[k].program)
                         for k, p in parts.items()},
                  files=files, note="연구·교육용. AMT 가중치 Apache-2.0, 사운드폰트 GeneralUser GS 라이선스 v2.0")
    (out_dir / "run.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "latest.txt").write_text(str(out_dir), encoding="utf-8")  # 버튼 패널이 결과 폴더를 찾는다
    log("DONE", out=out_dir, minutes=round((time.time() - t0) / 60, 1),
        parts=",".join(f"{k}:{len(p['notes'])}" for k, p in parts.items()))
    if args.push:
        return push(out_dir, record)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="멜로디 한 마디 → AI 반주 → 악기마다 따로 된 소리 (연구·교육용)")
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="설치(처음 한 번) → AI 반주 → 악기별 WAV·MIDI → [저장소에]")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--melody", help='멜로디 코드 (예: "M1 bpm=110 steps=16 notes=69@0+2,72@2+2")')
    src.add_argument("--melody-file", help="멜로디 코드가 든 글 파일")
    src.add_argument("--melody-mid", help="멜로디 MIDI 파일 (첫 음이 있는 마디)")
    r.add_argument("--steps", type=int, default=16, help="--melody-mid 일 때 칸 수 (16 = 4/4 한 마디)")
    r.add_argument("--bpm", type=float, help="템포를 바꿀 때")
    r.add_argument("--melody-instrument", default="piano")
    r.add_argument("--instruments", default=DEFAULT_INSTRUMENTS, help=f"쉼표로 (기본 {DEFAULT_INSTRUMENTS})")
    r.add_argument("--loops", type=int, default=4, help="한 마디를 몇 번 반복해 만들까 (반주가 자리 잡게 — 결과에 한 마디 파일도)")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--model", choices=tuple(MODELS), default="medium")
    r.add_argument("--name", default=None)
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--push", action="store_true", help=f"결과를 {km.REPO_SLUG} {RESULTS_SUBDIR}/<이름>/ 에 (GITHUB_TOKEN)")
    r.add_argument("--reinstall", action="store_true")
    r.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    s = sub.add_parser("setup", help="설치·점검만")
    s.add_argument("--reinstall", action="store_true")
    s.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    i = sub.add_parser("instruments", help="악기 목록")
    i.add_argument("--log-level", default="INFO")
    sm = sub.add_parser("samples", help="음색 견본 — 악기·톤마다 같은 짧은 악절 WAV (AI 없이 사운드폰트 소리)")
    sm.add_argument("--only", default="", help="묶음 쉼표로: " + ",".join(ac.SAMPLE_GROUPS) + " (비우면 전부)")
    sm.add_argument("--bpm", type=float, default=ac.SAMPLE_BPM)
    sm.add_argument("--out", default=str(OUT / "samples"))
    sm.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    args = ap.parse_args(argv)
    if args.command == "samples":
        km.setup_logging(args.log_level)
        groups = [g.strip() for g in args.only.split(",") if g.strip()]
        bad = [g for g in groups if g not in ac.SAMPLE_GROUPS]
        if bad or not 40 <= args.bpm <= 240:
            die("INPUT", error=f"--only {','.join(bad)}" if bad else f"--bpm {args.bpm}",
                hint="묶음: " + ",".join(ac.SAMPLE_GROUPS) + " · bpm 40~240")
        ensure_sound()
        index = make_samples(Path(args.out), groups, args.bpm)  # 같은 이름은 덮어씀 (견본은 매번 같은 소리)
        return 0 if index else 1
    if args.command == "instruments":
        for k, inst in ac.INSTRUMENTS.items():
            print(f"{k:16s} {inst.name_ko:14s} GM {inst.program:3d}  음역 {ac.note_label(inst.low)}~{ac.note_label(inst.high)}  "
                  f"{'반주' if inst.accomp else ''} {'멜로디' if inst.melody else ''}")
        return 0
    if args.command == "run":
        args.name = args.name or f"acc-{datetime.now():%Y%m%d-%H%M%S}"
        km.setup_logging(args.log_level, Path(args.out) / "accomp.log")
        return run(args)
    km.setup_logging(args.log_level)
    setup(force=args.reinstall)
    return 0


if __name__ == "__main__":
    sys.exit(main())
