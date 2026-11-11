#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VERSION: v1.1 — 2026-11-11 — 사용자 지시 "어쿠스틱 기타(단일 멜로디, 코드 진행 각각), 일렉트릭 기타(단일 멜로디, 코드 진행, 클린톤 등
#          톤별 각각), 그랜드 피아노, 드럼, 베이스, 신디사이저 샘플 좀 몇 초 들려주는 코드": 음색 견본 표 SAMPLES·sample_notes, 초 단위
#          렌더 render_notes (render 는 이것을 부름 — 결과 같음)
# (v1.0) — 2026-10-24 — 사용자 지시 "내가 멜로디 한 마디를 찍고 악기를 선택하면 그걸 기반으로 베이스, 드럼, 일렉기타, 피아노 등도
#          한 마디씩 각각 한 악기의 음원만 나오게 할 수 있어? … B 방식(AI 모델)으로 해봐"
r"""
멜로디 한 마디 → AI 반주(Anticipatory Music Transformer, Stanford CRFM) — 순수 계산 부분. kaggle_accomp.py 가 쓴다.

  parse_melody_code / melody_code   멜로디 코드 ('멜로디 찍기' 웹 페이지가 만드는 글) ↔ 음표
                                    예: M1 bpm=110 steps=16 notes=69@0+2,72@2+2,76@4+4  (음@시작칸+길이칸, 한 칸 = 16분음표)
  INSTRUMENTS                       악기 표 — GM 프로그램 번호(AMT 악기 번호와 같음, 드럼 = 128), 음역, 세기, 섞을 때 크기
  loop_notes                        한 마디를 N번 반복 → 초 단위 음표 (앞에 LEAD_IN 초 — AMT 가 0.05초 전 조건을 버려서)
  notes_to_events / events_to_notes AMT 토큰(시각·길이·악기×음) ↔ 음표,  to_controls  음표 → AMT 조건
  instrument_mask                   생성 중 다음 음표를 '이 악기·이 음역'만 나오게 (로짓을 -inf)
  quantize / steps_to_seconds       16분음표 격자에 맞추기 (마디 밖·중복 음 버림)
  write_midi                        표준 MIDI 파일 (표준 라이브러리만 — 템포·박자·악기 번호·드럼 10번 채널)
  render / mix / write_wav          사운드폰트(GeneralUser GS)로 악기별 소리 (tinysoundfont), 16bit WAV
  SAMPLES / sample_notes            음색 견본 — 같은 짧은 악절(C–G–Am–F, 100 BPM 두 마디)을 악기·톤마다 (render_notes 로 렌더)

불러올 때는 표준 라이브러리만 (numpy·tinysoundfont 는 render·mix·write_wav 안에서). ※ 연구·교육용.
"""
# CHANGELOG
#   v1.0 — 2026-10-24 — 새 파일 (위 함수들). AMT 어휘 상수는 anticipation/vocab.py(커밋 af37397)와 같은 값 — kaggle_accomp.py 가
#          실행 때 실제 패키지 값과 맞는지 확인한다. 영향: tools/kaggle_accomp.py, tools/kaggle_accomp_panel.py
#   v1.1 — 2026-11-11 — 음색 견본(사용자 지시): Sample·SAMPLE_GROUPS·SAMPLES(38개 — 사운드폰트 뱅크/프리셋 번호는 GeneralUser GS
#          v2.0.3 의 프리셋 표에서 확인)·sample_notes(멜로디·스트럼·아르페지오·뮤트·펑크 커팅·파워 코드·피아노·패드·스탭·베이스·드럼)·
#          strum. 렌더를 render_notes(초 단위 음표·음마다 세기·뱅크/프리셋)로 떼어 냄 — render 의 결과·시험은 그대로.
#          영향: render, 새 render_notes·strum·sample_notes, tools/kaggle_accomp.py samples 명령
import math
import re
import struct
from dataclasses import dataclass

MODULE_VERSION = "v1.1"

# ---- AMT 어휘 (anticipation/config.py·vocab.py 와 같은 값 — 실행 때 확인)
TIME_RESOLUTION = 100  # 10ms
MAX_TIME, MAX_DUR = 100 * TIME_RESOLUTION, 10 * TIME_RESOLUTION
MAX_PITCH, MAX_INSTR = 128, 129
MAX_NOTE = MAX_PITCH * MAX_INSTR
TIME_OFFSET = 0
DUR_OFFSET = TIME_OFFSET + MAX_TIME
NOTE_OFFSET = DUR_OFFSET + MAX_DUR
REST = NOTE_OFFSET + MAX_NOTE
CONTROL_OFFSET = NOTE_OFFSET + MAX_NOTE + 1
DRUMS = 128
LEAD_IN = 0.1  # 초 — anticipation.sample.generate 가 시각 5틱(0.05초) 전 조건을 버리므로 첫 음을 살리려고 띄운다
NEG_INF = -float("inf")

STEPS_PER_BEAT = 4  # 16분음표
NOTE_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_NAME = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)$")
_SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass(frozen=True)
class Instrument:
    key: str
    name_ko: str
    program: int      # GM 프로그램 (0~127), 드럼은 128 — AMT 의 악기 번호와 같다
    low: int          # 허용 음역 (MIDI)
    high: int
    velocity: int
    gain: float       # 섞을 때 크기
    accomp: bool      # 반주 악기로 고를 수 있나
    melody: bool      # 멜로디 악기로 고를 수 있나

    @property
    def drums(self):
        return self.program == DRUMS


INSTRUMENTS = {i.key: i for i in (
    Instrument("drums", "드럼", DRUMS, 35, 81, 105, 1.0, True, False),
    Instrument("bass", "베이스", 33, 28, 60, 105, 0.9, True, False),
    Instrument("piano", "피아노", 0, 21, 108, 90, 0.8, True, True),
    Instrument("epiano", "일렉트릭 피아노", 4, 28, 103, 90, 0.8, True, True),
    Instrument("guitar_clean", "일렉기타(깨끗한)", 27, 40, 88, 95, 0.75, True, True),
    Instrument("guitar_drive", "일렉기타(드라이브)", 29, 40, 88, 95, 0.7, True, True),
    Instrument("guitar_dist", "일렉기타(디스토션)", 30, 40, 88, 95, 0.65, True, True),
    Instrument("acoustic_guitar", "통기타", 25, 40, 88, 95, 0.75, True, True),
    Instrument("strings", "스트링", 48, 36, 96, 80, 0.55, True, True),
    Instrument("synth_pad", "신스 패드", 89, 36, 96, 80, 0.5, True, False),
    Instrument("brass", "브라스", 61, 40, 84, 90, 0.6, True, True),
    Instrument("synth_lead", "신스 리드", 80, 48, 96, 95, 0.8, False, True),
    Instrument("flute", "플루트", 73, 60, 96, 95, 0.8, False, True),
    Instrument("violin", "바이올린", 40, 55, 100, 95, 0.8, False, True),
    Instrument("sax", "알토 색소폰", 65, 49, 81, 95, 0.8, False, True),
    Instrument("trumpet", "트럼펫", 56, 52, 84, 95, 0.8, False, True),
    Instrument("voice", "코러스 '아'", 52, 48, 84, 95, 0.8, False, True),
)}
# 앞에서 만든 악기를 조건으로 다음 악기를 만든다 — 리듬(드럼·베이스) 먼저, 화음 악기, 그다음 채우는 악기
GENERATION_ORDER = ("drums", "bass", "piano", "epiano", "acoustic_guitar", "guitar_clean", "guitar_drive", "guitar_dist",
                    "strings", "synth_pad", "brass")


# ------------------------------------------------------------------ 멜로디 코드
def note_number(token):
    """'69' · 'A4' · 'C#5' · 'Eb3' → MIDI 번호."""
    token = token.strip()
    if re.fullmatch(r"\d{1,3}", token):
        n = int(token)
    else:
        m = _NAME.match(token)
        if not m:
            raise ValueError(f"음 이름을 모름: {token!r} (예: 69, A4, C#5, Eb3)")
        letter, acc, octave = m.groups()
        n = 12 * (int(octave) + 1) + _SEMI[letter.upper()] + (1 if acc == "#" else -1 if acc == "b" else 0)
    if not 0 <= n <= 127:
        raise ValueError(f"음 범위(0~127) 밖: {token!r}")
    return n


def note_label(n):
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


def parse_melody_code(text):
    """멜로디 코드 → {bpm, steps, notes: [(음, 시작 칸, 길이 칸)]} (시작 칸 순). 틀리면 ValueError."""
    text = (text or "").strip()
    if not text:
        raise ValueError("멜로디 코드가 비었음")
    fields = dict(re.findall(r"(\w+)\s*=\s*([^\s]+)", text))
    fields = {k.lower(): v for k, v in fields.items()}
    try:
        bpm = float(fields.get("bpm", "100"))
        steps = int(fields.get("steps", "16"))
    except ValueError as e:
        raise ValueError(f"bpm·steps 는 숫자: {e}") from None
    if not 40 <= bpm <= 240:
        raise ValueError(f"bpm 은 40~240: {bpm}")
    if steps % STEPS_PER_BEAT or not 4 <= steps <= 64:
        raise ValueError(f"steps 는 4의 배수, 4~64 (16 = 4/4 한 마디): {steps}")
    raw = fields.get("notes", "")
    notes = []
    for part in [p for p in raw.split(",") if p.strip()]:
        m = re.fullmatch(r"\s*([^@\s]+)@(\d+)\+(\d+)\s*", part)
        if not m:
            raise ValueError(f"음표 모양은 음@시작+길이 (예: 69@0+2): {part!r}")
        pitch, start, length = note_number(m.group(1)), int(m.group(2)), int(m.group(3))
        if length < 1 or start < 0 or start + length > steps:
            raise ValueError(f"마디 밖 음표: {part!r} (칸 0~{steps - 1})")
        notes.append((pitch, start, length))
    if not notes:
        raise ValueError("음표가 없음 (notes=69@0+2,…)")
    notes.sort(key=lambda n: (n[1], -n[0]))
    for a, b in zip(notes, notes[1:]):
        if b[1] < a[1] + a[2]:
            raise ValueError(f"겹치는 음표 — 멜로디는 한 번에 한 음: {note_label(a[0])}@{a[1]} 와 {note_label(b[0])}@{b[1]}")
    return dict(bpm=bpm, steps=steps, notes=notes)


def melody_code(melody):
    """parse_melody_code 의 반대."""
    bpm = melody["bpm"]
    bpm_text = str(int(bpm)) if float(bpm).is_integer() else str(bpm)
    notes = ",".join(f"{p}@{s}+{n}" for p, s, n in melody["notes"])
    return f"M1 bpm={bpm_text} steps={melody['steps']} notes={notes}"


# ------------------------------------------------------------------ 시간·AMT 토큰
def step_seconds(bpm):
    return 60.0 / bpm / STEPS_PER_BEAT


def loop_notes(melody, loops, lead_in=LEAD_IN):
    """멜로디 마디를 loops 번 반복 → [(시작 초, 길이 초, 음)]."""
    st = step_seconds(melody["bpm"])
    out = []
    for k in range(loops):
        for pitch, start, length in melody["notes"]:
            out.append((lead_in + (k * melody["steps"] + start) * st, length * st, pitch))
    return out


def notes_to_events(notes, instr):
    """[(시작 초, 길이 초, 음)] → AMT 사건 토큰 [시각, 길이, 악기×음, …] (시각순)."""
    tokens = []
    for start, dur, pitch in sorted(notes):
        t = int(round(start * TIME_RESOLUTION))
        d = min(MAX_DUR - 1, max(1, int(round(dur * TIME_RESOLUTION))))
        if not 0 <= t < MAX_TIME:
            raise ValueError(f"AMT 는 {MAX_TIME // TIME_RESOLUTION}초까지: {start:.2f}초")
        tokens += [TIME_OFFSET + t, DUR_OFFSET + d, NOTE_OFFSET + instr * MAX_PITCH + pitch]
    return tokens


def to_controls(events):
    return [tok + CONTROL_OFFSET for tok in events]


def events_to_notes(events):
    """AMT 사건 → [(시작 초, 길이 초, 음, 악기)] (쉼 토큰 REST 와 조건 토큰은 뺀다)."""
    out = []
    for time, dur, note in zip(events[0::3], events[1::3], events[2::3]):
        if not NOTE_OFFSET <= note < NOTE_OFFSET + MAX_NOTE:
            continue
        n = note - NOTE_OFFSET
        out.append(((time - TIME_OFFSET) / TIME_RESOLUTION, (dur - DUR_OFFSET) / TIME_RESOLUTION, n % MAX_PITCH, n // MAX_PITCH))
    return out


def instrument_mask(logits, instr, low=0, high=MAX_PITCH - 1):
    """1차원 로짓(torch·numpy)에서 음표 토큰을 이 악기·이 음역만 남기고 -inf. 쉼 토큰(REST)은 그대로."""
    a = NOTE_OFFSET + instr * MAX_PITCH + low
    b = NOTE_OFFSET + instr * MAX_PITCH + high + 1
    keep = logits[a:b]
    keep = keep.clone() if hasattr(keep, "clone") else keep.copy()
    logits[NOTE_OFFSET:NOTE_OFFSET + MAX_NOTE] = NEG_INF
    logits[a:b] = keep
    return logits


# ------------------------------------------------------------------ 격자
def quantize(notes, bpm, total_steps, lead_in=LEAD_IN, drums=False):
    """[(시작 초, 길이 초, 음, …)] → [(시작 칸, 길이 칸, 음)] — 16분음표 격자, 마디 밖은 버림, 같은 칸·같은 음은 하나."""
    st = step_seconds(bpm)
    seen, out = set(), []
    for note in notes:
        start, dur, pitch = note[0], note[1], note[2]
        s = int(round((start - lead_in) / st))
        if not 0 <= s < total_steps or (s, pitch) in seen:
            continue
        length = 1 if drums else max(1, int(round(dur / st)))
        out.append((s, min(length, total_steps - s), pitch))
        seen.add((s, pitch))
    return sorted(out)


def steps_to_seconds(notes_steps, bpm):
    st = step_seconds(bpm)
    return [(s * st, n * st, p) for s, n, p in notes_steps]


# ------------------------------------------------------------------ MIDI (표준 라이브러리)
TICKS_PER_BEAT = 480


def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def _track(events):
    """[(절대 틱, 메시지 bytes)] → MTrk 덩어리."""
    data, last = b"", 0
    for tick, msg in sorted(events, key=lambda e: (e[0], (e[1][0] & 0xF0) != 0x80)):  # 같은 틱이면 note-off 먼저 (안정 정렬)
        data += _vlq(tick - last) + msg
        last = tick
    data += _vlq(0) + b"\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(data)) + data


def write_midi(path, parts, bpm, steps_per_bar=16):
    """parts: [(Instrument, [(시작 칸, 길이 칸, 음)])] → 형식 1 MIDI. 드럼은 10번 채널(9), 나머지는 차례로."""
    tpq = TICKS_PER_BEAT
    tick = tpq // STEPS_PER_BEAT
    tempo = int(round(60_000_000 / bpm))
    beats_per_bar = steps_per_bar // STEPS_PER_BEAT
    head = [(0, b"\xff\x51\x03" + tempo.to_bytes(3, "big")), (0, bytes([0xFF, 0x58, 4, beats_per_bar, 2, 24, 8]))]
    chunks = [_track(head)]
    channel = 0
    for inst, notes in parts:
        ch = 9 if inst.drums else channel
        if not inst.drums:
            channel = channel + 1 if channel + 1 != 9 else 10
        name = inst.key.encode("ascii")
        ev = [(0, b"\xff\x03" + _vlq(len(name)) + name)]
        if not inst.drums:
            ev.append((0, bytes([0xC0 | ch, inst.program])))
        for s, n, p in notes:
            ev.append((s * tick, bytes([0x90 | ch, p, inst.velocity])))
            ev.append(((s + n) * tick, bytes([0x80 | ch, p, 0])))
        chunks.append(_track(ev))
    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), tpq)
    with open(path, "wb") as f:
        f.write(header + b"".join(chunks))
    return path


# ------------------------------------------------------------------ 소리 (numpy·tinysoundfont)
def render_notes(notes, sf2, bank=0, preset=0, drums=False, seconds=None, sr=44100, tail=1.5, synth_factory=None):
    """초 단위 음표 [(시작 초, 길이 초, 음, 세기)] 를 사운드폰트 한 소리(뱅크·프리셋)로 → float32 (N, 2).
    길이 = seconds(없으면 마지막 음 끝) + 꼬리 tail 초. 드럼은 10번 채널·뱅크 128, preset = 드럼 세트."""
    import numpy as np
    if synth_factory is None:
        import tinysoundfont
        synth_factory = tinysoundfont.Synth
    synth = synth_factory(samplerate=sr, gain=0)
    sfid = synth.sfload(str(sf2))
    ch = 9 if drums else 0
    synth.program_select(ch, sfid, 128 if drums else bank, preset, is_drums=drums)
    events = []
    for start, dur, p, vel in notes:
        events.append((start, 1, p, vel))
        events.append((start + dur, 0, p, 0))
    if seconds is None:
        seconds = max((e[0] for e in events), default=0.0)
    chunks, t = [], 0
    for when, on, p, vel in sorted(events, key=lambda e: (e[0], e[1])) + [(seconds + tail, -1, 0, 0)]:
        n = int(round(when * sr)) - t
        if n > 0:
            chunks.append(np.frombuffer(synth.generate(n), dtype=np.float32).reshape(-1, 2).copy())
            t += n
        if on == 1:
            synth.noteon(ch, p, vel)
        elif on == 0:
            synth.noteoff(ch, p)
    return np.concatenate(chunks) if chunks else np.zeros((0, 2), dtype=np.float32)


def render(notes_steps, inst, bpm, total_steps, sf2, sr=44100, tail=1.5, synth_factory=None):
    """한 악기만 → float32 (N, 2). 길이 = 격자 total_steps + 꼬리 tail 초."""
    notes = [(s, n, p, inst.velocity) for s, n, p in steps_to_seconds(notes_steps, bpm)]
    return render_notes(notes, sf2, 0, 0 if inst.drums else inst.program, inst.drums, total_steps * step_seconds(bpm), sr, tail,
                        synth_factory)


# ------------------------------------------------------------------ 음색 견본 (같은 악절을 악기·톤마다)
@dataclass(frozen=True)
class Sample:
    key: str
    group: str        # SAMPLE_GROUPS 키
    name_ko: str
    bank: int         # 사운드폰트 뱅크 (드럼은 128 — preset 이 드럼 세트)
    preset: int
    part: str         # sample_notes 의 악절 종류
    velocity: int = 95
    transpose: int = 0
    uses: str = ""    # 반주 도구(INSTRUMENTS)에서 같은 소리인 악기 키 — 없으면 아직 못 고름

    @property
    def drums(self):
        return self.bank == 128


SAMPLE_GROUPS = {"acoustic": "어쿠스틱 기타", "electric": "일렉트릭 기타", "piano": "그랜드 피아노", "drums": "드럼",
                 "bass": "베이스", "synth": "신디사이저"}
SAMPLE_BPM = 100
SAMPLE_STEPS = 32  # 두 마디 — 100 BPM 에서 4.8초 (+ 울림)
SAMPLES = (
    Sample("acoustic_steel_melody", "acoustic", "스틸 통기타 — 멜로디", 0, 25, "melody", 100, -12, "acoustic_guitar"),
    Sample("acoustic_steel_chords", "acoustic", "스틸 통기타 — 코드 진행 (스트럼)", 0, 25, "strum", 95, 0, "acoustic_guitar"),
    Sample("acoustic_nylon_melody", "acoustic", "나일론 기타 — 멜로디", 0, 24, "melody", 100, -12),
    Sample("acoustic_nylon_chords", "acoustic", "나일론 기타 — 코드 진행 (아르페지오)", 0, 24, "arpeggio", 95),
    Sample("acoustic_12string_chords", "acoustic", "12현 기타 — 코드 진행 (스트럼)", 8, 25, "strum", 95),
    Sample("electric_clean_melody", "electric", "일렉 클린 — 멜로디", 0, 27, "melody", 100, -12, "guitar_clean"),
    Sample("electric_clean_chords", "electric", "일렉 클린 — 코드 진행 (스트럼)", 0, 27, "strum", 95, 0, "guitar_clean"),
    Sample("electric_chorus_chords", "electric", "일렉 클린 + 코러스 — 코드 진행", 8, 27, "strum", 95),
    Sample("electric_jazz_melody", "electric", "재즈 기타 — 멜로디", 0, 26, "melody", 100, -12),
    Sample("electric_jazz_chords", "electric", "재즈 기타 — 코드 진행", 0, 26, "strum", 90),
    Sample("electric_muted_chords", "electric", "뮤트 — 코드 진행 (8분 뮤트)", 0, 28, "mute", 100),
    Sample("electric_funk_chords", "electric", "펑크 기타 — 코드 진행 (16분 커팅)", 8, 28, "funk", 100),
    Sample("electric_overdrive_melody", "electric", "오버드라이브 — 멜로디", 0, 29, "melody", 100, -12, "guitar_drive"),
    Sample("electric_overdrive_chords", "electric", "오버드라이브 — 코드 진행 (파워 코드)", 0, 29, "power", 100, 0, "guitar_drive"),
    Sample("electric_distortion_melody", "electric", "디스토션 — 멜로디", 0, 30, "melody", 100, -12, "guitar_dist"),
    Sample("electric_distortion_chords", "electric", "디스토션 — 코드 진행 (파워 코드)", 0, 30, "power", 100, 0, "guitar_dist"),
    Sample("piano_grand_melody", "piano", "그랜드 피아노 — 멜로디", 0, 0, "melody", 90, 0, "piano"),
    Sample("piano_grand_chords", "piano", "그랜드 피아노 — 코드 진행", 0, 0, "piano_chords", 85, 0, "piano"),
    Sample("piano_grand_both", "piano", "그랜드 피아노 — 멜로디 + 코드 (두 손)", 0, 0, "piano_both", 88, 0, "piano"),
    Sample("drums_standard", "drums", "드럼 — 스탠더드", 128, 0, "beat", 100, 0, "drums"),
    Sample("drums_room", "drums", "드럼 — 룸", 128, 8, "beat"),
    Sample("drums_power", "drums", "드럼 — 파워 (록)", 128, 16, "beat"),
    Sample("drums_electronic", "drums", "드럼 — 일렉트로닉", 128, 24, "beat"),
    Sample("drums_808", "drums", "드럼 — 808/909", 128, 25, "beat"),
    Sample("drums_jazz", "drums", "드럼 — 재즈", 128, 32, "beat"),
    Sample("drums_brush", "drums", "드럼 — 브러시", 128, 40, "beat"),
    Sample("bass_finger", "bass", "핑거 베이스", 0, 33, "bass", 100, 0, "bass"),
    Sample("bass_pick", "bass", "피크 베이스", 0, 34, "bass", 100),
    Sample("bass_fretless", "bass", "프렛리스 베이스", 0, 35, "bass", 100),
    Sample("bass_slap", "bass", "슬랩 베이스", 0, 36, "bass", 100),
    Sample("bass_synth", "bass", "신스 베이스 1", 0, 38, "bass", 100),
    Sample("bass_synth101", "bass", "신스 베이스 101", 1, 38, "bass", 100),
    Sample("synth_square_lead", "synth", "스퀘어 리드 — 멜로디", 0, 80, "melody", 95, 0, "synth_lead"),
    Sample("synth_saw_lead", "synth", "소우 리드 — 멜로디", 0, 81, "melody", 95),
    Sample("synth_polysynth", "synth", "폴리신스 — 코드 진행 (스탭)", 0, 90, "stabs", 90),
    Sample("synth_saw_stack", "synth", "브라이트 소우 스택 — 코드 진행 (스탭)", 11, 100, "stabs", 90),
    Sample("synth_warm_pad", "synth", "웜 패드 — 코드 진행", 0, 89, "pad", 85, 0, "synth_pad"),
    Sample("synth_halo_pad", "synth", "헤일로 패드 — 코드 진행", 0, 94, "pad", 85),
)

# C – G – Am – F, 코드 하나에 반 마디(8칸)
_PROGRESSION = ("C", "G", "Am", "F")
_GUITAR = {"C": (48, 52, 55, 60, 64), "G": (43, 47, 50, 55, 59, 67), "Am": (45, 52, 57, 60, 64), "F": (41, 48, 53, 57, 60, 65)}  # 개방현 모양
_POWER = {"C": (48, 55, 60), "G": (43, 50, 55), "Am": (45, 52, 57), "F": (41, 48, 53)}  # 근음 + 5도 + 옥타브
_PIANO_RH = {"C": (64, 67, 72), "G": (62, 67, 71), "Am": (64, 69, 72), "F": (65, 69, 72)}
_PAD = {"C": (60, 64, 67), "G": (59, 62, 67), "Am": (57, 60, 64), "F": (57, 60, 65)}
_ROOT = {"C": 36, "G": 31, "Am": 33, "F": 29}  # 베이스 근음 (C2·G1·A1·F1)
_MELODY = ((76, 0, 2), (74, 2, 2), (72, 4, 2), (74, 6, 2), (76, 8, 4), (79, 12, 4), (81, 16, 2), (79, 18, 2), (76, 20, 2),
           (74, 22, 2), (72, 24, 8))  # (음, 시작 칸, 길이 칸) — E5 D5 C5 D5 E5 G5 A5 G5 E5 D5 C5
_KICK, _SNARE, _HAT_C, _HAT_O, _CRASH = 36, 38, 42, 46, 49
SAMPLE_PARTS = ("melody", "strum", "arpeggio", "mute", "funk", "power", "piano_chords", "piano_both", "pad", "stabs", "bass", "beat")


def strum(start, dur, pitches, velocity, up=False, spread=0.012):
    """기타 스트럼 — 줄마다 spread 초씩 늦게 (다운 = 낮은 줄부터, 업 = 높은 줄부터·윗줄 4개만·조금 약하게)."""
    order = list(reversed(pitches))[:4] if up else list(pitches)
    vel = max(1, velocity - 15) if up else velocity
    return [(start + i * spread, max(0.02, dur - i * spread), p, vel) for i, p in enumerate(order)]


def sample_notes(sample, bpm=SAMPLE_BPM):
    """견본 하나 → 초 단위 음표 [(시작 초, 길이 초, 음, 세기)] (두 마디, 시작 칸 순)."""
    st, v, tr = step_seconds(bpm), sample.velocity, sample.transpose
    out = []

    def add(step, length, pitch, vel=v):
        out.append((step * st, length * st, pitch, vel))

    chords = [(c, i * 8) for i, c in enumerate(_PROGRESSION)]
    part = sample.part
    if part == "melody":
        for p, s, n in _MELODY:
            add(s, n * 0.95, p)
    elif part == "strum":  # 다운(박) · 다운 · 업
        for c, s in chords:
            for at, length, up, dv in ((0, 4, False, 0), (4, 2, False, -8), (6, 2, True, 0)):
                out += strum((s + at) * st, length * st, _GUITAR[c], v + dv, up)
    elif part == "arpeggio":  # 근음 → 윗줄 차례로 8분음표, 코드 끝까지 울림
        for c, s in chords:
            g = _GUITAR[c]
            for k, p in enumerate((g[0], g[2], g[3], g[-1])):
                add(s + 2 * k, 8 - 2 * k, p, v if k == 0 else v - 10)
    elif part == "mute":  # 낮은 세 줄 8분 뮤트 (짧게)
        for c, s in chords:
            for k in range(4):
                out += strum((s + 2 * k) * st, 0.6 * st, _GUITAR[c][:3], v if k == 0 else v - 12, spread=0.006)
    elif part == "funk":  # 윗줄 세 개 16분 커팅
        for c, s in chords:
            for at, dv in ((0, 0), (2, -15), (3, -5), (5, -15), (6, -5)):
                out += strum((s + at) * st, 0.45 * st, _GUITAR[c][-3:], v + dv, up=at % 2 == 1, spread=0.005)
    elif part == "power":  # 파워 코드 8분 (첫 박 강하게)
        for c, s in chords:
            for k in range(4):
                out += strum((s + 2 * k) * st, 1.8 * st, _POWER[c], v if k == 0 else v - 10, spread=0.006)
    elif part in ("piano_chords", "piano_both"):
        for c, s in chords:
            add(s, 8, _ROOT[c] + 12, v)       # 왼손 근음 옥타브
            add(s, 8, _ROOT[c] + 24, v - 10)
            if part == "piano_chords":
                for at in (0, 4):              # 오른손 화음 4분음표 두 번
                    for p in _PIANO_RH[c]:
                        add(s + at, 3.8, p, v - 5 if at else v)
            else:
                add(s + 4, 4, _ROOT[c] + 31, v - 15)  # 왼손 5도
        if part == "piano_both":
            for p, s, n in _MELODY:
                add(s, n * 0.95, p, v)
    elif part == "pad":
        for c, s in chords:
            for p in _PAD[c]:
                add(s, 8, p)
    elif part == "stabs":  # 박 · 엇박 · 엇박
        for c, s in chords:
            for at in (0, 3, 6):
                for p in _PAD[c]:
                    add(s + at, 1.5, p, v if at == 0 else v - 10)
    elif part == "bass":  # 근음 · 근음 · 5도 · 옥타브 8분
        for c, s in chords:
            for k, iv in enumerate((0, 0, 7, 12)):
                add(s + 2 * k, 1.8, _ROOT[c] + iv, v if k == 0 else v - 10)
    elif part == "beat":
        for s in range(0, SAMPLE_STEPS, 2):
            if s != 30:
                add(s, 1, _HAT_C, 85 if s % 4 == 0 else 70)
        for s in (0, 6, 8, 16, 22, 24, 26):
            add(s, 1, _KICK, 110)
        for s in (4, 12, 20, 28):
            add(s, 1, _SNARE, 105)
        add(30, 2, _HAT_O, 85)
        add(0, 4, _CRASH, 95)
    else:
        raise ValueError(f"모르는 악절: {part}")
    return sorted((t, d, p + tr, vel) for t, d, p, vel in out)  # transpose 는 멜로디 견본만 (드럼·화음은 0)


def bar_slice(audio, bpm, steps_per_bar, bar_index, sr=44100):
    """bar_index 번째 마디(0부터)만 — 이어 붙여 반복할 때 쓰는 한 마디 (앞 마디 꼬리 포함)."""
    st = step_seconds(bpm)
    a = int(round(bar_index * steps_per_bar * st * sr))
    b = int(round((bar_index + 1) * steps_per_bar * st * sr))
    return audio[a:b]


def common_gain(stems, peak_db=-1.0):
    """여러 트랙에 같은 크기를 곱해 가장 큰 트랙의 최고점을 peak_db 로 (트랙 사이 균형은 그대로)."""
    import numpy as np
    peak = max((float(np.abs(a).max()) for a in stems if len(a)), default=0.0)
    return 10 ** (peak_db / 20) / peak if peak > 0 else 1.0


def mix(stems, gains, peak_db=-1.0):
    """{키: (N,2)} · {키: 크기} → 합친 소리 (최고점 peak_db), 곱한 크기."""
    import numpy as np
    n = max((len(a) for a in stems.values()), default=0)
    out = np.zeros((n, 2), dtype=np.float32)
    for key, a in stems.items():
        out[:len(a)] += a * gains.get(key, 1.0)
    peak = float(np.abs(out).max()) if n else 0.0
    g = 10 ** (peak_db / 20) / peak if peak > 0 else 1.0
    return out * g, g


def write_wav(path, audio, sr=44100):
    """float (N,2) → 16bit PCM WAV (표준 라이브러리 wave)."""
    import wave
    import numpy as np
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


def describe(notes_steps):
    """로그용: 음 수·음역·쓰는 칸 수."""
    if not notes_steps:
        return dict(notes=0)
    pitches = [p for _, _, p in notes_steps]
    return dict(notes=len(notes_steps), low=note_label(min(pitches)), high=note_label(max(pitches)),
                onsets=len({s for s, _, _ in notes_steps}))


def seconds_label(x):
    return f"{int(x // 60)}:{x % 60:04.1f}" if x >= 60 else f"{x:.1f}s"


assert math.isinf(NEG_INF)
