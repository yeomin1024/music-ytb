# -*- coding: utf-8 -*-
# VERSION: v1.1 — 2026-11-11 — accomp.py v1.1: 음색 견본 표(SAMPLES)·악절(sample_notes)·초 단위 렌더(render_notes) 시험 추가
# (v1.0) — 2026-10-24 — tools/accomp.py v1.0 시험: 멜로디 코드, AMT 토큰 왕복, 악기·음역 거르기, 격자, MIDI, 렌더·섞기·WAV
#          (GPU·네트워크 없음. 실제 사운드폰트 렌더는 tinysoundfont 와 ACCOMP_TEST_SF2=<sf2 경로> 가 있을 때만)
r"""
  python3 -m unittest tools.tests.test_accomp -v     (저장소 최상위에서)
"""
import os
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
import accomp as ac  # noqa: E402

try:
    import numpy as np
except ImportError:
    np = None
try:
    import mido
except ImportError:
    mido = None
try:
    import tinysoundfont
except ImportError:
    tinysoundfont = None

CODE = "M1 bpm=120 steps=16 notes=E4@0+2,G4@2+2,A4@4+4,G4@8+2,E4@10+2,D4@12+4"


class FakeSynth:
    """tinysoundfont.Synth 대신 — 켜진 음 수만큼 일정한 값."""

    def __init__(self, samplerate=44100, gain=0):
        self.on, self.calls = set(), []

    def sfload(self, path):
        return 1

    def program_select(self, ch, sfid, bank, preset, is_drums=False):
        self.calls.append(("program", ch, bank, preset, is_drums))

    def noteon(self, ch, key, vel):
        self.on.add(key)

    def noteoff(self, ch, key):
        self.on.discard(key)

    def generate(self, n):
        return np.full(2 * n, 0.1 * len(self.on), dtype=np.float32).tobytes()


class TestMelodyCode(unittest.TestCase):
    def test_parse_and_round_trip(self):
        m = ac.parse_melody_code(CODE)
        self.assertEqual((m["bpm"], m["steps"]), (120.0, 16))
        self.assertEqual(m["notes"][:3], [(64, 0, 2), (67, 2, 2), (69, 4, 4)])
        self.assertEqual(ac.parse_melody_code(ac.melody_code(m)), m)
        self.assertEqual(ac.melody_code(m), "M1 bpm=120 steps=16 notes=64@0+2,67@2+2,69@4+4,67@8+2,64@10+2,62@12+4")

    def test_note_names(self):
        self.assertEqual([ac.note_number(t) for t in ("60", "C4", "c#4", "Eb3", "B-1", "G9")], [60, 60, 61, 51, 11, 127])
        self.assertEqual(ac.note_label(69), "A4")
        for bad in ("H4", "C10", "128", "C#"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ac.note_number(bad)

    def test_errors(self):
        cases = {"": "비었음", "M1 bpm=300 notes=60@0+1": "bpm", "M1 steps=10 notes=60@0+1": "steps",
                 "M1 notes=60@15+2": "마디 밖", "M1 notes=60@0+4,62@2+2": "겹치는", "M1 notes=60-0-1": "모양",
                 "M1 bpm=100": "음표가 없음"}
        for text, want in cases.items():
            with self.subTest(text=text):
                with self.assertRaises(ValueError) as cm:
                    ac.parse_melody_code(text)
                self.assertIn(want, str(cm.exception))


class TestTokens(unittest.TestCase):
    def test_loop_and_events_round_trip(self):
        m = ac.parse_melody_code(CODE)
        notes = ac.loop_notes(m, 2)
        self.assertEqual(len(notes), 12)
        self.assertAlmostEqual(notes[0][0], ac.LEAD_IN)
        self.assertAlmostEqual(notes[6][0], ac.LEAD_IN + 16 * ac.step_seconds(120))  # 두 번째 마디
        ev = ac.notes_to_events(notes, 73)
        self.assertEqual(len(ev), 36)
        back = ac.events_to_notes(ev + [ac.TIME_OFFSET + 5, ac.DUR_OFFSET, ac.REST])  # 쉼 토큰은 뺀다
        self.assertEqual([(round(s, 2), p, i) for s, _, p, i in back][:2], [(0.1, 64, 73), (0.35, 67, 73)])
        self.assertEqual(ac.to_controls(ev)[0], ev[0] + ac.CONTROL_OFFSET)
        with self.assertRaises(ValueError):
            ac.notes_to_events([(120.0, 0.5, 60)], 0)  # AMT 는 100초까지

    @unittest.skipIf(np is None, "numpy 없음")
    def test_instrument_mask(self):
        logits = np.zeros(ac.CONTROL_OFFSET + 10)
        ac.instrument_mask(logits, 33, 28, 60)
        notes = logits[ac.NOTE_OFFSET:ac.NOTE_OFFSET + ac.MAX_NOTE]
        allowed = np.flatnonzero(np.isfinite(notes))
        self.assertEqual(allowed.min(), 33 * 128 + 28)
        self.assertEqual(allowed.max(), 33 * 128 + 60)
        self.assertEqual(len(allowed), 33)
        self.assertTrue(np.isfinite(logits[ac.REST]))  # 쉼은 그대로
        self.assertTrue(np.isfinite(logits[:ac.NOTE_OFFSET]).all())  # 시각·길이 토큰은 건드리지 않음

    def test_quantize(self):
        st = ac.step_seconds(120)
        raw = [(ac.LEAD_IN + 0.02, 0.24, 36, 128), (ac.LEAD_IN + st * 3.9, st * 1.6, 38, 128),
               (ac.LEAD_IN + st * 4.1, st, 38, 128), (ac.LEAD_IN - 0.5, st, 40, 128), (ac.LEAD_IN + st * 40, st, 42, 128)]
        self.assertEqual(ac.quantize(raw, 120, 32, drums=True), [(0, 1, 36), (4, 1, 38)])  # 같은 칸 중복·마디 밖 버림
        self.assertEqual(ac.quantize([(ac.LEAD_IN + st * 30, st * 5, 60)], 120, 32), [(30, 2, 60)])  # 끝을 넘지 않게
        self.assertEqual(ac.steps_to_seconds([(4, 2, 60)], 120), [(4 * st, 2 * st, 60)])


class TestMidi(unittest.TestCase):
    def test_write_midi(self):
        parts = [(ac.INSTRUMENTS["piano"], [(0, 2, 60), (2, 2, 64)]), (ac.INSTRUMENTS["drums"], [(0, 1, 36), (4, 1, 38)]),
                 (ac.INSTRUMENTS["bass"], [(0, 4, 36)])]
        with tempfile.TemporaryDirectory() as d:
            path = ac.write_midi(Path(d) / "x.mid", parts, 110, 16)
            data = path.read_bytes()
            self.assertEqual(data[:4], b"MThd")
            fmt, ntracks, tpq = struct.unpack(">HHH", data[8:14])
            self.assertEqual((fmt, ntracks, tpq), (1, 4, 480))
            if mido is None:
                return
            mid = mido.MidiFile(str(path))
            self.assertAlmostEqual(mido.tempo2bpm(next(m.tempo for m in mid.tracks[0] if m.type == "set_tempo")), 110, places=2)
            drum_notes = [m for m in mid.tracks[2] if m.type == "note_on"]
            self.assertEqual({m.channel for m in drum_notes}, {9})  # 드럼 = 10번 채널
            bass = mid.tracks[3]
            self.assertEqual(next(m.program for m in bass if m.type == "program_change"), 33)
            self.assertEqual(next(m for m in bass if m.type == "note_off").time, 4 * 120)  # 4칸 = 4분음표 = 480틱


@unittest.skipIf(np is None, "numpy 없음")
class TestAudio(unittest.TestCase):
    def test_render_with_fake_synth_and_slices(self):
        synths = []

        def factory(**kw):
            synths.append(FakeSynth(**kw))
            return synths[-1]

        a = ac.render([(0, 4, 36), (8, 1, 38)], ac.INSTRUMENTS["drums"], 120, 16, "x.sf2", sr=1000, tail=0.5, synth_factory=factory)
        self.assertEqual(a.shape, (round((16 * ac.step_seconds(120) + 0.5) * 1000), 2))
        self.assertEqual(synths[0].calls[0], ("program", 9, 128, 0, True))  # 드럼 = 128 뱅크
        on = int(round(4 * ac.step_seconds(120) * 1000))
        self.assertAlmostEqual(float(a[on - 1, 0]), 0.1, places=5)
        self.assertEqual(float(a[on + 1, 0]), 0.0)
        bar = ac.bar_slice(a, 120, 16, 0, sr=1000)
        self.assertEqual(len(bar), 2000)  # 120 BPM 한 마디 = 2초

    def test_gain_mix_wav(self):
        x = {"a": np.full((100, 2), 0.5, np.float32), "b": np.full((50, 2), 0.25, np.float32)}
        g = ac.common_gain(list(x.values()))
        self.assertAlmostEqual(0.5 * g, 10 ** (-1 / 20), places=5)
        m, mg = ac.mix(x, {"a": 1.0, "b": 2.0})
        self.assertAlmostEqual(float(np.abs(m).max()), 10 ** (-1 / 20), places=5)
        with tempfile.TemporaryDirectory() as d:
            ac.write_wav(Path(d) / "m.wav", m, 8000)
            with wave.open(str(Path(d) / "m.wav")) as w:
                self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()), (2, 2, 8000, 100))

    @unittest.skipIf(tinysoundfont is None or not os.environ.get("ACCOMP_TEST_SF2"), "tinysoundfont·ACCOMP_TEST_SF2 없음")
    def test_real_soundfont_each_instrument_sounds(self):
        for key in ("drums", "bass", "guitar_clean", "piano", "strings", "flute"):
            inst = ac.INSTRUMENTS[key]
            notes = [(0, 2, 38 if inst.drums else max(inst.low, 48)), (4, 2, 42 if inst.drums else max(inst.low, 52))]
            a = ac.render(notes, inst, 120, 8, os.environ["ACCOMP_TEST_SF2"], tail=0.5)
            with self.subTest(key=key):
                self.assertGreater(float(np.abs(a).max()), 0.01)


class TestSamples(unittest.TestCase):
    """음색 견본 — 사용자가 부탁한 묶음·톤이 다 있고, 악절이 두 마디 안·음역 안·반주 도구 악기와 번호가 맞는지."""

    def test_catalog_covers_request(self):
        keys = [x.key for x in ac.SAMPLES]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual({x.group for x in ac.SAMPLES}, set(ac.SAMPLE_GROUPS))
        by = {x.key: x for x in ac.SAMPLES}
        for need in ("acoustic_steel_melody", "acoustic_steel_chords", "electric_clean_melody", "electric_clean_chords",
                     "electric_overdrive_melody", "electric_overdrive_chords", "electric_distortion_melody",
                     "electric_distortion_chords", "piano_grand_melody", "piano_grand_chords", "drums_standard", "bass_finger",
                     "synth_square_lead", "synth_warm_pad"):
            self.assertIn(need, by)
        self.assertEqual((by["piano_grand_melody"].bank, by["piano_grand_melody"].preset), (0, 0))  # GM 0 = 그랜드 피아노
        for x in ac.SAMPLES:
            with self.subTest(sample=x.key):
                self.assertIn(x.part, ac.SAMPLE_PARTS)
                self.assertEqual(x.drums, x.group == "drums")
                if x.uses:  # 반주 도구 악기와 같은 소리라고 적었으면 번호가 같아야
                    inst = ac.INSTRUMENTS[x.uses]
                    if inst.drums:
                        self.assertEqual((x.drums, x.preset), (True, 0))  # 반주 렌더는 드럼 세트 0 (스탠더드)
                    else:
                        self.assertEqual((x.bank, x.preset), (0, inst.program))

    def test_notes_inside_two_bars_and_ranges(self):
        st = ac.step_seconds(ac.SAMPLE_BPM)
        end = ac.SAMPLE_STEPS * st
        for x in ac.SAMPLES:
            notes = ac.sample_notes(x)
            with self.subTest(sample=x.key):
                self.assertTrue(notes)
                self.assertEqual(notes, sorted(notes))
                for t, d, p, v in notes:
                    self.assertGreaterEqual(t, 0)
                    self.assertLessEqual(t + d, end + 1e-6)
                    self.assertTrue(0 <= p <= 127 and 1 <= v <= 127 and d > 0)
                if x.group == "bass":
                    self.assertTrue(all(28 <= p <= 60 for _, _, p, _ in notes))  # E1~C4
                if x.drums:
                    self.assertEqual({p for _, _, p, _ in notes}, {36, 38, 42, 46, 49})
        mel = ac.sample_notes(ac.Sample("m", "electric", "m", 0, 27, "melody", 100, -12))
        self.assertEqual([p for _, _, p, _ in mel][:3], [64, 62, 60])  # 기타 멜로디는 한 옥타브 아래 (E4 D4 C4)

    def test_strum_order(self):
        down = ac.strum(1.0, 0.5, (48, 52, 55, 60, 64), 90)
        self.assertEqual([p for _, _, p, _ in down], [48, 52, 55, 60, 64])  # 낮은 줄부터
        self.assertEqual([round(t - 1.0, 3) for t, _, _, _ in down], [0, 0.012, 0.024, 0.036, 0.048])
        up = ac.strum(1.0, 0.5, (48, 52, 55, 60, 64), 90, up=True)
        self.assertEqual([p for _, _, p, _ in up], [64, 60, 55, 52])  # 높은 줄부터, 윗줄 4개
        self.assertEqual({v for _, _, _, v in up}, {75})
        chords = ac.sample_notes(ac.SAMPLES[[x.key for x in ac.SAMPLES].index("acoustic_steel_chords")])
        first = [p for t, _, p, _ in chords if t < 0.1]
        self.assertEqual(first, [48, 52, 55, 60, 64])  # 첫 박 = C 코드 다운 스트럼
        with self.assertRaises(ValueError):
            ac.sample_notes(ac.Sample("x", "synth", "x", 0, 0, "nope"))

    @unittest.skipIf(np is None, "numpy 없음")
    def test_render_notes_bank_preset_velocity(self):
        class VelSynth(FakeSynth):
            def noteon(self, ch, key, vel):
                super().noteon(ch, key, vel)
                self.calls.append(("on", ch, key, vel))

        made = []
        factory = lambda **kw: made.append(VelSynth(**kw)) or made[-1]  # noqa: E731
        a = ac.render_notes([(0.0, 0.5, 60, 77), (0.25, 0.5, 64, 99)], "x.sf2", 8, 27, sr=1000, tail=0.5, synth_factory=factory)
        self.assertEqual(a.shape, (1250, 2))  # 마지막 음 끝 0.75초 + 꼬리 0.5초
        self.assertEqual(made[0].calls[0], ("program", 0, 8, 27, False))
        self.assertEqual([c[3] for c in made[0].calls if c[0] == "on"], [77, 99])
        ac.render_notes([(0.0, 0.1, 36, 100)], "x.sf2", 0, 25, drums=True, seconds=1.0, sr=1000, tail=0, synth_factory=factory)
        self.assertEqual(made[1].calls[0], ("program", 9, 128, 25, True))  # 드럼 = 10번 채널·뱅크 128·세트 번호

    @unittest.skipIf(tinysoundfont is None or not os.environ.get("ACCOMP_TEST_SF2"), "tinysoundfont·ACCOMP_TEST_SF2 없음")
    def test_real_soundfont_every_sample_sounds(self):
        for x in ac.SAMPLES:
            a = ac.render_notes(ac.sample_notes(x), os.environ["ACCOMP_TEST_SF2"], x.bank, x.preset, x.drums, 4.8, tail=0.5)
            with self.subTest(sample=x.key):
                self.assertGreater(float(np.abs(a).max()), 0.005)


if __name__ == "__main__":
    unittest.main()
