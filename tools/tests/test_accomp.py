# -*- coding: utf-8 -*-
# VERSION: v1.0 — 2026-10-24 — tools/accomp.py v1.0 시험: 멜로디 코드, AMT 토큰 왕복, 악기·음역 거르기, 격자, MIDI, 렌더·섞기·WAV
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


if __name__ == "__main__":
    unittest.main()
