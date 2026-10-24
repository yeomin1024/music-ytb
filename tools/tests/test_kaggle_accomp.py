# -*- coding: utf-8 -*-
# VERSION: v1.0 — 2026-10-24 — tools/kaggle_accomp.py v1.0 시험 (가짜 AMT·가짜 torch·가짜 사운드폰트 — GPU·네트워크·GitHub 없음)
r"""
  python3 -m unittest tools.tests.test_kaggle_accomp -v     (저장소 최상위에서)

- generate_parts: 악기 차례(드럼→베이스→…), 바꿔 끼운 악기 필터를 생성이 따름, 조건이 악기마다 늘어남, 음이 적으면 다른 시드로,
  끝나면 원래 필터로 되돌림
- export: 악기마다 wav·1bar·mid, all.mid·mix — run: 기록·latest.txt, --push 는 로컬 bare 원격으로
"""
import json
import os
import random
import subprocess
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
import accomp as ac  # noqa: E402
import kaggle_accomp as ka  # noqa: E402

try:
    import numpy as np
except ImportError:
    np = None

CODE = "M1 bpm=120 steps=16 notes=E4@0+2,G4@2+2,A4@4+4,G4@8+2,E4@10+2,D4@12+4"


class FakeSample:
    """anticipation.sample 대신. 실제 add_token 처럼 모듈의 instr_logits 를 불러 허용된 음표만 낸다."""

    def __init__(self, empty_first=()):
        self.calls, self.empty_first = [], set(empty_first)
        self.instr_logits = self.original = lambda logits, history: logits

    def generate(self, model, start, end, controls=None, top_p=1.0):
        logits = np.zeros(ac.CONTROL_OFFSET + 10)
        logits = self.instr_logits(logits, [])
        allowed = np.flatnonzero(np.isfinite(logits[ac.NOTE_OFFSET:ac.NOTE_OFFSET + ac.MAX_NOTE]))
        instr = int(allowed[0] // 128)
        self.calls.append(dict(instr=instr, controls=len(controls) // 3, top_p=top_p, end=end))
        if instr in self.empty_first:
            self.empty_first.discard(instr)
            return [ac.TIME_OFFSET + 50, ac.DUR_OFFSET, ac.REST]
        rng = random.Random(len(self.calls))
        events, t = [], ac.LEAD_IN
        while t < end:
            note = int(allowed[rng.randrange(len(allowed))])
            events += [ac.TIME_OFFSET + int(round(t * 100)), ac.DUR_OFFSET + 20, ac.NOTE_OFFSET + note]
            t += 0.25
        return events + [ac.TIME_OFFSET + 60, ac.DUR_OFFSET, ac.REST]


def fake_render(notes, inst, bpm, total_steps, sf2, sr=44100, tail=1.5, synth_factory=None):
    n = int(round((total_steps * ac.step_seconds(bpm) + tail) * sr))
    return np.full((n, 2), 0.01 * (1 + len(notes) % 7), dtype=np.float32)


@unittest.skipIf(np is None, "numpy 없음")
class TestGenerate(unittest.TestCase):
    def test_order_mask_conditioning_retry(self):
        melody = ac.parse_melody_code(CODE)
        fake = FakeSample(empty_first={128})  # 드럼이 처음엔 아무것도 안 냄 → 다른 시드로
        seeds = []
        parts = ka.generate_parts(None, melody, ac.INSTRUMENTS["flute"], ["piano", "bass", "drums", "guitar_clean"], 2, 7, 0.9,
                                  sample_module=fake, seed_fn=seeds.append)
        self.assertEqual(list(parts), ["drums", "bass", "piano", "guitar_clean"])  # 리듬 먼저
        self.assertEqual([c["instr"] for c in fake.calls], [128, 128, 33, 0, 27])
        self.assertEqual(seeds, [7, 8, 107, 207, 307])  # 악기마다 seed+100·i, 다시 할 때 +1
        self.assertEqual(parts["drums"]["attempts"], 2)
        controls = [c["controls"] for c in fake.calls]
        self.assertEqual(controls[0], 12)  # 멜로디 6음 × 2마디
        self.assertTrue(controls[2] > controls[1] and controls[3] > controls[2] and controls[4] > controls[3])  # 앞 악기가 조건에 더해짐
        for key, part in parts.items():
            inst = ac.INSTRUMENTS[key]
            with self.subTest(key=key):
                self.assertTrue(part["notes"])
                self.assertTrue(all(inst.low <= p <= inst.high for _, _, p in part["notes"]))
                self.assertTrue(all(0 <= s < 32 for s, _, _ in part["notes"]))
        self.assertIs(fake.instr_logits, fake.original)  # 끝나면 되돌림
        self.assertAlmostEqual(fake.calls[0]["end"], ac.LEAD_IN + 32 * ac.step_seconds(120) + 0.01)

    def test_restores_filter_on_error(self):
        fake = FakeSample()
        fake.generate = mock.Mock(side_effect=RuntimeError("cuda"))
        with self.assertRaises(RuntimeError):
            ka.generate_parts(None, ac.parse_melody_code(CODE), ac.INSTRUMENTS["piano"], ["bass"], 1, 1, 0.9,
                              sample_module=fake, seed_fn=lambda s: None)
        self.assertIs(fake.instr_logits, fake.original)


@unittest.skipIf(np is None, "numpy 없음")
class TestExportAndRun(unittest.TestCase):
    def test_export_files(self):
        melody = ac.parse_melody_code(CODE)
        parts = {"drums": dict(notes=[(0, 1, 36), (4, 1, 38)]), "bass": dict(notes=[(0, 4, 40)])}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(ac, "render", side_effect=fake_render):
            files = ka.export(Path(d) / "x", melody, ac.INSTRUMENTS["piano"], parts, 2, sf2="x.sf2", sr=1000)
            for key in ("melody", "drums", "bass"):
                for suffix in (".wav", "_1bar.wav", ".mid"):
                    self.assertIn(f"{key}{suffix}", files)
                    self.assertTrue((Path(d) / "x" / f"{key}{suffix}").is_file())
            with wave.open(str(Path(d) / "x" / "drums_1bar.wav")) as w:
                self.assertEqual(w.getnframes(), 2000)  # 120 BPM 한 마디 = 2초 × 1000
            self.assertTrue((Path(d) / "x" / "all.mid").read_bytes().startswith(b"MThd"))

    def _fake_modules(self, fake):
        anticipation = types.ModuleType("anticipation")
        anticipation.sample = fake
        torch = types.ModuleType("torch")
        torch.manual_seed = lambda s: None
        return {"anticipation": anticipation, "anticipation.sample": fake, "torch": torch}

    def test_run_writes_record(self):
        fake = FakeSample()
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(sys.modules, self._fake_modules(fake)), \
                mock.patch.object(ka, "setup", return_value={"torch": "x"}), \
                mock.patch.object(ka, "load_model", return_value=(None, {"model": ka.MODELS["medium"], "revision": "abc"})), \
                mock.patch.object(ac, "render", side_effect=fake_render), mock.patch.object(ka.km, "setup_logging"), \
                mock.patch.object(ka.km, "push_check") as push_check:
            rc = ka.main(["run", "--melody", CODE, "--melody-instrument", "flute", "--instruments", "bass,drums",
                          "--loops", "2", "--name", "acc-1", "--out", d])
            self.assertEqual(rc, 0)
            push_check.assert_not_called()
            out = Path((Path(d) / "latest.txt").read_text(encoding="utf-8"))
            rec = json.loads((out / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(list(rec["parts"]), ["drums", "bass"])
            self.assertEqual(rec["melody_code"], ac.melody_code(ac.parse_melody_code(CODE)))
            self.assertEqual(rec["model"]["revision"], "abc")
            self.assertTrue((out / "bass_1bar.wav").is_file())
            ka.main(["run", "--melody", CODE, "--instruments", "bass", "--loops", "1", "--name", "acc-1", "--out", d])
            self.assertTrue((Path(d) / "latest.txt").read_text(encoding="utf-8").endswith("acc-1-2"))  # 덮어쓰지 않음

    def test_input_errors(self):
        for argv, want in ((["--melody", "M1 notes=60@0+40"], "멜로디"), (["--melody", CODE, "--instruments", "kazoo"], "반주 악기"),
                           (["--melody", CODE, "--melody-instrument", "drums"], "멜로디 악기"),
                           (["--melody", CODE, "--loops", "9"], "--loops"),
                           (["--melody", "M1 bpm=40 steps=64 notes=60@0+1", "--loops", "8"], "100초")):
            with self.subTest(want=want), mock.patch.object(ka.km, "setup_logging"), \
                    mock.patch.object(ka, "setup") as setup, mock.patch.object(ka.km, "log") as log:
                with self.assertRaises(SystemExit):
                    ka.main(["run", *argv, "--name", "x"])
                setup.assert_not_called()  # 입력이 틀리면 설치·모델도 시작하지 않음
                self.assertIn(want, " ".join(str(v) for c in log.call_args_list for v in c.kwargs.values()))

    def test_push_to_local_remote(self):
        """진짜 GitHub 대신 임시 bare 저장소 — results/accomp/<이름>/ 에 wav·mid·run.json·README."""
        fake = FakeSample()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            git = lambda *a, cwd=None: subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True)  # noqa: E731
            git("init", "-q", "--bare", "-b", "main", str(d / "origin.git"))
            git("clone", "-q", str(d / "origin.git"), str(d / "clone"))
            (d / "clone" / "README.md").write_text("x")
            git("-c", "user.name=t", "-c", "user.email=t@t", "add", "README.md", cwd=d / "clone")
            git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=d / "clone")
            git("push", "-q", "origin", "HEAD:main", cwd=d / "clone")
            env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
            with mock.patch.dict(sys.modules, self._fake_modules(fake)), \
                    mock.patch.object(ka, "setup", return_value={}), mock.patch.object(ka, "ROOT", d / "clone"), \
                    mock.patch.object(ka.km, "ROOT", d / "clone"), mock.patch.object(ka.km, "WORK", d / "work"), \
                    mock.patch.object(ka, "load_model", return_value=(None, {"model": "m", "revision": None})), \
                    mock.patch.object(ac, "render", side_effect=fake_render), mock.patch.object(ka.km, "setup_logging"), \
                    mock.patch.object(ka.km, "push_check"), mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(ka.km.time, "sleep"):
                rc = ka.main(["run", "--melody", CODE, "--instruments", "drums,bass", "--loops", "1", "--name", "acc-p",
                              "--out", str(d / "out"), "--push"])
            self.assertEqual(rc, 0)
            tree = git("ls-tree", "-r", "--name-only", "main", cwd=d / "origin.git").stdout.split()
            for f in ("drums_1bar.wav", "bass.mid", "mix.wav", "run.json", "README.md"):
                self.assertIn(f"results/accomp/acc-p/{f}", tree)
            readme = git("show", "main:results/accomp/acc-p/README.md", cwd=d / "origin.git").stdout
            self.assertIn("| 드럼 |", readme)
            self.assertIn("M1 bpm=120", readme)


class TestMidiInput(unittest.TestCase):
    @unittest.skipIf(np is None, "numpy 없음")
    def test_melody_from_own_midi(self):
        try:
            import mido  # noqa: F401
        except ImportError:
            self.skipTest("mido 없음")
        m = ac.parse_melody_code(CODE)
        with tempfile.TemporaryDirectory() as d:
            ac.write_midi(Path(d) / "m.mid", [(ac.INSTRUMENTS["piano"], [(s, n, p) for p, s, n in m["notes"]])], 120, 16)
            back = ka.melody_from_midi(Path(d) / "m.mid")
        self.assertEqual((back["bpm"], back["notes"]), (120.0, m["notes"]))


if __name__ == "__main__":
    unittest.main()
