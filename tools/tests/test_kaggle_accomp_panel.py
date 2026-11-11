# -*- coding: utf-8 -*-
# VERSION: v1.1 — 2026-11-11 — kaggle_accomp_panel.py v1.1: samples()(음색 견본 플레이어) 시험 추가
# (v1.0) — 2026-10-24 — kaggle_accomp_panel.py v1.0 시험 (가짜 위젯 — GPU·네트워크·GitHub 없음)
r"""
  python3 -m unittest tools.tests.test_kaggle_accomp_panel -v     (저장소 최상위에서)

- 멜로디 코드 요약·입력 확인(코드·MIDI·악기·길이·이름·토큰)·명령 만들기
- 이번 실행 결과 폴더 찾기(latest.txt 가 실행 전 것이면 무시)·결과 표
- 버튼: 코드 → run 명령(악기 순서·옵션), 실행 중 잠김, 끝나면 악기마다 한 마디·반복 플레이어 / 코드 없이 MIDI → --melody-mid
"""
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "tests"))
import kaggle_accomp_panel as kap  # noqa: E402
from test_kaggle_music_panel import FakeWidget, fake_modules  # noqa: E402

CODE = "M1 bpm=120 steps=16 notes=E4@0+2,G4@2+2,A4@4+4,G4@8+2,E4@10+2,D4@12+4"


def fake_modules_accomp(shown, displayed):
    mods = fake_modules(shown, displayed)
    W = mods["ipywidgets"]
    for name in ("Dropdown", "IntText"):
        setattr(W, name, type(name, (FakeWidget,), {}))
    W.HBox = lambda children, **k: FakeWidget(children=children)
    return mods


def fake_result(out_root, name, parts=("drums", "bass")):
    d = Path(out_root) / name
    d.mkdir(parents=True)
    for key in ("melody", *parts, "mix"):
        (d / f"{key}.wav").write_bytes(b"RIFF")
        (d / f"{key}_1bar.wav").write_bytes(b"RIFF")
    rec = dict(melody_instrument="flute", melody=dict(notes=6),
               parts={k: dict(notes=n, instrument=kap.ac.INSTRUMENTS[k].name_ko) for n, k in enumerate(parts, start=8)})
    (d / "run.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    (Path(out_root) / "latest.txt").write_text(str(d), encoding="utf-8")
    return d


class TestHelpers(unittest.TestCase):
    def test_melody_summary(self):
        self.assertEqual(kap.melody_summary(CODE), "음 6개 · 120 BPM · 16칸 · D4~A4")
        self.assertEqual(kap.melody_summary(" "), "아직 멜로디 코드가 없음")
        self.assertIn("코드 오류", kap.melody_summary("M1 notes=60@0+4,62@2+2"))

    def test_check_inputs(self):
        ok = dict(code=CODE, midi_files=[], keys=["drums", "bass"], loops=4, name="acc-01", push=True, token_present=True)
        self.assertIsNone(kap.check_inputs(**ok))
        cases = [
            (dict(code=""), "멜로디 코드를 붙여 넣거나"),
            (dict(code="M1 notes=60@15+2"), "마디 밖"),
            (dict(code="", midi_files=[("a.mid", b"x"), ("b.mid", b"y")]), "하나만"),
            (dict(code="", midi_files=[("a.mp3", b"x")]), "MIDI 파일이 아님"),
            (dict(code="", midi_files=[("a.mid", b"")]), "비었음"),
            (dict(keys=[]), "하나 이상"),
            (dict(keys=["flute"]), "모르는 반주 악기"),  # 멜로디 전용 악기
            (dict(loops=9), "1~8"),
            (dict(name="내 반주"), "영문"),
            (dict(token_present=False), "GITHUB_TOKEN"),
        ]
        for change, want in cases:
            with self.subTest(change=change):
                self.assertIn(want, kap.check_inputs(**{**ok, **change}) or "")
        self.assertIsNone(kap.check_inputs(**{**ok, "code": "", "midi_files": [("Song.MID", b"MThd")]}))
        self.assertIsNone(kap.check_inputs(**{**ok, "push": False, "token_present": False}))
        self.assertIsNone(kap.check_inputs(**{**ok, "code": "M1 bpm=40 notes=60@0+1", "loops": 8}))  # 40 BPM 6초 × 8 = 48초

    def test_too_long(self):
        with mock.patch.object(kap, "MAX_SECONDS", 20):
            self.assertIsNone(kap.check_inputs(CODE, [], ["bass"], 8, "a", False, False))  # 120 BPM 2초 × 8 = 16.1초
            self.assertIn("너무 김", kap.check_inputs("M1 bpm=60 notes=60@0+1", [], ["bass"], 8, "a", False, False))

    def test_build_command(self):
        cmd = kap.build_command(CODE.replace(" ", "  \n"), None, "flute", ["drums", "bass"], 2, 7, "small", "acc-x", True, python="py")
        self.assertEqual(cmd[:3], ["py", str(kap.TOOLS / "kaggle_accomp.py"), "run"])
        self.assertEqual(cmd[cmd.index("--melody") + 1], CODE)  # 한 줄로
        for flag, want in (("--melody-instrument", "flute"), ("--instruments", "drums,bass"), ("--loops", "2"), ("--seed", "7"),
                           ("--model", "small"), ("--name", "acc-x")):
            self.assertEqual(cmd[cmd.index(flag) + 1], want)
        self.assertIn("--push", cmd)
        cmd = kap.build_command(None, Path("/tmp/x.mid"), "piano", ["bass"], 4, 1, "medium", "a", False)
        self.assertEqual(cmd[cmd.index("--melody-mid") + 1], "/tmp/x.mid")
        self.assertNotIn("--melody", cmd)
        self.assertNotIn("--push", cmd)

    def test_latest_result_and_rows(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(kap.latest_result(d, 0))
            res = fake_result(d, "acc-1")
            self.assertEqual(kap.latest_result(d, 0), res)
            self.assertIsNone(kap.latest_result(d, time.time() + 5))  # 실행 전에 남은 latest.txt 는 무시
            (res / "bass_1bar.wav").unlink()
            rows = kap.result_rows(res)
            self.assertEqual([r[0] for r in rows], ["melody", "drums", "bass", "mix"])
            self.assertEqual(rows[0][1:3], (f"{kap.ac.INSTRUMENTS['flute'].name_ko} (멜로디)", 6))
            self.assertEqual(rows[1][2], 8)
            self.assertIsNone(rows[2][3])  # 없는 파일은 None
            self.assertEqual(rows[3][1:3], ("합친 소리", None))
            self.assertEqual(rows[3][4], res / "mix.wav")


class TestPanel(unittest.TestCase):
    def test_code_flow(self):
        shown, displayed, calls = [], [], []
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)

            def fake_run(cmd, write=print, log_file=None):
                calls.append((cmd, shown[0][8].disabled, log_file))  # [8] = 반주 만들기 버튼 — 실행 중 잠김
                fake_result(d, cmd[cmd.index("--name") + 1])
                return 0

            with mock.patch.dict(sys.modules, fake_modules_accomp(shown, displayed)), mock.patch.object(kap.ka, "OUT", d), \
                    mock.patch.object(kap, "UPLOAD_DIR", d / "upload"), mock.patch.dict(os.environ, {"GITHUB_TOKEN": "x"}), \
                    mock.patch.object(kap.kp, "run_command", side_effect=fake_run):
                self.assertEqual(kap.panel(), 0)
                note, code, summary, mid_up, opt_row, _title, inst_box, run_row, make_btn, out = shown[0]
                mel_inst, loops, seed = opt_row.children
                model, name, push = run_row.children
                self.assertIn(kap.MELODY_PAGE_URL, note.value)
                checked = [cb.description for cb in inst_box.children if cb.value]
                self.assertEqual(len(checked), 4)  # 기본: 드럼·베이스·클린 기타·피아노
                make_btn.click()  # 코드 없음 → 실행 안 함
                self.assertEqual(calls, [])
                code.set_value(CODE)
                self.assertIn("음 6개", summary.value)
                for cb in inst_box.children:  # 드럼·베이스만
                    cb.value = cb.description in (kap.ac.INSTRUMENTS["drums"].name_ko, kap.ac.INSTRUMENTS["bass"].name_ko)
                mel_inst.value, loops.value, seed.value, name.value = "flute", 2, 5, "acc-t"
                make_btn.click()
                self.assertFalse(make_btn.disabled)
            self.assertEqual(len(calls), 1)
            cmd, locked, log_file = calls[0]
            self.assertTrue(locked)
            self.assertEqual(log_file, kap.LOG_FILE)
            self.assertEqual(cmd[cmd.index("--melody") + 1], CODE)
            self.assertEqual(cmd[cmd.index("--instruments") + 1], "drums,bass")
            self.assertEqual((cmd[cmd.index("--loops") + 1], cmd[cmd.index("--seed") + 1]), ("2", "5"))
            self.assertIn("--push", cmd)
            played = [x[1] for x in displayed if isinstance(x, tuple) and x[0] == "audio"]
            self.assertIn(str(d / "acc-t" / "drums_1bar.wav"), played)
            self.assertIn(str(d / "acc-t" / "mix.wav"), played)
            self.assertEqual(len(played), 8)  # 멜로디·드럼·베이스·합친 소리 × (한 마디·반복)

    def test_midi_flow_and_failure(self):
        shown, displayed, calls = [], [], []
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with mock.patch.dict(sys.modules, fake_modules_accomp(shown, displayed)), mock.patch.object(kap.ka, "OUT", d), \
                    mock.patch.object(kap, "UPLOAD_DIR", d / "upload"), mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch.object(kap.kp, "run_command", side_effect=lambda cmd, **k: calls.append(cmd) or 1):
                kap.panel()
                note, code, summary, mid_up, opt_row, _title, inst_box, run_row, make_btn, out = shown[0]
                model, name, push = run_row.children
                mid_up.value = ({"name": "my tune.mid", "content": memoryview(b"MThd")},)
                make_btn.click()  # 토큰 없음 + 올리기 켬 → 실행 안 함
                self.assertEqual(calls, [])
                push.value = False
                make_btn.click()  # 실행은 했지만 결과 없음(실패) → 플레이어 없음
            self.assertEqual(len(calls), 1)
            mid = Path(calls[0][calls[0].index("--melody-mid") + 1])
            self.assertEqual((mid, mid.read_bytes()), (d / "upload" / "my_tune.mid", b"MThd"))
            self.assertNotIn("--push", calls[0])
            self.assertEqual(displayed[1:], [])  # 첫 display = 패널

    def test_no_widgets(self):
        with mock.patch.dict(sys.modules, {"ipywidgets": None}):
            self.assertEqual(kap.panel(), 1)


def fake_ipython(displayed):
    disp = types.ModuleType("IPython.display")
    disp.display = displayed.append
    disp.HTML = lambda data=None, **k: ("html", data)  # IPython.display.HTML 처럼 위치 인자
    disp.Audio = lambda filename=None, **k: ("audio", filename)
    ip = types.ModuleType("IPython")
    ip.display = disp
    return {"IPython": ip, "IPython.display": disp}


class TestSamples(unittest.TestCase):
    def test_build_samples_command(self):
        cmd = kap.build_samples_command(" electric, drums ", 90, Path("/o"), python="py")
        self.assertEqual(cmd, ["py", str(kap.TOOLS / "kaggle_accomp.py"), "samples", "--out", "/o", "--only", "electric,drums",
                               "--bpm", "90"])
        self.assertNotIn("--only", kap.build_samples_command("", None, Path("/o")))

    def test_players_grouped_in_order(self):
        displayed, calls = [], []
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)

            def fake_run(cmd, write=print, log_file=None):
                calls.append(cmd)
                out = Path(cmd[cmd.index("--out") + 1])
                items = [dict(key=x.key, group=x.group, group_ko=kap.ac.SAMPLE_GROUPS[x.group], name=x.name_ko,
                              soundfont=f"{x.bank}/{x.preset}", uses=x.uses, file=f"{x.group}/{x.key}.wav")
                         for x in kap.ac.SAMPLES if x.group in ("electric", "drums")]
                out.mkdir(parents=True)
                (out / "index.json").write_text(json.dumps({"samples": items}, ensure_ascii=False), encoding="utf-8")
                return 0

            with mock.patch.dict(sys.modules, fake_ipython(displayed)), mock.patch.object(kap.ka, "OUT", d), \
                    mock.patch.object(kap.kp, "run_command", side_effect=fake_run):
                self.assertEqual(kap.samples("electric,drums"), 0)
            self.assertEqual(calls[0][calls[0].index("--only") + 1], "electric,drums")
            heads = [x[1] for x in displayed if x[0] == "html" and x[1].startswith("<h3>")]
            self.assertEqual(heads, ["<h3>일렉트릭 기타</h3>", "<h3>드럼</h3>"])
            audio = [x[1] for x in displayed if x[0] == "audio"]
            self.assertEqual(audio[0], str(d / "samples" / "electric" / "electric_clean_melody.wav"))
            self.assertEqual(len(audio), len([x for x in kap.ac.SAMPLES if x.group in ("electric", "drums")]))
            self.assertTrue(any("guitar_clean" in x[1] for x in displayed if x[0] == "html"))
            self.assertTrue(any("아직 없음" in x[1] for x in displayed if x[0] == "html"))

    def test_failure_shows_nothing(self):
        displayed = []
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(sys.modules, fake_ipython(displayed)), \
                mock.patch.object(kap.ka, "OUT", Path(d)), mock.patch.object(kap.kp, "run_command", return_value=1):
            self.assertEqual(kap.samples(), 1)
        self.assertEqual(displayed, [])


if __name__ == "__main__":
    unittest.main()
