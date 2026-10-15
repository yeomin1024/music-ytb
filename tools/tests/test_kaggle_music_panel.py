# -*- coding: utf-8 -*-
# VERSION: v1.0 — 2026-10-15 — tools/kaggle_music_panel.py v1.0 시험 (가짜 위젯 — GPU·네트워크·GitHub 없음)
r"""
  python3 -m unittest tools.tests.test_kaggle_music_panel -v     (저장소 최상위에서)

- 올린 파일 읽기(ipywidgets 7·8 모양), 안전한 파일 이름, 입력 확인, 명령 만들기
- 실행: 줄마다 로그 파일에도, 셀 멈춤(KeyboardInterrupt)이면 하위 프로세스까지 멈춤
- 버튼: 올린 곡을 /tmp 쪽에 저장 → 명령 실행, 실행 중 잠김, 끝나면 결과 표와 플레이어
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
import kaggle_music_panel as kp  # noqa: E402


class FakeWidget:
    def __init__(self, *a, **k):
        self.__dict__.update(k)
        self.__dict__.setdefault("value", ())  # 진짜 FileUpload 도 처음에는 빈 값
        self.disabled = False
        self._click = None
        self._observers = []

    def on_click(self, fn):
        self._click = fn

    def click(self):
        self._click(self)

    def observe(self, fn, names=None):
        self._observers.append(fn)

    def set_value(self, value):
        self.value = value
        for fn in self._observers:
            fn({"new": value})

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_modules(shown, displayed):
    W = types.ModuleType("ipywidgets")
    for name in ("FileUpload", "Text", "BoundedIntText", "Checkbox", "Button", "HTML", "Output", "Layout"):
        setattr(W, name, type(name, (FakeWidget,), {}))
    W.HBox = lambda children, **k: FakeWidget(children=children)
    W.VBox = lambda children, **k: shown.append(children) or FakeWidget(children=children)
    disp = types.ModuleType("IPython.display")
    disp.display = lambda x: displayed.append(x)
    disp.Audio = lambda filename=None, **k: ("audio", filename)
    ip = types.ModuleType("IPython")
    ip.display = disp
    return {"ipywidgets": W, "IPython": ip, "IPython.display": disp}


class TestHelpers(unittest.TestCase):
    def test_uploaded_files_both_ipywidgets_versions(self):
        v8 = ({"name": "song.mp3", "type": "audio/mpeg", "size": 3, "content": memoryview(b"abc")},)
        v7 = {"song.mp3": {"metadata": {"name": "song.mp3"}, "content": b"abc"}}
        self.assertEqual(kp.uploaded_files(v8), [("song.mp3", b"abc")])
        self.assertEqual(kp.uploaded_files(v7), [("song.mp3", b"abc")])
        self.assertEqual(kp.uploaded_files(()), [])
        self.assertEqual(kp.uploaded_files({}), [])

    def test_safe_filename(self):
        self.assertEqual(kp.safe_filename("내 노래 (live).MP3"), "내_노래_live.mp3")
        self.assertEqual(kp.safe_filename("../../etc/passwd.wav"), "passwd.wav")
        self.assertEqual(kp.safe_filename("???.flac"), "reference.flac")

    def test_check_inputs(self):
        ok = [("a.mp3", b"x")]
        self.assertIsNone(kp.check_inputs(ok, "jpop-01", 4, True, True))
        self.assertIsNone(kp.check_inputs(ok, "jpop-01", 4, False, False))  # 올리기를 끄면 토큰 없어도 됨
        cases = {
            "올려": ([], "a", 4, False, False),
            "하나만": (ok * 2, "a", 4, False, False),
            "오디오 파일이 아님": ([("a.txt", b"x")], "a", 4, False, False),
            "비었음": ([("a.mp3", b"")], "a", 4, False, False),
            "이름은": (ok, "한글이름", 4, False, False),
            "곡 수": (ok, "a", 0, False, False),
            "GITHUB_TOKEN": (ok, "a", 4, True, False),
        }
        for want, args in cases.items():
            with self.subTest(want=want):
                self.assertIn(want, kp.check_inputs(*args))

    def test_save_upload_and_command(self):
        with tempfile.TemporaryDirectory() as d:
            path = kp.save_upload("my song.mp3", b"abc", Path(d) / "up")
            self.assertEqual(path.read_bytes(), b"abc")
            self.assertEqual(path.name, "my_song.mp3")
        cmd = kp.build_command(Path("/tmp/x.mp3"), "jpop-01", 4, " J-pop band ", True, python="py")
        self.assertEqual(cmd[:3], ["py", str(kp.TOOLS / "kaggle_music_yue2.py"), "run"])
        self.assertEqual(cmd[3:], ["--ref", "/tmp/x.mp3", "--count", "4", "--name", "jpop-01", "--style-extra", "J-pop band", "--push"])
        self.assertNotIn("--style-extra", kp.build_command("x", "n", 2, "", False))
        self.assertNotIn("--push", kp.build_command("x", "n", 2, "", False))

    def test_results_rows(self):
        with tempfile.TemporaryDirectory() as d:
            nd = Path(d)
            (nd / "seed-1").mkdir()
            (nd / "seed-1" / "similarity.json").write_text(json.dumps({"verdict_ko": "통과"}), encoding="utf-8")
            (nd / "seed-1" / "kaggle_run.json").write_text(json.dumps({"metrics": {"seconds": 61.0}}), encoding="utf-8")
            (nd / "seed-1" / "audio.mp3").write_bytes(b"a")
            (nd / "seed-2").mkdir()
            rows = kp.results_rows(nd)
            self.assertEqual(rows[0], ("seed-1", 61.0, "통과", nd / "seed-1" / "audio.mp3"))
            self.assertEqual(rows[1], ("seed-2", None, "검사 전", None))


class TestRunCommand(unittest.TestCase):
    def test_streams_and_logs(self):
        with tempfile.TemporaryDirectory() as d:
            lines = []
            rc = kp.run_command([sys.executable, "-c", "print('[GEN] a'); print('[GEN] b')"], write=lines.append,
                                log_file=Path(d) / "panel.log")
            self.assertEqual(rc, 0)
            self.assertIn("[GEN] b", lines)
            self.assertIn("[GEN] a", (Path(d) / "panel.log").read_text(encoding="utf-8"))

    def test_interrupt_stops_child_group(self):
        """셀 멈춤 → 자식과 손자 프로세스까지 멈춤 (GPU 작업 프로세스가 남지 않게)."""
        with tempfile.TemporaryDirectory() as d:
            marker = Path(d) / "grandchild.pid"
            child = ("import subprocess, sys, time; "
                     f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
                     f"open({str(marker)!r}, 'w').write(str(p.pid)); print('started', flush=True); time.sleep(60)")

            def write(line):
                if line == "started":
                    raise KeyboardInterrupt

            t0 = time.time()
            rc = kp.run_command([sys.executable, "-c", child], write=write, log_file=Path(d) / "panel.log")
            self.assertEqual(rc, 130)
            self.assertLess(time.time() - t0, 30)
            gpid = int(marker.read_text())
            for _ in range(50):  # 손자 프로세스가 끝났는지 (좀비면 /proc 상태 Z)
                try:
                    state = Path(f"/proc/{gpid}/stat").read_text().split()[2]
                except FileNotFoundError:
                    state = "gone"
                if state in ("gone", "Z"):
                    break
                time.sleep(0.1)
            self.assertIn(state, ("gone", "Z"))


class TestPanel(unittest.TestCase):
    def test_upload_and_click(self):
        shown, displayed, calls = [], [], []
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            out_dir = d / "music"
            (out_dir / "jpop-01" / "seed-1").mkdir(parents=True)
            (out_dir / "jpop-01" / "seed-1" / "audio.mp3").write_bytes(b"a")
            (out_dir / "jpop-01" / "prompt.txt").write_text("J-pop band, piano, 150 BPM\n", encoding="utf-8")
            with mock.patch.dict(sys.modules, fake_modules(shown, displayed)), \
                    mock.patch.object(kp, "UPLOAD_DIR", d / "upload"), mock.patch.object(kp.km, "OUT", out_dir), \
                    mock.patch.dict(os.environ, {"GITHUB_TOKEN": "x"}):
                def fake_run(cmd, write=print, log_file=None):
                    note, upload_row, text_row, opt_row, go, out = shown[0]
                    calls.append((cmd, go.disabled, upload_row.children[0].disabled))
                    return 0
                with mock.patch.object(kp, "run_command", side_effect=fake_run):
                    self.assertEqual(kp.panel(), 0)
                    note, upload_row, text_row, opt_row, go, out = shown[0]
                    up, status = upload_row.children
                    name, extra = text_row.children
                    count, push = opt_row.children
                    self.assertEqual((count.value, push.value), (kp.DEFAULT_COUNT, True))  # 기본 4곡, 올리기 켬
                    go.click()  # 곡을 안 올림 → 실행 안 함
                    self.assertEqual(calls, [])
                    up.set_value(({"name": "Fav Song.mp3", "content": memoryview(b"MP3DATA")},))
                    self.assertIn("Fav_Song.mp3", status.value)
                    name.value, extra.value = "jpop-01", "J-pop band"
                    go.click()
                    self.assertFalse(go.disabled)  # 끝나면 다시 풀림
            self.assertEqual(len(calls), 1)
            cmd, go_locked, up_locked = calls[0]
            self.assertTrue(go_locked and up_locked)  # 실행 중에는 잠김
            ref = Path(cmd[cmd.index("--ref") + 1])
            self.assertEqual(ref, d / "upload" / "Fav_Song.mp3")
            self.assertEqual(ref.read_bytes(), b"MP3DATA")
            self.assertEqual(cmd[cmd.index("--name") + 1], "jpop-01")
            self.assertIn("--push", cmd)
            self.assertIn(("audio", str(out_dir / "jpop-01" / "seed-1" / "audio.mp3")), displayed)

    def test_no_widgets(self):
        with mock.patch.dict(sys.modules, {"ipywidgets": None}):
            self.assertEqual(kp.panel(), 1)


if __name__ == "__main__":
    unittest.main()
