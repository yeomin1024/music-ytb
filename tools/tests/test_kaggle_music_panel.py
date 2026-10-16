# -*- coding: utf-8 -*-
# VERSION: v1.1 — 2026-10-16 — kaggle_music_panel.py v1.1 시험: ① 분석 → ② Claude 답 읽기 → ③ 곡 만들기 (가짜 위젯 — GPU·네트워크·GitHub 없음)
#          (v1.0: 업로드·한 번에 만들기)
r"""
  python3 -m unittest tools.tests.test_kaggle_music_panel -v     (저장소 최상위에서)

- 올린 파일 읽기(ipywidgets 7·8 모양), 안전한 파일 이름, 입력 확인, 명령 만들기
- 실행: 줄마다 로그 파일에도, 셀 멈춤(KeyboardInterrupt)이면 하위 프로세스까지 멈춤
- 버튼: ① 올린 곡·멘트를 /tmp 쪽에 저장 → analyze 명령 → Claude 에게 보낼 글 칸, ② 답 → STYLE·SECTIONS·가사 칸,
  ③ run 명령(가사 넣기면 노래 모드), 실행 중 잠김, 끝나면 결과 표와 플레이어
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
    for name in ("FileUpload", "Text", "Textarea", "BoundedIntText", "Checkbox", "Button", "HTML", "Output", "Layout"):
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

    def test_check_upload(self):
        ok = [("a.mp3", b"x")]
        self.assertIsNone(kp.check_upload(ok, "my-01"))
        cases = {"올려": ([], "a"), "하나만": (ok * 2, "a"), "오디오 파일이 아님": ([("a.txt", b"x")], "a"),
                 "비었음": ([("a.mp3", b"")], "a"), "이름은": (ok, "한글이름")}
        for want, args in cases.items():
            with self.subTest(want=want):
                self.assertIn(want, kp.check_upload(*args))

    def test_check_make(self):
        lyr = "[Verse]\n빗물이 번진 창가\n"
        self.assertIsNone(kp.check_make(True, "pop", "Intro,Verse", False, "", 4, True, True))
        self.assertIsNone(kp.check_make(True, "pop", "", True, lyr, 4, False, False))  # 노래 모드는 SECTIONS 안 봄
        cases = {
            "① 분석": (False, "pop", "Intro", False, "", 4, False, False),
            "STYLE 이 비었음": (True, " ", "Intro", False, "", 4, False, False),
            "한 줄": (True, "a\nb", "Intro", False, "", 4, False, False),
            "태그 줄과 가사": (True, "pop", "", True, "가사만", 4, False, False),
            "SECTIONS": (True, "pop", "인트로", False, "", 4, False, False),
            "곡 수": (True, "pop", "Intro", False, "", 0, False, False),
            "GITHUB_TOKEN": (True, "pop", "Intro", False, "", 4, True, False),
        }
        for want, args in cases.items():
            with self.subTest(want=want):
                self.assertIn(want, kp.check_make(*args))

    def test_save_upload_and_commands(self):
        with tempfile.TemporaryDirectory() as d:
            path = kp.save_upload("my song.mp3", b"abc", Path(d) / "up")
            self.assertEqual(path.read_bytes(), b"abc")
            self.assertEqual(path.name, "my_song.mp3")
        tool = str(kp.TOOLS / "kaggle_music_yue2.py")
        self.assertEqual(kp.build_analyze_command("/r.mp3", "n1", "/c.txt", True, python="py"),
                         ["py", tool, "analyze", "--ref", "/r.mp3", "--name", "n1", "--comment-file", "/c.txt", "--lyrics"])
        self.assertNotIn("--lyrics", kp.build_analyze_command("/r.mp3", "n1", None, False))
        song = kp.build_run_command("/r.mp3", "n1", 3, True, " pop ", "", "/l.txt", python="py")
        self.assertEqual(song, ["py", tool, "run", "--ref", "/r.mp3", "--name", "n1", "--count", "3", "--style", "pop",
                                "--mode", "song", "--lyrics-file", "/l.txt", "--push"])
        inst = kp.build_run_command("/r.mp3", "n1", 2, False, "pop", "Intro,Verse", None, python="py")
        self.assertEqual(inst[-4:], ["--mode", "instrumental", "--sections", "Intro,Verse"])

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


CLAUDE_ANSWER = """```
STYLE: J-pop band, rainy night, reverb-drenched clean electric guitar, minor key, 148 BPM, Korean, warm female vocal
SECTIONS: Intro, Verse, Chorus, Outro
LYRICS:
[Intro]
[Verse]
빗물이 번진 버스 창가에
[Chorus]
멈추지 마 이 밤이 끝나도
[Outro]
```"""


class TestPanel(unittest.TestCase):
    def test_three_steps(self):
        shown, displayed, calls = [], [], []
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            out_dir = d / "music"

            def fake_run(cmd, write=print, log_file=None):
                widgets = shown[0]
                calls.append((cmd, widgets[4].disabled))  # widgets[4] = ① 버튼 — 실행 중 잠김
                name = cmd[cmd.index("--name") + 1]
                nd = out_dir / name
                if cmd[2] == "analyze":
                    nd.mkdir(parents=True)
                    (nd / "claude_request.txt").write_text("CLAUDE 에게 보낼 글", encoding="utf-8")
                    (nd / "prompt.txt").write_text("pop, piano, 120 BPM\n", encoding="utf-8")
                else:
                    (nd / "seed-1").mkdir()
                    (nd / "seed-1" / "audio.mp3").write_bytes(b"a")
                    (nd / "seed-1" / "similarity.json").write_text(json.dumps({"verdict_ko": "통과"}), encoding="utf-8")
                return 0

            with mock.patch.dict(sys.modules, fake_modules(shown, displayed)), \
                    mock.patch.object(kp, "UPLOAD_DIR", d / "upload"), mock.patch.object(kp.km, "OUT", out_dir), \
                    mock.patch.dict(os.environ, {"GITHUB_TOKEN": "x"}), mock.patch.object(kp, "run_command", side_effect=fake_run):
                self.assertEqual(kp.panel(), 0)
                (note, upload_row, comment, opt_row, analyze_btn, request, answer, parse_btn, style, sections, lyrics,
                 make_row, make_btn, out) = shown[0]
                up, status = upload_row.children
                want_lyrics, name = opt_row.children
                count, push = make_row.children
                make_btn.click()  # ① 전 → 실행 안 함
                analyze_btn.click()  # 곡 없음 → 실행 안 함
                self.assertEqual(calls, [])
                up.set_value(({"name": "Fav Song.mp3", "content": memoryview(b"MP3DATA")},))
                comment.value = "비 오는 밤 버스 창가"
                want_lyrics.value = True
                name.value = "my-01"
                analyze_btn.click()
                self.assertEqual(request.value, "CLAUDE 에게 보낼 글")
                self.assertEqual(style.value, "pop, piano, 120 BPM")  # Claude 를 건너뛸 때 쓰는 자동 프롬프트
                answer.value = "잡담"
                parse_btn.click()
                self.assertEqual(style.value, "pop, piano, 120 BPM")  # 형식이 틀리면 칸을 바꾸지 않음
                answer.value = CLAUDE_ANSWER
                parse_btn.click()
                self.assertTrue(style.value.startswith("J-pop band, rainy night"))
                self.assertEqual(sections.value, "Intro,Verse,Chorus,Outro")
                self.assertIn("빗물이 번진 버스 창가에", lyrics.value)
                lyrics.value = lyrics.value.replace("빗물이", "빗방울이")  # 가사 고치기
                count.value = 2
                make_btn.click()
                self.assertFalse(make_btn.disabled)
            self.assertEqual(len(calls), 2)
            (acmd, locked), (rcmd, _) = calls
            self.assertTrue(locked)
            self.assertEqual(acmd[2], "analyze")
            ref = Path(acmd[acmd.index("--ref") + 1])
            self.assertEqual((ref, ref.read_bytes()), (d / "upload" / "Fav_Song.mp3", b"MP3DATA"))
            comment_file = Path(acmd[acmd.index("--comment-file") + 1])
            self.assertEqual(comment_file.read_text(encoding="utf-8"), "비 오는 밤 버스 창가")
            self.assertIn("--lyrics", acmd)
            self.assertEqual(rcmd[2], "run")
            self.assertEqual(rcmd[rcmd.index("--ref") + 1], str(ref))
            self.assertEqual(rcmd[rcmd.index("--name") + 1], "my-01")
            self.assertEqual(rcmd[rcmd.index("--count") + 1], "2")
            self.assertEqual(rcmd[rcmd.index("--mode") + 1], "song")
            lyrics_file = Path(rcmd[rcmd.index("--lyrics-file") + 1])
            self.assertIn("빗방울이 번진", lyrics_file.read_text(encoding="utf-8"))  # 고친 가사로
            self.assertIn("--push", rcmd)
            self.assertIn(("audio", str(out_dir / "my-01" / "seed-1" / "audio.mp3")), displayed)

    def test_no_widgets(self):
        with mock.patch.dict(sys.modules, {"ipywidgets": None}):
            self.assertEqual(kp.panel(), 1)


if __name__ == "__main__":
    unittest.main()
