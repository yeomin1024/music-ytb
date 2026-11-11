# -*- coding: utf-8 -*-
# VERSION: v1.0 — 2026-11-11 — tools/ai_samples.py v1.0 시험: 견본 표(사용자가 부탁한 악기·톤·스트링), 고르기, 작업 나누기,
#          잘라 내기 명령, 트랙 비중·'한 악기' 판정 (GPU·네트워크 없음)
r"""
  python3 -m unittest tools.tests.test_ai_samples -v     (저장소 최상위에서)
"""
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
import ai_samples as ais  # noqa: E402
import kaggle_music_yue2 as km  # noqa: E402


class TestCatalog(unittest.TestCase):
    def test_covers_request(self):
        by = {s.key: s for s in ais.AI_SAMPLES}
        self.assertEqual(len(by), len(ais.AI_SAMPLES))
        self.assertEqual({s.group for s in ais.AI_SAMPLES}, set(ais.GROUPS))
        # 어쿠스틱 기타 멜로디·코드, 일렉 톤마다 멜로디·코드, 그랜드 피아노, 스트링, 드럼, 베이스, 신스
        for need in ("acoustic_melody", "acoustic_chords", "electric_clean_melody", "electric_clean_chords",
                     "electric_crunch_melody", "electric_crunch_chords", "electric_dist_melody", "electric_dist_chords",
                     "piano_melody", "piano_chords", "strings_melody", "strings_chords", "drums_acoustic", "bass_electric",
                     "synth_lead", "synth_pad"):
            self.assertIn(need, by)
        self.assertIn("grand piano", by["piano_melody"].style)
        self.assertIn("string section", by["strings_chords"].style)

    def test_prompts_ask_for_one_instrument(self):
        for s in ais.AI_SAMPLES:
            with self.subTest(sample=s.key):
                style = km.instrumental_style(s.style)
                self.assertLessEqual(len(style), 400)  # Claude 에게 보낼 글과 같은 한도
                self.assertTrue(style.startswith("Instrumental, solo "))
                self.assertIn("no vocals", style)
                self.assertIn("no other instruments", style)
                self.assertIn(s.stem, ais.DEMUCS_STEMS)
                self.assertIn(s.plan_mode, ("melody", "full"))
                if "멜로디" in s.name_ko:
                    self.assertEqual(s.plan_mode, "melody")  # 단일 멜로디 = 선율만 계획
                if "코드" in s.name_ko:
                    self.assertEqual(s.plan_mode, "full")
                want = {"acoustic": "guitar", "electric": "guitar", "piano": "piano", "drums": "drums", "bass": "bass",
                        "strings": "other", "synth": "other"}[s.group]
                self.assertEqual(s.stem, want)

    def test_select(self):
        self.assertEqual([s.group for s in ais.select("strings")], ["strings", "strings"])
        self.assertEqual([s.key for s in ais.select("", "piano_chords, strings_melody")], ["piano_chords", "strings_melody"])
        self.assertEqual(len(ais.select()), len(ais.AI_SAMPLES))
        for bad in (("guitar", ""), ("", "nope")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ais.select(*bad)
        with self.assertRaises(ValueError):
            ais.select("drums", "piano_chords")  # 겹치는 것이 없음


class TestJobs(unittest.TestCase):
    def test_plan_jobs_round_robin_and_seeds(self):
        picked = ais.select("piano,strings")
        plan = ais.plan_jobs(picked, [0, 1], seed=5, tries=3, sections="Verse", seconds=8)
        self.assertEqual({g: [j["key"] for j in js] for g, js in plan.items()},
                         {0: ["piano_melody", "strings_melody"], 1: ["piano_chords", "strings_chords"]})
        job = plan[0][1]
        self.assertEqual((job["seeds"], job["stem"], job["seconds"], job["group_ko"]), ([5, 6, 7], "other", 8, "스트링"))
        for bad in (dict(tries=0), dict(tries=6)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ais.plan_jobs(picked, [0], **bad)
        with self.assertRaises(ValueError):
            ais.plan_jobs(picked, [])

    def test_clip_and_mp3_commands(self):
        cmd = ais.clip_cmd("a.flac", "c.wav", 10)
        af = cmd[cmd.index("-af") + 1]
        self.assertIn("silenceremove=start_periods=1:start_threshold=-45dB", af)
        self.assertIn("atrim=0:10", af)
        self.assertIn("afade=t=out:st=9.40:d=0.6", af)
        self.assertEqual(cmd[-1], "c.wav")
        self.assertEqual(ais.mp3_cmd("c.wav", "c.mp3")[-3:], ["-b:a", "192k", "c.mp3"])


class TestVerdict(unittest.TestCase):
    def test_shares_and_verdict(self):
        shares = ais.stem_shares({"guitar": 8.0, "drums": 1.5, "vocals": 0.5, "piano": -1})
        self.assertEqual(shares, {"guitar": 0.8, "drums": 0.15, "vocals": 0.05, "piano": 0.0})
        self.assertEqual(ais.stem_shares({"a": 0}), {"a": 0.0})
        self.assertEqual([ais.solo_verdict(x)[0] for x in (0.8, 0.79, 0.5, 0.49, None)], ["solo", "mixed", "mixed", "poor", "none"])
        self.assertEqual(ais.others_text(shares, "guitar"), "드럼 15% · 보컬 5%")
        self.assertEqual(ais.others_text({"guitar": 1.0}, "guitar"), "없음")


if __name__ == "__main__":
    unittest.main()
