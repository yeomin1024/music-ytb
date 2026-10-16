# -*- coding: utf-8 -*-
# VERSION: v1.0 — 2026-10-16 — tools/music_analysis.py v1.0 시험: 트랙 수치의 방향(합성 신호), 멜로디 특징·구성, Claude 에게 보낼 글,
#          Claude 답 읽기 (GPU·네트워크 없음 — 트랙 수치 시험만 numpy 필요)
r"""
  python3 -m unittest tools.tests.test_music_analysis -v     (저장소 최상위에서)
"""
import json
import sys
import unittest
from fractions import Fraction as F
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
import music_analysis as ma  # noqa: E402

try:
    import numpy as np
except ImportError:  # numpy 가 없으면 트랙 수치 시험만 건너뜀
    np = None

SR = 44100


def _notes(freq, n=24, gap=0.8, dur=0.08):
    x = np.zeros(int(n * gap * SR) + SR)
    t = np.arange(int(dur * SR)) / SR
    tone = np.sin(2 * np.pi * freq * t) * np.exp(-t * 20)
    for k in range(n):
        s = int(k * gap * SR)
        x[s:s + len(tone)] += tone
    return x * 0.5


@unittest.skipIf(np is None, "numpy 없음")
class TestStemFeatures(unittest.TestCase):
    """합성 신호로 '방향'만 확인한다 (기준값 보정은 아님)."""

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.dry = _notes(440)

    def test_stereo_width(self):
        mono = ma.stem_features(np.stack([self.dry, self.dry]), SR)
        noise = 0.3 * self.rng.standard_normal(len(self.dry)) * (np.abs(self.dry) > 0.01)
        wide = ma.stem_features(np.stack([self.dry, np.roll(self.dry, int(0.012 * SR)) + noise]), SR)
        self.assertLess(mono["width"], 0.01)
        self.assertGreater(wide["width"], 0.15)
        self.assertEqual(ma.stem_words(mono)["스테레오"], "가운데(모노에 가까움)")
        self.assertEqual(ma.stem_words(wide)["스테레오"], "넓게 퍼짐(스테레오)")

    def test_reverb_tail(self):
        t = np.arange(int(1.5 * SR)) / SR
        ir = self.rng.standard_normal(len(t)) * np.exp(-t * 3)
        ir[0] = 1
        wet = np.convolve(self.dry, ir / (np.abs(ir).sum() / 20))[:len(self.dry)]
        wet *= 0.5 / np.abs(wet).max()
        dry_f, wet_f = ma.stem_features(self.dry, SR), ma.stem_features(wet, SR)
        self.assertLess(dry_f["tail"], 0.05)
        self.assertGreater(wet_f["tail"], dry_f["tail"] + 0.1)
        self.assertEqual(ma.stem_words(dry_f)["잔향·지속"], "짧고 건조")

    def test_dense_notes_tail_unknown(self):
        dense = _notes(220, n=60, gap=0.25, dur=0.2)
        f = ma.stem_features(dense, SR)
        self.assertIsNone(f["tail"])  # 다음 소리가 0.4초 안에 오면 잔향을 재지 않음
        self.assertIn("잴 수 없음", ma.stem_words(f)["잔향·지속"])

    def test_compression_and_brightness(self):
        t = np.arange(10 * SR) / SR
        clean = 0.5 * np.sin(2 * np.pi * 220 * t)
        squashed = np.clip(4 * clean, -0.5, 0.5)
        self.assertLess(ma.stem_features(squashed, SR)["crest_db"], ma.stem_features(clean, SR)["crest_db"])
        self.assertEqual(ma.stem_words(ma.stem_features(squashed, SR))["다이내믹"], "강하게 압축됨")
        dark, bright = ma.stem_features(_notes(110), SR), ma.stem_features(_notes(5000), SR)
        self.assertLess(dark["centroid_hz"], 1200)
        self.assertGreater(bright["centroid_hz"], 3000)

    def test_shares_and_silence(self):
        feats = {"vocals": ma.stem_features(self.dry, SR), "piano": ma.stem_features(0.1 * self.dry, SR),
                 "guitar": ma.stem_features(np.zeros(SR), SR)}
        ma.add_shares(feats)
        self.assertAlmostEqual(sum(f["share"] for f in feats.values()), 1.0, places=2)
        self.assertGreater(feats["vocals"]["share"], 0.9)
        self.assertTrue(feats["guitar"]["silent"])
        self.assertEqual(ma.stem_words(feats["guitar"]), {"비중": "거의 없음"})
        self.assertNotIn("거친 정도", ma.stem_words(feats["vocals"], "drums"))  # 드럼은 원래 소음 성분이 많아 안 씀

    def test_shapes(self):
        a = ma.stem_features(np.stack([self.dry, self.dry]), SR)
        b = ma.stem_features(np.stack([self.dry, self.dry]).T, SR)  # (N, 2)
        c = ma.stem_features(self.dry, SR)  # 모노
        self.assertEqual(a["centroid_hz"], b["centroid_hz"])
        self.assertEqual(a["centroid_hz"], c["centroid_hz"])


class TestMelody(unittest.TestCase):
    def test_melody_features(self):
        line = [(F(i, 2), p) for i, p in enumerate([60, 62, 64, 65, 67, 67, 65, 64, 62, 60, 72, 60])]
        f = ma.melody_features(line)
        self.assertEqual(f["notes"], 12)
        self.assertEqual((f["low"], f["high"]), ("C4", "G4"))  # 5%·95% 위치 (튀는 음 하나는 빼고)
        self.assertEqual(f["notes_per_beat"], 2.0)  # 8분음표
        self.assertEqual(f["repeat_ratio"], round(1 / 11, 2))
        self.assertGreater(f["step_ratio"], 0.6)
        w = ma.melody_words(f)
        self.assertEqual((w["진행"], w["리듬"]), ("순차 진행 위주", "8분음표 위주"))
        self.assertIsNone(ma.melody_features(line[:5]))
        self.assertEqual(ma.melody_words(None), {})

    def test_form_sections(self):
        abc = "X:1\n% intro\nV: Vocal\nZ|\n% verse\nV: Vocal\nC|\n% verse\nV: Vocal\nD|\n% chorus\nV: Vocal\nE|\n"
        self.assertEqual(ma.form_sections(abc), ["Intro", "Verse", "Chorus"])
        self.assertEqual(ma.form_sections(""), [])


ANALYSIS = {
    "librosa": {"seconds": 214.0, "tempo_bpm": 151.9},
    "sheetsage2": {"ok": True, "bpm": 152, "key": "A minor"},
    "tags": {"genres_evidence": [["Pop music", 0.61]], "moods_evidence": [["Exciting music", 0.33]],
             "instruments_evidence": [["Electric guitar", 0.52]]},
    "stems": {"vocals": {"share": 0.31, "centroid_hz": 2400.0, "width": 0.04, "crest_db": 12.0, "tail": 0.3,
                         "flatness": 0.05, "words": {"비중": "중심", "밝기": "중간 밝기", "스테레오": "약간 퍼짐", "다이내믹": "보통",
                                                     "잔향·지속": "보통", "거친 정도": "매끈한 편"}},
              "drums": {"share": 0.25, "centroid_hz": 3500.0, "width": 0.1, "crest_db": 8.0, "tail": None, "flatness": 0.4,
                        "words": {"비중": "중심", "밝기": "밝고 날카로운", "스테레오": "약간 퍼짐", "다이내믹": "강하게 압축됨",
                                  "잔향·지속": "잴 수 없음(소리가 촘촘함)"}},
              "piano": {"share": 0.01, "silent": False, "words": {"비중": "거의 없음"}}},
    "melody_features": {"range_semitones": 12, "step_ratio": 0.66, "leap_ratio": 0.1, "notes_per_beat": 2.0, "repeat_ratio": 0.2},
    "melody_words": {"음역": "보통 음역", "진행": "순차 진행 위주", "리듬": "8분음표 위주"},
    "form": ["Intro", "Verse", "Chorus", "Verse", "Chorus", "Outro"],
}


class TestClaudeRequest(unittest.TestCase):
    def test_with_lyrics(self):
        text = ma.claude_request(ANALYSIS, "비 오는 밤 버스 창가에서 듣고 싶은 노래", lyrics=True)
        for want in ("비 오는 밤 버스 창가", "사진을 첨부", "새 가사", "한국어", "보컬의 언어", "STYLE:", "SECTIONS:", "LYRICS:",
                     "약 152 BPM", "조성 A minor", "보컬: 비중 31%(중심)", "밝기 2400Hz", "잔향·지속 0.3(보통)",
                     "음역 12반음", "원곡 구성: Intro, Verse, Chorus", "원곡 제목·가수·가사·멜로디를 쓰지 말고", "Pop music 0.61"):
            self.assertIn(want, text, want)
        self.assertIn("피아노: 비중 1%(거의 없음)", text)  # 거의 없는 트랙은 비중만
        self.assertNotIn("피아노: 비중 1%(거의 없음) · 밝기", text)

    def test_instrumental(self):
        text = ma.claude_request(ANALYSIS, "", lyrics=False)
        self.assertIn("보컬·노래 관련 낱말은 넣지 않기", text)
        self.assertIn("\"없음\" 한 단어", text)
        self.assertIn("(여기에 멘트를 적어 주세요)", text)
        self.assertNotIn("새 가사", text)

    def test_without_stems(self):
        text = ma.claude_request({"librosa": {"seconds": 10, "tempo_bpm": 90}, "stems": {}, "stems_error": "No module named 'demucs'"},
                                 "x", lyrics=False)
        self.assertIn("트랙 분리 못 함", text)
        self.assertIn("약 90 BPM", text)


ANSWER = """좋아요! 아래처럼 정했어요.

```
STYLE: J-pop band, bittersweet rainy night, reverb-drenched clean electric guitar, warm electric bass, tight drums, airy strings, minor key, 148 BPM, Korean, warm breathy female vocal
SECTIONS: Intro, Verse, Chorus, Verse, Chorus, Outro
LYRICS:
[Intro]

[verse]
빗물이 번진 버스 창가에
너의 이름을 적어 보다가

[Chorus]
멈추지 마 이 밤이 끝나도
[Verse]
젖은 신호등 아래서
[Chorus]
멈추지 마 이 밤이 끝나도
[Outro]
```
사진의 푸른 색감과 비 오는 밤을 반영했어요.
"""


class TestParseAnswer(unittest.TestCase):
    def test_lyrics_answer(self):
        result, errors, warnings = ma.parse_claude_answer(ANSWER, want_lyrics=True)
        self.assertEqual(errors, [])
        self.assertTrue(result["style"].startswith("J-pop band, bittersweet"))
        self.assertEqual(result["sections"], "Intro,Verse,Chorus,Verse,Chorus,Outro")
        lines = result["lyrics"].splitlines()
        self.assertEqual(lines[0], "[Intro]")
        self.assertIn("[Verse]", lines)  # [verse] → [Verse]
        self.assertIn("", lines[lines.index("[Chorus]") - 1:lines.index("[Chorus]")])  # 태그 앞 빈 줄
        self.assertNotIn("사진의 푸른", result["lyrics"])  # 코드 블록 밖 설명은 안 들어감
        self.assertEqual(warnings, [])

    def test_instrumental_answer(self):
        text = "```\nSTYLE: lo-fi piano, 80 BPM\nSECTIONS: Intro, Verse, Chorus, Outro\nLYRICS:\n없음\n```"
        result, errors, warnings = ma.parse_claude_answer(text, want_lyrics=False)
        self.assertEqual((errors, warnings, result["lyrics"]), ([], [], ""))
        result, errors, warnings = ma.parse_claude_answer(ANSWER, want_lyrics=False)
        self.assertEqual(errors, [])
        self.assertIn("가사 넣기를 끈 상태라", warnings[0])

    def test_errors(self):
        _, errors, _ = ma.parse_claude_answer("그냥 잡담", want_lyrics=False)
        self.assertTrue(any("STYLE" in e for e in errors) and any("SECTIONS" in e for e in errors))
        _, errors, _ = ma.parse_claude_answer("STYLE: x\nSECTIONS: Intro, Pre-Chorus\nLYRICS:\n없음", want_lyrics=False)
        self.assertTrue(any("Pre-Chorus" in e for e in errors))
        _, errors, _ = ma.parse_claude_answer("STYLE: x\nSECTIONS: Verse\nLYRICS:\n그냥 가사만 있고 태그 없음", want_lyrics=True)
        self.assertTrue(any("태그" in e for e in errors))

    def test_warnings(self):
        text = "STYLE: x\nSECTIONS: Verse, Chorus\nLYRICS:\n[Chorus]\n가\n[Verse]\n나"
        _, errors, warnings = ma.parse_claude_answer(text, want_lyrics=True)
        self.assertEqual(errors, [])
        self.assertTrue(any("순서" in w for w in warnings))
        long_text = "STYLE: x\nSECTIONS: Verse\nLYRICS:\n[Verse]\n" + "\n".join(["라"] * 45)
        self.assertTrue(any("45줄" in w for w in ma.parse_claude_answer(long_text, want_lyrics=True)[2]))

    def test_round_trip_request_format(self):
        """claude_request 가 보여 주는 답 형식 예시를 그대로 읽을 수 있어야 한다 (형식이 서로 어긋나지 않게)."""
        req = ma.claude_request(ANALYSIS, "x", lyrics=True)
        example = req.split("```")[1]
        filled = example.replace("STYLE: ...", "STYLE: pop").replace("SECTIONS: Intro, Verse, Chorus, ...",
                                                                      "SECTIONS: Verse, Chorus").replace("...", "가사")
        result, errors, _ = ma.parse_claude_answer(filled, want_lyrics=True)
        self.assertEqual(errors, [])
        self.assertEqual(result["lyrics"], "[Verse]\n가사\n\n[Chorus]\n가사\n")


if __name__ == "__main__":
    unittest.main()
