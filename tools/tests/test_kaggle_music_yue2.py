# -*- coding: utf-8 -*-
# VERSION: v1.2 — 2026-10-15 — kaggle_music_yue2.py v1.2 시험 추가: 참고 곡 분석(조성·태그·프롬프트), 멜로디 검사(같은 멜로디·조옮김·
#          박자 표기·구간 복사는 '검토 필요', 무작위 멜로디는 통과), 올리기(원곡·원곡 악보는 절대 안 올림, 로컬 git 원격으로 push 흐름)
#          (v1.1: music-ytb 로 옮김 · v1.0: 입력·T4 패치·환경 — 가짜 모듈·가짜 nvidia-smi, GPU·네트워크·모델 없음)
r"""
  python3 -m unittest tools.tests.test_kaggle_music_yue2 -v     (저장소 최상위에서)

- GPU·네트워크·모델·GitHub 없이 돈다. 올리기 시험은 임시 폴더의 로컬 git 원격(bare)만 쓴다.
"""
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import types
import unittest
from fractions import Fraction as F
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
import kaggle_music_yue2 as km  # noqa: E402


def fake_modules():
    """공식 yue2.cuda_graph.GraphAR·yue2.nar.synthesize 와 같은 인자 모양의 가짜."""
    calls = {"graph": [], "nar": []}

    class GraphAR:
        def __init__(self, model, prefixes, max_tokens, *, capture=True, attention_backend="auto", fuse_projections=False):
            calls["graph"].append(attention_backend)

    def synthesize(model, prefix, codec, seed, steps=32, context=24576, attention="sdpa", offload_ar=False,
                   cancelled=None, query_chunk_size=None, on_progress=None):
        calls["nar"].append(query_chunk_size)
        return "latents"

    return types.SimpleNamespace(GraphAR=GraphAR), types.SimpleNamespace(synthesize=synthesize), calls


# 멜로디 시험용: 단조로운(우연히 겹치기 쉬운) 무작위 팝 멜로디 — 구간 반복 포함
MAJOR = [0, 2, 4, 5, 7, 9, 11]
STEPS = [(0, 30), (1, 25), (-1, 25), (2, 8), (-2, 8), (3, 2), (-3, 2)]
DURS = [(F(1, 2), 60), (F(1), 30), (F(1, 4), 5), (F(2), 5)]


def _pick(rng, table):
    return rng.choices([x for x, _ in table], [w for _, w in table])[0]


def random_song(rng, tonic=60):
    def phrase(n=14):
        d, notes = rng.randint(0, 6), []
        for _ in range(n):
            d = max(-3, min(11, d + _pick(rng, STEPS)))
            notes.append((d, _pick(rng, DURS)))
        return notes
    a, b, c = phrase(), phrase(), phrase()
    t, line = F(0), []
    for ph in (a, b, a, c, c, b, a):
        for d, dur in ph:
            line.append((t, tonic + 12 * (d // 7) + MAJOR[d % 7]))
            t += dur
        t += F(1)
    return line


def score_of(line, voice="Vocal"):
    """melody_line 이 읽는 abc_tools Score 모양의 가짜."""
    notes = [[t, p, F(1, 2)] for t, p in line]
    voices = {"Vocal": types.SimpleNamespace(notes=notes if voice == "Vocal" else [], key="C"),
              "Ins": types.SimpleNamespace(notes=notes if voice == "Ins" else [], key="C")}
    return types.SimpleNamespace(voices=voices, bpm=120)


class TestInputs(unittest.TestCase):
    def test_seeds(self):
        self.assertEqual(km.parse_seeds("1, 2,3"), [1, 2, 3])
        for bad in ("", "1,1", "-1", "a", str(2**63)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                km.parse_seeds(bad)

    def test_resolve_seeds(self):
        self.assertEqual(km.resolve_seeds("7,9", 4, True), [7, 9])
        self.assertEqual(km.resolve_seeds(None, None, True), [1, 2, 3, 4])  # 참고 곡이면 기본 4곡
        self.assertEqual(km.resolve_seeds(None, None, False), [1])
        self.assertEqual(km.resolve_seeds(None, 3, False), [1, 2, 3])
        for bad in (0, 65):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                km.resolve_seeds(None, bad, True)

    def test_sections_match_official_default(self):
        # 공식 instrumental.py plan_with_model 의 기본 계획 가사
        self.assertEqual(km.section_lyrics(km.DEFAULT_SECTIONS), "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n")
        self.assertEqual(km.section_lyrics(" Intro , Verse 2 "), "[Intro]\n\n[Verse 2]\n")
        for bad in ("", "인트로", "Verse]"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                km.section_lyrics(bad)

    def test_instrumental_style_like_official_prepare(self):
        self.assertEqual(km.instrumental_style("calm lo-fi piano, 80 BPM."),
                         "Instrumental, calm lo-fi piano, 80 BPM, no vocals, no singing, no choir, no spoken words.")
        self.assertEqual(km.instrumental_style("Instrumental, piano, no vocals"),
                         "Instrumental, piano, no vocals, no singing, no choir, no spoken words.")
        self.assertTrue(km.instrumental_style("").startswith("Instrumental, Expressive instrumental music"))

    def test_assign_seeds(self):
        self.assertEqual(km.assign_seeds([1, 2, 3], [0, 1]), {0: [1, 3], 1: [2]})
        self.assertEqual(km.assign_seeds([5], [0, 1]), {0: [5]})
        with self.assertRaises(ValueError):
            km.assign_seeds([1], [])

    def test_fresh_dir_never_overwrites(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "seed-1"
            self.assertEqual(km.fresh_dir(p), p)
            p.mkdir()
            self.assertEqual(km.fresh_dir(p).name, "seed-1-2")

    def test_find_reference(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(FileNotFoundError):
                km.find_reference("auto", root)
            (root / "ds").mkdir()
            (root / "ds" / "notes.txt").write_text("x")
            song = root / "ds" / "My Song.MP3"
            song.write_bytes(b"x")
            self.assertEqual(km.find_reference("auto", root), song)  # 확장자 대소문자 무관, 오디오 아닌 파일 무시
            (root / "ds" / "b.wav").write_bytes(b"x")
            with self.assertRaises(ValueError):
                km.find_reference("auto", root)
            self.assertEqual(km.find_reference(str(song)), song)
            with self.assertRaises(FileNotFoundError):
                km.find_reference(str(root / "none.mp3"))


class TestAnalysis(unittest.TestCase):
    def test_estimate_key(self):
        # 프로필 그 자체 → 그 조성
        c = km.estimate_key(list(km.KK_MAJOR))
        self.assertEqual((c["tonic"], c["mode"]), ("C", "major"))
        a_minor = [km.KK_MINOR[(i - 9) % 12] for i in range(12)]
        k = km.estimate_key(a_minor)
        self.assertEqual((k["tonic"], k["mode"]), ("A", "minor"))
        self.assertGreater(k["margin"], 0)
        # 실제 음 분포: G 장조 음계 음만
        g_major = [1.0 if pc in {7, 9, 11, 0, 2, 4, 6} else 0.05 for pc in range(12)]
        g_major[7] = 2.0
        self.assertEqual(km.estimate_key(g_major)["tonic"], "G")
        with self.assertRaises(ValueError):
            km.estimate_key([1.0] * 11)

    def test_abc_key_mode(self):
        self.assertEqual(km.abc_key_mode("F#m"), ("F#", "minor"))
        self.assertEqual(km.abc_key_mode("Bb"), ("Bb", "major"))
        self.assertEqual(km.abc_key_mode(""), (None, None))

    def test_select_tags(self):
        probs = {"Pop music": 0.6, "Rock music": 0.3, "Jazz": 0.16, "Exciting music": 0.4, "Happy music": 0.2,
                 "Electric guitar": 0.5, "Cello": 0.3, "String section": 0.25, "Piano": 0.12, "Flute": 0.05,
                 "Singing": 0.9, "Music": 0.99}
        tags = km.select_tags(probs)
        self.assertEqual(tags["genres"], ["pop", "rock"])  # 최대 2개
        self.assertEqual(tags["moods"], ["energetic and exciting"])  # 최대 1개
        self.assertEqual(tags["instruments"], ["electric guitar", "strings", "piano"])  # Cello·String section → strings 한 번, Flute 은 기준 미만
        self.assertEqual(tags["instruments_evidence"][1], ("Cello", 0.3))
        self.assertNotIn("Singing", json.dumps(tags))  # 목소리 태그는 프롬프트에 안 들어감

    def test_build_prompt(self):
        tags = {"genres": ["pop", "rock"], "moods": ["energetic and exciting"], "instruments": ["electric guitar", "strings"]}
        self.assertEqual(km.build_prompt(tags, 151.7, "minor", "J-pop band."),
                         "J-pop band, rock, energetic and exciting, electric guitar, strings, minor key, 152 BPM")  # 'pop' 은 extra 에 있음
        self.assertEqual(km.build_prompt({}, 0, None), "")
        self.assertEqual(km.build_prompt({}, 90, "major"), "major key, 90 BPM")


class TestMelody(unittest.TestCase):
    def test_melody_line_skyline(self):
        vocal = types.SimpleNamespace(notes=[[F(0), 60, F(1)], [F(1), 62, F(1)]])
        ins = types.SimpleNamespace(notes=[[F(0), 67, F(1)], [F(2), 55, F(1)]])
        score = types.SimpleNamespace(voices={"Vocal": vocal, "Ins": ins})
        self.assertEqual(km.melody_line(score), [(F(0), 67), (F(1), 62), (F(2), 55)])

    def test_tokens_invariant_to_key_and_tempo_notation(self):
        line = [(F(0), 60), (F(1, 2), 62), (F(1), 64), (F(2), 60)]
        moved = [(t * 2, p + 7) for t, p in line]  # 5도 위 + 두 배 박자 표기
        self.assertEqual(km.melody_tokens(line), km.melody_tokens(moved))
        self.assertEqual(km.melody_tokens(line), ([2, 2, -4], [None, F(1), F(2)]))

    def test_longest_run_needs_moving_notes(self):
        a = [0, 0, 0, 0, 0, 0, 1, 2]
        b = [5, 0, 0, 0, 0, 0, 0, 9]
        self.assertEqual(km.longest_run(a, b)[0], 6)
        self.assertEqual(km.longest_run(a, b, moving=lambda t: t != 0)[0], 0)  # 같은 음 반복만인 구간은 안 셈
        c = [1, 2, -1, 2, 3, 0]
        self.assertEqual(km.longest_run(c, [9] + c, moving=lambda t: t != 0), (6, 6, 7))

    def test_copies_are_flagged(self):
        rng = random.Random(3)
        base, other = random_song(rng, 60), random_song(rng, 62)
        cases = {
            "같은 멜로디": base,
            "조옮김 + 두 배 박자 표기": [(t * 2, p + 5) for t, p in base],
        }
        seg = base[20:30]  # 10음 구간을 다른 곡 가운데에 끼움
        off = other[60][0] - seg[0][0]
        tail_off = seg[-1][0] + off + F(1) - other[60][0]
        cases["10음 구간 복사"] = other[:60] + [(t + off, p) for t, p in seg] + [(t + tail_off, p) for t, p in other[60:]]
        t, rerhythm = F(0), []
        for _, p in base:  # 음정 그대로, 리듬만 새로
            rerhythm.append((t, p))
            t += _pick(rng, DURS)
        cases["리듬만 바꿈"] = rerhythm
        for label, gen in cases.items():
            with self.subTest(label=label):
                m = km.compare_melodies(base, gen, 120, 120)
                self.assertEqual(km.judge(m), "review", m)

    def test_random_melodies_pass(self):
        rng = random.Random(42)
        songs = [random_song(rng, rng.choice([55, 57, 60, 62, 64])) for _ in range(12)]
        flagged = [(i, j) for i in range(12) for j in range(i + 1, 12)
                   if km.judge(km.compare_melodies(songs[i], songs[j], 120, 120)) != "pass"]
        self.assertEqual(flagged, [])  # 단조로운 무작위 멜로디 66쌍 — 우연히 '검토 필요'가 나오면 안 됨

    def test_compare_positions_and_short(self):
        self.assertIsNone(km.compare_melodies([(F(0), 60), (F(1), 62)], [(F(0), 60)] * 5, 120, 120))
        line = random_song(random.Random(5))
        m = km.compare_melodies(line, line, 120, 60)
        self.assertEqual(m["pitch_ref_at_s"], 0.0)
        self.assertEqual(m["run_pitch"], len(line) - 1)

    def test_judge(self):
        self.assertEqual(km.judge(None, has_ref=False), "no_ref")
        self.assertEqual(km.judge(None), "unchecked")
        self.assertEqual(km.judge(dict(run_rhythm=km.FLAG_RUN_RHYTHM - 1, run_pitch=km.FLAG_RUN_PITCH - 1)), "pass")
        self.assertEqual(km.judge(dict(run_rhythm=km.FLAG_RUN_RHYTHM, run_pitch=0)), "review")
        self.assertEqual(km.judge(dict(run_rhythm=0, run_pitch=km.FLAG_RUN_PITCH)), "review")


def make_name_dir(root, ref_line, gens):
    """가짜 결과 폴더: reference/(ref.wav·받아 적은 악보) + seed-*/(score·mp3·flac·기록). 악보는 '시각:음' 줄 글로."""
    name_dir = Path(root) / "bgm-test"
    tdir = name_dir / "reference" / "transcription"
    tdir.mkdir(parents=True)
    (name_dir / "reference" / "ref.wav").write_bytes(b"RIFF-original")
    as_text = lambda line: "\n".join(f"{t}:{p}" for t, p in line)  # noqa: E731
    if ref_line is not None:
        (tdir / "score.abc").write_text(as_text(ref_line), encoding="utf-8")
    for seed, line in gens.items():
        d = name_dir / seed
        d.mkdir()
        (d / "score.abc").write_text(as_text(line), encoding="utf-8")
        for f in ("audio.mp3", "audio.flac"):
            (d / f).write_bytes(b"audio")
        (d / "kaggle_run.json").write_text(json.dumps({"metrics": {"seconds": 125.0}}), encoding="utf-8")
    (name_dir / "analysis.json").write_text(json.dumps({"prompt": {"style": "pop, piano, 120 BPM"},
                                                        "reference": {"sha256": "ab" * 32}}), encoding="utf-8")
    (name_dir / "prompt.txt").write_text("pop, piano, 120 BPM\n", encoding="utf-8")
    return name_dir


def fake_parse_abc(text):
    line = [(F(t), int(p)) for t, p in (row.split(":") for row in text.splitlines())]
    return score_of(line)


class TestSimilarityAndPush(unittest.TestCase):
    def setUp(self):
        rng = random.Random(7)
        self.ref = random_song(rng, 60)
        self.gens = {"seed-1": random_song(rng, 62), "seed-2": [(t, p + 2) for t, p in self.ref]}

    def test_run_similarity_verdicts(self):
        with tempfile.TemporaryDirectory() as d:
            name_dir = make_name_dir(d, self.ref, self.gens)
            res = km.run_similarity(name_dir, parse_abc=fake_parse_abc)
            self.assertEqual(res["seed-1"]["verdict"], "pass")
            self.assertEqual(res["seed-2"]["verdict"], "review")  # 조옮김한 원곡 멜로디
            saved = json.loads((name_dir / "seed-2" / "similarity.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["verdict_ko"], "검토 필요")
            self.assertEqual(saved["baseline_between_generated"]["pairs"], 1)

    def test_run_similarity_without_transcription(self):
        with tempfile.TemporaryDirectory() as d:
            name_dir = make_name_dir(d, None, self.gens)  # 원곡 받아 적기 실패
            res = km.run_similarity(name_dir, parse_abc=fake_parse_abc)
            self.assertEqual({r["verdict"] for r in res.values()}, {"unchecked"})

    def test_publish_files_never_includes_reference(self):
        with tempfile.TemporaryDirectory() as d:
            name_dir = make_name_dir(d, self.ref, self.gens)
            res = km.run_similarity(name_dir, parse_abc=fake_parse_abc)
            files = km.publish_files(name_dir, res, set(km.PUSH_DEFAULT))
            rels = sorted(str(rel) for _, rel in files)
            self.assertIn("results/bgm-test/seed-1/audio.mp3", rels)
            self.assertIn("results/bgm-test/analysis.json", rels)
            self.assertFalse(any("seed-2" in r for r in rels))  # 검토 필요 → 안 올림
            self.assertFalse(any("reference" in r or r.endswith(".wav") or r.endswith(".flac") for r in rels))
            everything = km.publish_files(name_dir, res, {"pass", "review", "unchecked", "no_ref"}, include_flac=True)
            self.assertFalse(any("reference" in str(src) for src, _ in everything))  # 다 허용해도 원곡은 없음
            self.assertTrue(any(str(rel).endswith("seed-2/audio.flac") for _, rel in everything))

    def test_readme(self):
        with tempfile.TemporaryDirectory() as d:
            name_dir = make_name_dir(d, self.ref, self.gens)
            res = km.run_similarity(name_dir, parse_abc=fake_parse_abc)
            for rec in res.values():
                rec["seconds"] = 125.0
            text = km.results_readme("bgm-test", res, set(km.PUSH_DEFAULT),
                                     json.loads((name_dir / "analysis.json").read_text(encoding="utf-8")))
            self.assertIn("[mp3](seed-1/audio.mp3)", text)
            self.assertIn("검토 필요 — 올리지 않음", text)
            self.assertIn("2:05", text)
            self.assertNotIn("ref.wav", text)

    def test_git_env_keeps_token_out_of_files(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(km, "WORK", Path(d)), \
                mock.patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_secretvalue"}):
            env, has = km.git_env()
            self.assertTrue(has)
            self.assertNotIn("ghp_secretvalue", Path(env["GIT_ASKPASS"]).read_text())
            self.assertEqual(env["GIT_TERMINAL_PROMPT"], "0")

    def test_worker_env_drops_token(self):
        with mock.patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_x", "PYTHONPATH": "/kaggle/lib"}):
            env = km.worker_env(0)
        self.assertNotIn("GITHUB_TOKEN", env)
        self.assertNotIn("PYTHONPATH", env)

    def test_push_check_needs_token(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(km, "WORK", Path(d)), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GITHUB_TOKEN", None)
            with self.assertRaises(SystemExit):
                km.push_check()

    def test_push_results_to_local_remote(self):
        """진짜 GitHub 대신 임시 bare 저장소 — 복사·커밋·pull --rebase·push 흐름과 '원곡 안 올림' 확인."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            run = lambda *a, cwd=None: subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True)  # noqa: E731
            run("init", "-q", "--bare", "-b", "main", str(d / "origin.git"))
            run("clone", "-q", str(d / "origin.git"), str(d / "clone"))
            (d / "clone" / "README.md").write_text("x")
            run("-c", "user.name=t", "-c", "user.email=t@t", "add", "README.md", cwd=d / "clone")
            run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init", cwd=d / "clone")
            run("push", "-q", "origin", "HEAD:main", cwd=d / "clone")
            # 다른 곳에서 먼저 올라간 커밋 (pull --rebase 가 필요)
            run("clone", "-q", str(d / "origin.git"), str(d / "other"))
            (d / "other" / "other.txt").write_text("y")
            run("-c", "user.name=t", "-c", "user.email=t@t", "add", "other.txt", cwd=d / "other")
            run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "other", cwd=d / "other")
            run("push", "-q", "origin", "HEAD:main", cwd=d / "other")
            name_dir = make_name_dir(d / "out", self.ref, self.gens)
            res = km.run_similarity(name_dir, parse_abc=fake_parse_abc)
            env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
            with mock.patch.object(km, "ROOT", d / "clone"), mock.patch.object(km, "WORK", d / "work"), \
                    mock.patch.dict(os.environ, env, clear=True), mock.patch.object(km.time, "sleep"):
                rc = km.push_results(name_dir, res, set(km.PUSH_DEFAULT))
            self.assertEqual(rc, 0)
            tree = run("ls-tree", "-r", "--name-only", "main", cwd=d / "origin.git").stdout.split()
            self.assertIn("results/bgm-test/seed-1/audio.mp3", tree)
            self.assertIn("results/bgm-test/README.md", tree)
            self.assertIn("other.txt", tree)
            self.assertFalse(any("seed-2" in p or "reference" in p or p.endswith(".wav") for p in tree))


class TestT4Patches(unittest.TestCase):
    def test_graph_attention_forced_to_sdpa(self):
        cg, nar, calls = fake_modules()
        applied = km.apply_t4_patches(cg, nar, 256)
        self.assertEqual(applied, ["graph_attention=sdpa", "nar_query_chunk=256"])
        cg.GraphAR("model", [[1, 2]], 10)                       # 공식 호출 모양 (auto)
        cg.GraphAR("model", [[1, 2]], 10, attention_backend="flash")
        self.assertEqual(calls["graph"], ["sdpa", "sdpa"])

    def test_nar_chunk_only_when_caller_did_not_set(self):
        cg, nar, calls = fake_modules()
        km.apply_t4_patches(cg, nar, 256)
        # pipeline.synthesize 와 같은 키워드 호출 (query_chunk_size 없음)
        self.assertEqual(nar.synthesize("m", [1], [2], 7, steps=32, context=24576, offload_ar=True,
                                        cancelled=None, on_progress=None), "latents")
        nar.synthesize("m", [1], [2], 7, query_chunk_size=64)
        self.assertEqual(calls["nar"], [256, 64])

    def test_reapply_does_not_stack(self):
        cg, nar, calls = fake_modules()
        km.apply_t4_patches(cg, nar, 256)
        km.apply_t4_patches(cg, nar, 128)
        nar.synthesize("m", [1], [2], 7)
        cg.GraphAR("model", [[1]], 5)
        self.assertEqual(calls["nar"], [128])
        self.assertEqual(calls["graph"], ["sdpa"])
        self.assertFalse(getattr(nar.synthesize.__wrapped__, "_kaggle_t4", False))

    def test_bad_chunk(self):
        cg, nar, _ = fake_modules()
        for bad in (0, -1, True, 1.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                km.apply_t4_patches(cg, nar, bad)


class TestEnv(unittest.TestCase):
    def test_query_gpus_parses_nvidia_smi(self):
        out = "0, Tesla T4, 15360, 560.35.03, 7.5\n1, Tesla T4, 15360, 560.35.03, 7.5\n2, X, 100, 1, [N/A]\n"
        done = subprocess.CompletedProcess([], 0, stdout=out, stderr="")
        with mock.patch.object(km.subprocess, "run", return_value=done):
            rows, err = km.query_gpus()
        self.assertIsNone(err)
        self.assertEqual([(r["index"], r["cap"], r["mem_mib"]) for r in rows], [(0, (7, 5), 15360), (1, (7, 5), 15360), (2, None, 100)])

    def test_query_gpus_missing_nvidia_smi(self):
        with mock.patch.object(km.subprocess, "run", side_effect=FileNotFoundError("nvidia-smi")):
            rows, err = km.query_gpus()
        self.assertEqual(rows, [])
        self.assertIn("nvidia-smi", err)

    def test_setup_stops_on_p100_before_installing(self):
        p100 = [dict(index=0, name="Tesla P100-PCIE-16GB", mem_mib=16384, driver="560", cap=(6, 0))]
        with mock.patch.object(km, "query_gpus", return_value=(p100, None)), \
             mock.patch.object(km, "run_cmd") as run_cmd, self.assertRaises(SystemExit):
            km.setup()
        run_cmd.assert_not_called()

    def test_setup_stops_without_gpu(self):
        with mock.patch.object(km, "query_gpus", return_value=([], "no device")), \
             mock.patch.object(km, "run_cmd") as run_cmd, self.assertRaises(SystemExit):
            km.setup()
        run_cmd.assert_not_called()

    def test_worker_env_isolated(self):
        with mock.patch.dict(km.os.environ, {"PYTHONPATH": "/kaggle/lib", "HF_TOKEN": "x"}):
            env = km.worker_env(1)
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")
        self.assertEqual(env["HF_TOKEN"], "x")  # 모델이 동의 후 다운로드(gated)면 필요 — 그대로 넘김

    def test_worker_cmd(self):
        args = km.argparse.Namespace(mode="song", style="pop", name="s1", out="/o", sections=km.DEFAULT_SECTIONS,
                                     plan_mode="full", cot="full", backend="torch", chunk=256, offload_ar="auto",
                                     model_rev=km.MODEL_REV, vae_rev=km.VAE_REV, log_level="INFO",
                                     lyrics_file="lyrics.txt", quiet=False)
        cmd = km.worker_cmd(args, [3, 5], fetch_only=True)
        self.assertEqual(cmd[2], "_worker")
        self.assertIn("3,5", cmd)
        self.assertIn("--fetch-only", cmd)
        self.assertTrue(cmd[cmd.index("--lyrics-file") + 1].endswith("lyrics.txt"))

    def test_run_flow_with_reference(self):
        """run 전체 순서 — 가짜 분석·가짜 생성 프로세스: 분석 프롬프트가 생성에 쓰이고, 시드가 GPU 마다 나뉘고, 검사 뒤 올리기."""
        rng = random.Random(11)
        ref_line = random_song(rng)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "input").mkdir()
            (d / "input" / "fav.mp3").write_bytes(b"x")
            name_dir = d / "out" / "bgm-x"
            seen = {"cmds": []}

            def fake_streamed(cmd, env, prefix):
                seen["cmds"].append((prefix, cmd, env.get("CUDA_VISIBLE_DEVICES")))
                if "_analyze" in cmd:
                    (name_dir / "reference" / "transcription").mkdir(parents=True)
                    (name_dir / "reference" / "transcription" / "score.abc").write_text(
                        "\n".join(f"{t}:{p}" for t, p in ref_line), encoding="utf-8")
                    (name_dir / "analysis.json").write_text(json.dumps({"prompt": {"style": "pop, piano, strings, 150 BPM"}}),
                                                            encoding="utf-8")
                return 0

            class FakeProc:
                def __init__(self, cmd, env=None, **kw):
                    seen["cmds"].append(("worker", cmd, env.get("CUDA_VISIBLE_DEVICES")))
                    seeds = cmd[cmd.index("--seeds") + 1].split(",")
                    for s in seeds:  # seed 1 은 원곡 멜로디 그대로(검토 필요), 나머지는 새 멜로디
                        sd = name_dir / f"seed-{s}"
                        sd.mkdir(parents=True)
                        line = ref_line if s == "1" else random_song(random.Random(int(s)), 62)
                        (sd / "score.abc").write_text("\n".join(f"{t}:{p}" for t, p in line), encoding="utf-8")
                        (sd / "audio.flac").write_bytes(b"a")
                        (sd / "audio.mp3").write_bytes(b"a")
                        (sd / "kaggle_run.json").write_text("{}", encoding="utf-8")
                    self.stdout = iter(["[GEN] ok\n"])

                def wait(self):
                    return 0

            gpus = [dict(index=0, name="T4", mem_mib=15360, driver="560", cap=(7, 5)),
                    dict(index=1, name="T4", mem_mib=15360, driver="560", cap=(7, 5))]
            original_similarity = km.run_similarity
            sim = lambda nd: original_similarity(nd, parse_abc=fake_parse_abc)  # noqa: E731
            with mock.patch.object(km, "setup", return_value=gpus), \
                    mock.patch.object(km, "push_check") as push_check, \
                    mock.patch.object(km, "run_streamed", side_effect=fake_streamed), \
                    mock.patch.object(km.subprocess, "Popen", FakeProc), \
                    mock.patch.object(km, "run_similarity", side_effect=sim), \
                    mock.patch.object(km, "push_results", return_value=0) as push_results, \
                    mock.patch.object(km, "setup_logging"):
                with mock.patch.object(km.find_reference, "__defaults__", (d / "input",)):
                    rc = km.main(["run", "--ref", "auto", "--push", "--name", "bgm-x", "--out", str(d / "out")])
            self.assertEqual(rc, 0)
            push_check.assert_called_once()
            workers = [(c, g) for p, c, g in seen["cmds"] if p == "worker"]
            self.assertEqual(sorted((g, c[c.index("--seeds") + 1]) for c, g in workers), [("0", "1,3"), ("1", "2,4")])
            self.assertTrue(all(c[c.index("--style") + 1] == "pop, piano, strings, 150 BPM" for c, _ in workers))
            results, allow = push_results.call_args[0][1], push_results.call_args[0][2]
            self.assertEqual(allow, set(km.PUSH_DEFAULT))
            self.assertEqual(results["seed-1"]["verdict"], "review")
            self.assertEqual({results[s]["verdict"] for s in ("seed-2", "seed-3", "seed-4")}, {"pass"})

    def test_same_name_other_reference_stops(self):
        """v1.3: 같은 --name 에 다른 곡이면 예전 분석을 쓰지 않고 멈춤 · 같은 곡이면 분석을 건너뜀."""
        gpus = [dict(index=0, name="T4", mem_mib=15360, driver="560", cap=(7, 5))]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ref = d / "song.mp3"
            ref.write_bytes(b"new song")
            name_dir = d / "out" / "bgm-x"
            name_dir.mkdir(parents=True)
            argv = ["run", "--ref", str(ref), "--name", "bgm-x", "--out", str(d / "out")]
            (name_dir / "analysis.json").write_text(json.dumps({"reference": {"sha256": "0" * 64}, "prompt": {"style": "x"}}))
            with mock.patch.object(km, "setup", return_value=gpus), mock.patch.object(km, "setup_logging"), \
                    mock.patch.object(km, "run_streamed", return_value=1) as streamed:
                with self.assertRaises(SystemExit):
                    km.main(argv)
                streamed.assert_not_called()  # 분석도 생성도 시작하지 않음
                (name_dir / "analysis.json").write_text(json.dumps({"reference": {"sha256": km.sha256_file(ref)},
                                                                    "prompt": {"style": "pop"}}))
                with self.assertRaises(SystemExit):  # 여기서는 가짜 모델 받기가 실패해서 멈춤
                    km.main(argv)
                self.assertEqual([c.args[2] for c in streamed.call_args_list], ["fetch"])  # 분석은 건너뜀

    def test_analyze_command_writes_claude_request(self):
        """v1.4: analyze = 분석(가짜) → claude_request.txt (멘트·가사 여부 반영) → 출력. 생성은 하지 않는다."""
        gpus = [dict(index=1, name="T4", mem_mib=15360, driver="560", cap=(7, 5))]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ref = d / "fav.mp3"
            ref.write_bytes(b"song")
            comment = d / "c.txt"
            comment.write_text("비 오는 밤 버스 창가", encoding="utf-8")
            name_dir = d / "out" / "my-01"

            def fake_streamed(cmd, env, prefix):
                self.assertEqual((prefix, env["CUDA_VISIBLE_DEVICES"]), ("analyze", "1"))
                name_dir.mkdir(parents=True)
                (name_dir / "analysis.json").write_text(json.dumps({
                    "reference": {"sha256": km.sha256_file(ref)}, "librosa": {"seconds": 200, "tempo_bpm": 120},
                    "stems": {"vocals": {"share": 0.4, "centroid_hz": 2000.0, "width": 0.05, "crest_db": 12, "tail": 0.2,
                                         "words": {"비중": "중심"}}}, "prompt": {"style": "pop"}}), encoding="utf-8")
                return 0

            out = io.StringIO()
            with mock.patch.object(km, "setup", return_value=gpus), mock.patch.object(km, "setup_logging"), \
                    mock.patch.object(km, "run_streamed", side_effect=fake_streamed) as streamed, \
                    mock.patch.object(km, "worker", side_effect=AssertionError("생성하면 안 됨")), \
                    mock.patch("sys.stdout", out):
                rc = km.main(["analyze", "--ref", str(ref), "--name", "my-01", "--out", str(d / "out"),
                              "--comment-file", str(comment), "--lyrics"])
                self.assertEqual(rc, 0)
                km.main(["analyze", "--ref", str(ref), "--name", "my-01", "--out", str(d / "out")])  # 두 번째는 분석을 건너뜀
            self.assertEqual(streamed.call_count, 1)
            text = (name_dir / "claude_request.txt").read_text(encoding="utf-8")
            printed = out.getvalue()
            self.assertIn(km.CLAUDE_BEGIN, printed)
            self.assertIn("(여기에 멘트를 적어 주세요)", text)  # 두 번째 실행(멘트·가사 없음)으로 다시 씀
            self.assertIn("비 오는 밤 버스 창가", printed)  # 첫 번째 실행의 글
            self.assertIn("새 가사", printed)
            self.assertIn("보컬: 비중 40%(중심)", text)
            self.assertNotIn("fav.mp3", text)

    def test_cli_parses_ref_run(self):
        with mock.patch.object(km, "run", return_value=0) as run, mock.patch.object(km, "setup_logging"):
            km.main(["run", "--ref", "auto", "--count", "3", "--style-extra", "J-pop band", "--push", "--name", "x"])
        args = run.call_args[0][0]
        self.assertEqual((args.ref, args.count, args.style_extra, args.push, args.seeds), ("auto", 3, "J-pop band", True, None))


if __name__ == "__main__":
    unittest.main()
