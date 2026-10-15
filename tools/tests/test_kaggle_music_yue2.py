# -*- coding: utf-8 -*-
# VERSION: v1.1 — 2026-10-15 — music-ytb 저장소로 옮김 (내용 같음. v1.0: tools/kaggle_music_yue2.py 시험 — 가짜 모듈·가짜 nvidia-smi, GPU·네트워크·모델 없음)
r"""
  python3 -m unittest tools.tests.test_kaggle_music_yue2 -v     (저장소 최상위에서)

- 입력: 시드·섹션·연주곡 스타일(공식 prepare 규칙과 같은 결과)·GPU 나누기·이전 결과 안 덮기
- T4 패치: GraphAR 어텐션이 항상 'sdpa', NAR 질의 묶음은 호출자가 안 줬을 때만, 다시 적용해도 겹치지 않음
- 환경: nvidia-smi 줄 읽기, P100(연산 능력 6.0)은 설치 전에 멈춤
"""
import subprocess
import sys
import tempfile
import types
import unittest
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


class TestInputs(unittest.TestCase):
    def test_seeds(self):
        self.assertEqual(km.parse_seeds("1, 2,3"), [1, 2, 3])
        for bad in ("", "1,1", "-1", "a", str(2**63)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                km.parse_seeds(bad)

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


if __name__ == "__main__":
    unittest.main()
