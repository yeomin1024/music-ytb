#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VERSION: v1.0 — 2026-11-11 — 사용자 지시 "2번으로 소리까지 AI가 만드는 셀" + "스트링 소리도 추가 … 한 악기씩만 나오도록 해야해":
#          YuE2 로 악기·톤마다 몇 초짜리 견본 — 견본 표·작업 나누기·잘라 내기 명령·'한 악기' 판정 (순수 계산)
r"""
AI(YuE2) 음색 견본의 순수 계산 부분 — kaggle_music_yue2.py samples 명령·_worker --jobs 가 쓴다.

  AI_SAMPLES        견본 표: 악기·톤마다 영어 프롬프트(혼자 연주·다른 악기 없음·100 BPM·C장조), 계획 방식
                    (melody = 선율만 → 단일 멜로디, full = 화음까지 → 코드 진행), 걸러 낼 트랙(Demucs htdemucs_6s 이름)
  select            --only 묶음·--keys 견본 고르기
  plan_jobs         GPU 마다 견본 나누기 (견본마다 시드 tries 개 — 한 악기 비중이 모자라면 다음 시드)
  clip_cmd          ffmpeg: 앞의 조용한 부분 빼고 → seconds 초 → 끝 0.6초 줄임
  stem_shares       트랙별 에너지 비중 (Demucs 결과)
  solo_verdict      비중 → 판정 말

'한 악기만' 을 지키는 방법: YuE2 는 원래 노래 전체를 만드는 모델이라 혼자 연주를 보장하지 않는다 →
  (1) 프롬프트로 혼자 연주를 요청 → (2) Demucs 로 나눠 목표 트랙 비중을 잼 → (3) SOLO_MIN 미만이면 다음 시드로 다시
  → (4) 마지막 파일은 목표 트랙만 걸러 낸 소리 (AI 원래 소리도 따로 남겨 비교).
  Demucs 트랙은 drums·bass·guitar·piano·vocals·other 6개라 스트링·신스는 둘 다 'other' — 두 소리가 함께 나오면 나눌 수 없다.
※ 연구·교육용.
"""
# CHANGELOG
#   v1.0 — 2026-11-11 — 새 파일 (위 함수들). 영향: kaggle_music_yue2.py v1.6 samples·_worker --jobs, kaggle_music_panel.py v1.2 ai_samples
from dataclasses import dataclass

MODULE_VERSION = "v1.0"
DEMUCS_STEMS = ("drums", "bass", "other", "vocals", "guitar", "piano")  # htdemucs_6s
SOLO_MIN = 0.80      # 목표 트랙 에너지 비중이 이만큼 이상이면 '한 악기' — 경험칙 (Demucs 분리 오차가 있음)
MIXED_MIN = 0.50     # 이것 미만이면 모델이 혼자 연주를 거의 안 한 것
DEFAULT_SECONDS = 10
DEFAULT_TRIES = 3
DEFAULT_SECTIONS = "Verse"  # 계획 섹션 하나 — 짧게 (멜로디가 있는 섹션)
TEMPO_KEY = "100 BPM, C major"
GROUPS = {"acoustic": "어쿠스틱 기타", "electric": "일렉트릭 기타", "piano": "그랜드 피아노", "strings": "스트링",
          "drums": "드럼", "bass": "베이스", "synth": "신디사이저"}
_ALONE = "no other instruments, no drums, no bass, no percussion, no vocals"


@dataclass(frozen=True)
class AISample:
    key: str
    group: str
    name_ko: str
    style: str       # YuE2 스타일 (instrumental_style 이 'Instrumental,'·'no vocals …' 를 더함)
    plan_mode: str   # melody | full
    stem: str        # 걸러 낼 Demucs 트랙


def _s(desc, alone=_ALONE, tempo=TEMPO_KEY):
    return f"{desc}, solo performance, dry studio recording, {alone}, {tempo}"


AI_SAMPLES = (
    AISample("acoustic_melody", "acoustic", "통기타 — 단일 멜로디",
             _s("solo steel-string acoustic guitar playing a simple single-note melody, fingerpicked, warm and intimate"), "melody", "guitar"),
    AISample("acoustic_chords", "acoustic", "통기타 — 코드 진행",
             _s("solo steel-string acoustic guitar strumming a pop chord progression, bright open chords"), "full", "guitar"),
    AISample("electric_clean_melody", "electric", "일렉 클린톤 — 단일 멜로디",
             _s("solo clean electric guitar playing a single-note melody, glassy clean tone, light chorus and reverb"), "melody", "guitar"),
    AISample("electric_clean_chords", "electric", "일렉 클린톤 — 코드 진행",
             _s("solo clean electric guitar playing a chord progression, clean tone, strummed and arpeggiated chords"), "full", "guitar"),
    AISample("electric_crunch_melody", "electric", "일렉 크런치(오버드라이브) — 단일 멜로디",
             _s("solo overdriven electric guitar playing a single-note melody, warm crunchy tube amp tone"), "melody", "guitar"),
    AISample("electric_crunch_chords", "electric", "일렉 크런치(오버드라이브) — 코드 진행",
             _s("solo overdriven electric guitar playing rhythm chords, crunchy classic rock tone"), "full", "guitar"),
    AISample("electric_dist_melody", "electric", "일렉 디스토션 — 단일 멜로디",
             _s("solo heavily distorted electric guitar playing a single-note lead melody, high-gain tone with long sustain"), "melody", "guitar"),
    AISample("electric_dist_chords", "electric", "일렉 디스토션 — 코드 진행 (파워 코드)",
             _s("solo heavily distorted electric guitar playing palm-muted power chords, high-gain rock tone"), "full", "guitar"),
    AISample("electric_jazz_chords", "electric", "재즈 기타 — 코드 진행",
             _s("solo hollow-body jazz electric guitar playing mellow jazz chords, warm dark tone"), "full", "guitar"),
    AISample("piano_melody", "piano", "그랜드 피아노 — 단일 멜로디",
             _s("solo concert grand piano playing a simple single-note melody with the right hand"), "melody", "piano"),
    AISample("piano_chords", "piano", "그랜드 피아노 — 코드 진행",
             _s("solo concert grand piano playing a chord progression with both hands"), "full", "piano"),
    AISample("strings_melody", "strings", "스트링 — 단일 멜로디 (현악 합주)",
             _s("solo string section of violins, violas and cellos playing a legato single-line melody, orchestral strings only"),
             "melody", "other"),
    AISample("strings_chords", "strings", "스트링 — 코드 진행 (현악 합주)",
             _s("solo string section playing sustained chords, lush orchestral strings only, violins, violas and cellos"), "full", "other"),
    AISample("drums_acoustic", "drums", "드럼 — 어쿠스틱 드럼 세트",
             _s("solo acoustic drum kit playing a pop rock groove with kick, snare and hi-hat",
                "no melody, no guitar, no bass, no piano, no other instruments, no vocals", "100 BPM"), "melody", "drums"),
    AISample("drums_electronic", "drums", "드럼 — 전자 드럼 (드럼 머신)",
             _s("solo electronic drum machine beat, punchy 808 kick, crisp snare and hi-hats",
                "no melody, no synth, no bass, no other instruments, no vocals", "100 BPM"), "melody", "drums"),
    AISample("bass_electric", "bass", "일렉 베이스 — 베이스 라인",
             _s("solo electric bass guitar playing a groovy fingerstyle bass line", "no other instruments, no drums, no guitar, no vocals"),
             "melody", "bass"),
    AISample("bass_synth", "bass", "신스 베이스 — 베이스 라인",
             _s("solo analog synth bass playing a deep punchy bass line", "no other instruments, no drums, no vocals"), "melody", "bass"),
    AISample("synth_lead", "synth", "신스 리드 — 단일 멜로디",
             _s("solo analog synthesizer lead playing a single-note melody, bright sawtooth"), "melody", "other"),
    AISample("synth_pad", "synth", "신스 패드 — 코드 진행",
             _s("solo warm analog synthesizer pad playing sustained chords, lush and wide"), "full", "other"),
)


def select(only="", keys=""):
    """--only 묶음(쉼표)·--keys 견본(쉼표) → 견본 목록 (표 순서). 모르는 이름이면 ValueError."""
    groups = [g.strip() for g in (only or "").split(",") if g.strip()]
    names = [k.strip() for k in (keys or "").split(",") if k.strip()]
    bad = [g for g in groups if g not in GROUPS] + [k for k in names if k not in {s.key for s in AI_SAMPLES}]
    if bad:
        raise ValueError(f"모르는 이름: {', '.join(bad)} (묶음: {', '.join(GROUPS)})")
    picked = [s for s in AI_SAMPLES if (not groups or s.group in groups) and (not names or s.key in names)]
    if not picked:
        raise ValueError("고른 견본이 없음")
    return picked


def plan_jobs(samples, gpu_indices, seed=1, tries=DEFAULT_TRIES, sections=DEFAULT_SECTIONS, seconds=DEFAULT_SECONDS):
    """견본을 GPU 에 돌아가며 → {gpu: [작업 dict]}. 견본마다 시드 seed..seed+tries-1 (앞 시드가 '한 악기'면 거기서 멈춤)."""
    if not gpu_indices:
        raise ValueError("GPU 가 없음")
    if not 1 <= tries <= 5:
        raise ValueError(f"tries 는 1~5: {tries}")
    plan = {}
    for i, s in enumerate(samples):
        job = dict(key=s.key, group=s.group, group_ko=GROUPS[s.group], name=s.name_ko, style=s.style, plan_mode=s.plan_mode,
                   stem=s.stem, seeds=list(range(seed, seed + tries)), sections=sections, seconds=seconds)
        plan.setdefault(gpu_indices[i % len(gpu_indices)], []).append(job)
    return plan


def clip_cmd(src, dst, seconds=DEFAULT_SECONDS, fade=0.6):
    """ffmpeg 명령: 앞의 조용한 부분(-45dB 아래) 빼고 → seconds 초 → 끝 fade 초 줄임 → 44.1kHz 스테레오."""
    af = (f"silenceremove=start_periods=1:start_threshold=-45dB,atrim=0:{seconds},"
          f"afade=t=out:st={max(0.0, seconds - fade):.2f}:d={fade}")
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-af", af, "-ar", "44100", "-ac", "2", str(dst)]


def mp3_cmd(src, dst, bitrate="192k"):
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-b:a", bitrate, str(dst)]


def stem_shares(energies):
    """{트랙: 에너지} → {트랙: 비중} (합 1, 소수 셋째 자리)."""
    total = sum(max(0.0, float(e)) for e in energies.values())
    return {k: (round(max(0.0, float(e)) / total, 3) if total > 0 else 0.0) for k, e in energies.items()}


def solo_verdict(share):
    """목표 트랙 비중 → (코드, 말)."""
    if share is None:
        return "none", "잴 수 없음"
    if share >= SOLO_MIN:
        return "solo", "거의 그 악기만 (남은 소리는 걸러 냄)"
    if share >= MIXED_MIN:
        return "mixed", "다른 소리가 조금 섞여 걸러 냄 — 걸러 낸 흔적이 들릴 수 있음"
    return "poor", "모델이 혼자 연주를 잘 안 함 — 걸러 낸 소리가 약하거나 흔적이 많을 수 있음"


def others_text(shares, stem, top=2):
    """목표 밖 트랙 중 큰 것 → '드럼 12% · 보컬 5%' (로그·화면용)."""
    ko = {"drums": "드럼", "bass": "베이스", "other": "그 밖(스트링·신스 등)", "vocals": "보컬", "guitar": "기타", "piano": "피아노"}
    rest = sorted(((v, k) for k, v in shares.items() if k != stem and v >= 0.01), reverse=True)[:top]
    return " · ".join(f"{ko.get(k, k)} {v:.0%}" for v, k in rest) or "없음"
