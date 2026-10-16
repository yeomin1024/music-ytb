#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VERSION: v1.0 — 2026-10-16 — 사용자 지시 "음악 하나랑 그 음악을 듣고 느낀 멘트와 떠오르는 사진 각각 업로드하면 트랙을 분리한 후 각각
#          멜로디, 이펙트 등을 분석하고 변형해서 내 멘트와 이미지 종합해서 음악 하나 … 가사 여부는 내가 체크" + "멘트·사진은 내가
#          클로드한테 물어보고 결과 넣을테니까 클로드한테 뭐라고 얘기해야하는지 알려줘 · 선택 2(가사 초안을 보고 고친 뒤 생성)는 그렇게 해"
r"""
참고 곡 분석의 순수 계산 부분 — kaggle_music_yue2.py(_analyze)·kaggle_music_panel.py 가 함께 쓴다.

  stem_features(x, sr)        트랙 하나(스테레오) → 크기·활동 비율·밝기·거친 정도(평탄도)·다이내믹·음량 폭·스테레오 폭·잔향/지속(근사)
  stem_words(f)               위 수치 → 한국어 말 (아래 기준 — 합성 신호로 방향만 확인한 보정 전 값)
  melody_features(line)       받아 적은 멜로디 → 음역·순차/도약 비율·같은 음 반복·리듬 밀도 (음표 자체는 밖으로 내지 않는다)
  form_sections(abc_text)     받아 적은 악보의 구간 이름(% verse …) → 곡 구성
  claude_request(...)         Claude 대화에 붙일 글 (분석 숫자 + 내 멘트 + '사진 첨부' 안내 + 답 형식)
  parse_claude_answer(...)    Claude 답(STYLE / SECTIONS / LYRICS) → 도구 입력, 오류·주의

불러올 때는 표준 라이브러리만 쓴다 (numpy 는 stem_features 안에서). ※ 수치는 근사치. 연구·교육용.
"""
# CHANGELOG
#   v1.0 — 2026-10-16 — 새 파일 (위 함수들). 영향: kaggle_music_yue2.py v1.4 _analyze·analyze 명령, kaggle_music_panel.py v1.1
import math
import re

MODULE_VERSION = "v1.0"
NOTE_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
STEM_KO = {"vocals": "보컬", "drums": "드럼", "bass": "베이스", "guitar": "기타", "piano": "피아노",
           "other": "그 밖(신스·스트링·관악 등)"}
SECTION_NAMES = ("Intro", "Verse", "Chorus", "Bridge", "Outro")  # Claude 에게 이 중에서 고르게 함 (YuE2 공식 예시의 태그)
FRAME, HOP = 2048, 1024

# 말로 바꾸는 기준 (보정 전 — 합성 신호로 방향만 확인. 숫자를 함께 넘기므로 Claude 가 숫자도 본다)
SHARE_WORDS = ((0.25, "중심"), (0.08, "또렷함"), (0.02, "받쳐 줌"), (0.0, "거의 없음"))
CENTROID_WORDS = ((3000, "밝고 날카로운"), (1200, "중간 밝기"), (0, "어둡고 따뜻한"))
WIDTH_WORDS = ((0.15, "넓게 퍼짐(스테레오)"), (0.03, "약간 퍼짐"), (0.0, "가운데(모노에 가까움)"))
CREST_WORDS = ((15, "다이내믹(압축 적음)"), (9, "보통"), (0, "강하게 압축됨"))
TAIL_WORDS = ((0.4, "잔향·지속 김"), (0.15, "보통"), (0.0, "짧고 건조"))
FLAT_ROUGH = 0.15  # 드럼 말고: 평탄도(소음 성분)가 이보다 크면 '거친 편'. 디스토션 판별은 아님 — 찌그러뜨린 사인파도 평탄도 0 (합성 시험)
MIN_ONSETS_FOR_TAIL = 5


def _pick(table, value):
    for threshold, word in table:
        if value >= threshold:
            return word
    return table[-1][1]


def _db(x):
    return 20 * math.log10(x) if x > 0 else -120.0


# ------------------------------------------------------------------ 트랙 수치 (numpy)
def stem_features(x, sr):
    """트랙 하나 → 수치 dict. x: (2, N) 또는 (N, 2) 또는 (N,) — 범위 -1~1.
    rms_db·active_ratio(최고 프레임 -30dB 이상 비율)·centroid_hz(밝기, 활동 프레임 중앙값)·flatness(0~1, 소음 성분 — 거친 정도)·
    crest_db(최고/평균 — 작을수록 압축)·lra_db(활동 프레임 크기 p95-p10)·width(옆/전체 에너지)·tail(소리 시작 0.2~0.4초 뒤
    에너지/시작 에너지 — 잔향·지속 근사, 시작이 5번 미만이면 None)."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = np.stack([a, a])
    elif a.shape[0] != 2 and a.shape[-1] == 2:
        a = a.T
    left, right = a[0], a[1]
    mid, side = (left + right) / 2, (left - right) / 2
    energy = float(np.mean(left ** 2 + right ** 2) / 2)
    out = dict(rms_db=round(_db(math.sqrt(energy)), 1), energy=energy)
    if energy < 1e-12:
        return out | dict(active_ratio=0.0, silent=True)
    mono = mid if len(mid) >= FRAME else np.pad(mid, (0, FRAME - len(mid)))
    frames = sliding_window_view(mono, FRAME)[::HOP]
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    frame_db = 20 * np.log10(frame_rms + 1e-12)
    active = frame_db > max(float(frame_db.max()) - 30, -60)
    window = np.hanning(FRAME)
    freqs = np.fft.rfftfreq(FRAME, 1 / sr)
    cents, flats = [], []
    idx = np.flatnonzero(active)
    for k in range(0, len(idx), 512):
        spec = np.abs(np.fft.rfft(frames[idx[k:k + 512]] * window, axis=1)) ** 2 + 1e-12
        cents.append((spec * freqs).sum(axis=1) / spec.sum(axis=1))
        flats.append(np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1))
    rms_active = float(np.sqrt(np.mean(frame_rms[active] ** 2)))
    hop_s = HOP / sr
    a0, a1 = max(1, round(0.2 / hop_s)), max(2, round(0.4 / hop_s))
    rises = np.flatnonzero((np.diff(frame_db, prepend=frame_db[0]) >= 6) & active)
    ratios = []
    for i, start in enumerate(rises):
        nxt = rises[i + 1] if i + 1 < len(rises) else len(frame_rms)
        if nxt - start <= a1 or start + a1 >= len(frame_rms):
            continue  # 다음 소리가 0.4초 안에 오면 잔향이 아니라 다음 음을 재게 되므로 뺀다
        peak = float(np.max(frame_rms[start:start + 3]) ** 2)
        tail = float(np.mean(frame_rms[start + a0:start + a1] ** 2))
        ratios.append(tail / peak if peak > 0 else 0.0)
    out.update(
        active_ratio=round(float(np.mean(active)), 3),
        centroid_hz=round(float(np.median(np.concatenate(cents))), 0),
        flatness=round(float(np.median(np.concatenate(flats))), 3),
        crest_db=round(_db(float(np.max(np.abs(mid))) / rms_active), 1) if rms_active > 0 else None,
        lra_db=round(float(np.percentile(frame_db[active], 95) - np.percentile(frame_db[active], 10)), 1),
        width=round(float(np.mean(side ** 2) / (np.mean(mid ** 2) + np.mean(side ** 2) + 1e-20)), 3),
        tail=round(float(np.median(ratios)), 3) if len(ratios) >= MIN_ONSETS_FOR_TAIL else None,
        tail_onsets=len(ratios),
    )
    return out


def add_shares(features):
    """{트랙: 수치} 에 share(그 트랙 에너지 / 모든 트랙 에너지 합)를 넣는다."""
    total = sum(f.get("energy", 0.0) for f in features.values())
    for f in features.values():
        f["share"] = round(f.get("energy", 0.0) / total, 3) if total > 0 else 0.0
    return features


def stem_words(f, stem=None):
    """수치 → 말 dict (있는 수치만)."""
    words = {}
    if "share" in f:
        words["비중"] = _pick(SHARE_WORDS, f["share"])
    if f.get("silent"):
        return words
    if f.get("centroid_hz") is not None:
        words["밝기"] = _pick(CENTROID_WORDS, f["centroid_hz"])
    if f.get("width") is not None:
        words["스테레오"] = _pick(WIDTH_WORDS, f["width"])
    if f.get("crest_db") is not None:
        words["다이내믹"] = _pick(CREST_WORDS, f["crest_db"])
    words["잔향·지속"] = _pick(TAIL_WORDS, f["tail"]) if f.get("tail") is not None else "잴 수 없음(소리가 촘촘함)"
    if stem != "drums" and f.get("flatness") is not None:
        words["거친 정도"] = "거친 편(소음 성분 많음)" if f["flatness"] > FLAT_ROUGH else "매끈한 편"
    return words


# ------------------------------------------------------------------ 멜로디 특징 (음표는 내보내지 않음)
def note_name(midi):
    return f"{NOTE_NAMES[int(midi) % 12]}{int(midi) // 12 - 1}"


def melody_features(line):
    """[(onset 4분음표, MIDI 음)] → 특징 dict. 음이 8개 미만이면 None."""
    if len(line) < 8:
        return None
    pitches = sorted(p for _, p in line)
    n = len(pitches)
    lo, hi = pitches[int(0.05 * (n - 1))], pitches[int(0.95 * (n - 1))]
    intervals = [line[i + 1][1] - line[i][1] for i in range(n - 1)]
    moving = [i for i in intervals if i != 0]
    iois = sorted(float(line[i + 1][0] - line[i][0]) for i in range(n - 1))
    median_ioi = iois[len(iois) // 2] or 1e-9
    return dict(
        notes=n, range_semitones=hi - lo, low=note_name(lo), high=note_name(hi),
        step_ratio=round(sum(1 for i in moving if abs(i) <= 2) / len(moving), 2) if moving else 0.0,
        leap_ratio=round(sum(1 for i in moving if abs(i) >= 5) / len(moving), 2) if moving else 0.0,
        repeat_ratio=round(intervals.count(0) / len(intervals), 2),
        mean_abs_interval=round(sum(abs(i) for i in moving) / len(moving), 2) if moving else 0.0,
        notes_per_beat=round(1 / median_ioi, 2),
    )


def melody_words(f):
    if not f:
        return {}
    words = {"음역": "좁은 음역" if f["range_semitones"] <= 9 else "보통 음역" if f["range_semitones"] <= 15 else "넓은 음역"}
    if f["step_ratio"] >= 0.6:
        words["진행"] = "순차 진행 위주"
    elif f["leap_ratio"] >= 0.25:
        words["진행"] = "도약이 많음"
    else:
        words["진행"] = "순차·도약 섞임"
    words["리듬"] = ("16분음표가 많은 촘촘한 리듬" if f["notes_per_beat"] >= 3 else
                   "8분음표 위주" if f["notes_per_beat"] >= 1.5 else "느긋한 리듬(4분음표 이상)")
    if f["repeat_ratio"] >= 0.3:
        words["반복"] = "같은 음 반복이 많음"
    return words


def form_sections(abc_text):
    """악보의 구간 줄('% verse') → ['Verse', 'Chorus', …] (같은 이름이 이어지면 하나로)."""
    out = []
    for line in (abc_text or "").splitlines():
        if line.startswith("% "):
            label = line[2:].strip().title()
            if label and (not out or out[-1] != label):
                out.append(label)
    return out


# ------------------------------------------------------------------ Claude 에게 보낼 글
def _summary_lines(analysis):
    lib = analysis.get("librosa", {})
    sheet = analysis.get("sheetsage2", {})
    tempo = sheet.get("bpm") if sheet.get("ok") else lib.get("tempo_bpm")
    lines = [f"- 길이 {lib.get('seconds', '?')}초, 템포 약 {round(float(tempo)) if tempo else '?'} BPM"
             + (f", 조성 {sheet['key']}" if sheet.get("ok") and sheet.get("key") else "")]
    tags = analysis.get("tags", {})
    evidence = [f"{lb} {p}" for kind in ("genres", "moods", "instruments") for lb, p in tags.get(kind + "_evidence", [])]
    if evidence:
        lines.append("- 장르·분위기·악기 태그 (AudioSet 확률 0~1): " + ", ".join(evidence))
    stems = analysis.get("stems", {})
    if stems:
        lines.append("- 트랙별 (비중 = 곡 전체 소리에서 그 트랙의 비율):")
        for stem in ("vocals", "drums", "bass", "guitar", "piano", "other"):
            f = stems.get(stem)
            if not f:
                continue
            w = f.get("words", {})
            parts = [f"비중 {round(100 * f.get('share', 0))}%({w.get('비중', '?')})"]
            if not f.get("silent") and f.get("share", 0) >= 0.02:
                parts += [f"밝기 {int(f['centroid_hz'])}Hz({w.get('밝기')})" if f.get("centroid_hz") is not None else "",
                          f"스테레오 폭 {f['width']}({w.get('스테레오')})" if f.get("width") is not None else "",
                          f"다이내믹 {f['crest_db']}dB({w.get('다이내믹')})" if f.get("crest_db") is not None else "",
                          f"잔향·지속 {f['tail']}({w.get('잔향·지속')})" if f.get("tail") is not None else f"잔향·지속 {w.get('잔향·지속')}",
                          f"거친 정도 {f.get('flatness')}({w['거친 정도']})" if w.get("거친 정도") else ""]
            lines.append(f"  · {STEM_KO.get(stem, stem)}: " + " · ".join(p for p in parts if p))
    elif analysis.get("stems_error"):
        lines.append(f"- 트랙 분리 못 함 ({analysis['stems_error'][:80]}) — 장르·분위기 태그만 참고")
    mel, mw = analysis.get("melody_features"), analysis.get("melody_words", {})
    if mel:
        lines.append(f"- 멜로디 특징 (음표는 알려 주지 않음): 음역 {mel['range_semitones']}반음({mw.get('음역')}) · "
                     f"순차 진행 {round(100 * mel['step_ratio'])}% · 도약 {round(100 * mel['leap_ratio'])}% ({mw.get('진행')}) · "
                     f"리듬 밀도 {mel['notes_per_beat']}음/박({mw.get('리듬')})" + (f" · {mw['반복']}" if mw.get("반복") else ""))
    if analysis.get("form"):
        lines.append("- 원곡 구성: " + ", ".join(analysis["form"]))
    return lines


def claude_request(analysis, comment="", lyrics=False):
    """Claude 대화에 붙일 글. 사진은 사용자가 그 대화에 따로 첨부한다."""
    vocal = ("보컬의 언어·성별·음색도 넣기 (예: \"Korean, warm breathy female vocal\")" if lyrics
             else "보컬·노래 관련 낱말은 넣지 않기 (연주곡)")
    lyric_rule = ("내 멘트와 사진에서 떠오르는 **새 가사**, 한국어. 섹션마다 [Verse]·[Chorus] 같은 태그 줄로 시작하고 순서는 SECTIONS 와 같게 "
                  "(Intro·Outro 는 태그만 써도 됨). 한 줄 8~14음절, 전체 16~28줄, 후렴은 짧고 기억에 남게 반복"
                  if lyrics else "\"없음\" 한 단어 (연주곡)")
    form = ", ".join(analysis.get("form") or []) or "알 수 없음"
    comment = (comment or "").strip() or "(여기에 멘트를 적어 주세요)"
    return "\n".join([
        "아래 [원곡 분석]은 내가 좋아하는 곡을 도구로 트랙별로 나눠 잰 숫자야 (너는 소리를 못 들으니 숫자로 설명할게).",
        "이 분석에서 느껴지는 분위기·악기 구성·소리 질감에, [내 멘트]와 첨부한 사진에서 떠오르는 감정과 장면을 합쳐서",
        "오픈소스 작곡 AI(YuE2)로 만들 **새 곡**의 설정을 써 줘.",
        "",
        "지킬 것",
        "1. 원곡을 따라 하지 않기 — 원곡 제목·가수·가사·멜로디를 쓰지 말고, 분위기·악기 구성·소리 질감만 참고.",
        "2. STYLE — 영어 한 줄, 쉼표로 구분, 400자 이하. 순서: 장르, 분위기(멘트·사진 반영), 악기와 소리 질감"
        "(예: \"reverb-drenched clean electric guitar\", \"tight punchy drums\", \"wide lush strings\"), 장조/단조, 템포(숫자 BPM). "
        + vocal + ".",
        f"3. SECTIONS — 곡 구성. {', '.join(SECTION_NAMES)} 중에서 골라 쉼표로 6~9개. 참고 — 원곡 구성: {form}",
        "4. LYRICS — " + lyric_rule + ".",
        "5. 답은 아래 형식 그대로 코드 블록 하나에. 코드 블록 밖에는 왜 그렇게 정했는지 3줄 이내.",
        "",
        "```",
        "STYLE: ...",
        "SECTIONS: Intro, Verse, Chorus, ...",
        "LYRICS:",
        *(["[Verse]", "...", "", "[Chorus]", "..."] if lyrics else ["없음"]),
        "```",
        "",
        "[내 멘트]",
        comment,
        "",
        "[사진]",
        "(이 메시지에 사진을 첨부했어 — 색감·장소·시간대·감정을 반영해 줘)",
        "",
        "[원곡 분석 — 숫자는 도구가 잰 근사치]",
        *_summary_lines(analysis),
    ]).strip() + "\n"


# ------------------------------------------------------------------ Claude 답 읽기
_TAG = re.compile(r"^\[([A-Za-z][A-Za-z0-9 -]{0,30})\]\s*$")
_KEY = re.compile(r"^\s*(STYLE|SECTIONS|LYRICS)\s*[:：]\s*(.*)$", re.I)


def parse_claude_answer(text, want_lyrics):
    """Claude 답 → ({style, sections, lyrics}, 오류 목록, 주의 목록). 코드 블록 안팎 모두 읽는다 (블록이 있으면 블록 안만)."""
    errors, warnings = [], []
    body = text or ""
    blocks = re.findall(r"```[^\n]*\n(.*?)```", body, flags=re.S)
    if blocks:
        body = max(blocks, key=lambda b: ("STYLE" in b.upper(), len(b)))
    fields, current = {"style": None, "sections": None, "lyrics": []}, None
    for raw in body.splitlines():
        m = _KEY.match(raw)
        if m:
            key, value = m.group(1).lower(), m.group(2).strip()
            current = key
            if key == "lyrics":
                if value:
                    fields["lyrics"].append(value)
            else:
                fields[key] = value
            continue
        if current == "lyrics":
            fields["lyrics"].append(raw.rstrip())
    style = (fields["style"] or "").strip().strip("\"'`")
    if not style:
        errors.append("STYLE 줄이 없음")
    elif len(style) > 400:
        warnings.append(f"STYLE 이 {len(style)}자 — 400자 이하 권장")
    sections = [s.strip().title() for s in (fields["sections"] or "").split(",") if s.strip()]
    if not sections:
        errors.append("SECTIONS 줄이 없음")
    bad = [s for s in sections if s not in SECTION_NAMES]
    if bad:
        errors.append(f"SECTIONS 에 쓸 수 없는 이름: {', '.join(bad)} (쓸 수 있는 것: {', '.join(SECTION_NAMES)})")
    lines = [ln for ln in fields["lyrics"]]
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    lyrics = ""
    if want_lyrics:
        tags = [m.group(1).title() for ln in lines if (m := _TAG.match(ln.strip()))]
        sung = [ln for ln in lines if ln.strip() and not _TAG.match(ln.strip())]
        if not tags or not sung:
            errors.append("LYRICS 에 [Verse] 같은 태그 줄과 가사가 있어야 함 (가사 넣기를 켰음)")
        else:
            bad_tags = [t for t in tags if t not in SECTION_NAMES]
            if bad_tags:
                errors.append(f"가사 태그에 쓸 수 없는 이름: {', '.join(bad_tags)}")
            if sections and [t for t in tags] != sections:
                warnings.append("가사 태그 순서가 SECTIONS 와 다름 — 노래는 가사 태그 순서를 따른다")
            if len(sung) > 40:
                warnings.append(f"가사가 {len(sung)}줄 — 길면 생성 한도에 닿아 잘릴 수 있음 (16~28줄 권장)")
            out, prev_blank = [], True
            for ln in lines:  # 태그 앞에는 빈 줄 하나, 태그는 [Verse] 모양으로
                m = _TAG.match(ln.strip())
                if m:
                    if not prev_blank:
                        out.append("")
                    out.append(f"[{m.group(1).title()}]")
                    prev_blank = False
                elif ln.strip():
                    out.append(ln.strip())
                    prev_blank = False
                elif not prev_blank:
                    out.append("")
                    prev_blank = True
            lyrics = "\n".join(out).strip() + "\n"
    elif lines and "".join(lines).strip() not in ("없음", "none", "None", "-"):
        warnings.append("가사 넣기를 끈 상태라 LYRICS 는 쓰지 않음")
    return dict(style=style, sections=",".join(sections), lyrics=lyrics), errors, warnings
