"""
Paper Coach: a writing quality and citation checker for academic and management papers.

Run it:
    pip install -r requirements.txt
    streamlit run app.py

What it does (all rule based, no AI model, runs offline on any laptop):
    1.  Citation checker        claims with figures or "studies show" but no citation
    2.  Reference matcher       in-text citations vs your reference list
    3.  Readability and rhythm  Flesch, grade level, sentence length chart, monotony
    4.  Filler and buzzwords    wordy phrases, vague words, clichés, with fixes
    5.  Repetition finder       repeated openers, words, phrases, near-duplicate sentences
    6.  Argument structure      topic sentence, evidence, so-what, length per paragraph
    7.  Highlighter             long sentences, passive voice, filler, uncited claims
    8.  Source overlap checker  finds copied wording against sources you upload
    9.  File support            .docx  .pdf  .txt  .md  or paste
    10. Export + compare        Markdown, HTML, CSV reports and a before/after view
"""
from __future__ import annotations

import bisect
import collections
import datetime
import difflib
import html
import io
import re
import statistics

import pandas as pd
import streamlit as st

# =============================================================================
# Constants and patterns
# =============================================================================

YEAR = r"(?:19|20)\d{2}[a-z]?"
WORDS = re.compile(r"\b[\w'’\-]+\b")
ALPHA_WORDS = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
CAP_WORD = re.compile(r"\b([A-Z][A-Za-z'’\-]+)")

STOP = set(
    """a about above after again against all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further had has
    have having he her here hers him his how i if in into is it its itself just me more most my no nor
    not of off on once only or other our out over own same she should so some such than that the their
    theirs them then there these they this those through to too under until up very was we were what
    when where which while who whom why will with would you your may might must shall one however thus
    therefore within among across towards toward""".split()
)

CIT_STOP = {
    "See", "Also", "Source", "Sources", "Figure", "Fig", "Table", "In", "From", "Adapted", "Based",
    "Note", "Notes", "Chapter", "Page", "Section", "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October", "November", "December", "FY", "Q", "Appendix",
    "Eq", "Equation", "Cf", "Eg", "Ie",
}

PAREN_CIT = re.compile(r"\(([^()]*?\b" + YEAR + r"\b[^()]*?)\)")
NARR_CIT = re.compile(
    r"\b([A-Z][A-Za-z'’\-]+)(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][A-Za-z'’\-]+)?(?:'s|’s)?\s*\(\s*("
    + YEAR
    + r")\s*\)"
)
NUM_CIT = re.compile(r"\[\s*\d+(?:\s*[,–\-]\s*\d+)*\s*\]")
REF_HEAD = re.compile(
    r"(?im)^[ ]*(?:\d+[.)]?[ ]+)?(references?|reference list|bibliography|works cited|sources cited|list of references)[ ]*:?[ ]*$"
)
ENTRY_START = re.compile(r"^\s*(?:(?:\[\d+\]|\d+[.)])\s+\S|[A-Z][A-Za-z'’\-]+,\s+[A-Z])")

ABBR = re.compile(
    r"(?<![A-Za-z])(?:e\.g\.|i\.e\.|et al\.|vs\.|Fig\.|Figs\.|Eq\.|Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.|Inc\.|Ltd\."
    r"|Co\.|approx\.|cf\.|pp\.|U\.S\.|U\.K\.|[A-Z]\.(?=\s[A-Z]\.)|[A-Z]\.(?=\s[A-Z][a-z]))"
)
SENT_BOUNDARY = re.compile(r"[.!?]+[\"”’)\]]*\s+(?=[\"“(\[]?[A-Z0-9])")

# claims that usually need a source
CLAIM_PATTERNS = [
    ("statistic or figure", "high", r"\b\d+(?:\.\d+)?\s?(?:%|percent|per cent)"),
    ("statistic or figure", "high", r"\b\d[\d,.]*\s+(?:million|billion|trillion|thousand)\b"),
    ("attributes a finding to research or experts", "high",
     r"\b(?:studies|research|surveys?|reports?|evidence|data|literature|scholars|researchers|experts|analysts)"
     r"\s+(?:show|shows|showed|suggest|suggests|suggested|indicate|indicates|indicated|found|find|reveal|"
     r"reveals|revealed|demonstrate|demonstrates|confirm|confirms)\b"),
    ("attributes a finding to research or experts", "high", r"\baccording to\b"),
    ("attributes a finding to research or experts", "high",
     r"\b(?:it has been|has been|have been)\s+(?:shown|found|proven|demonstrated|established)\b"),
    ("appeals to common knowledge", "high",
     r"\b(?:it is|is|are)\s+(?:widely|well|generally|commonly)\s+(?:known|accepted|recognized|recognised|believed|agreed)\b"),
    ("vague group of authorities", "high",
     r"\b(?:many|most|several|some)\s+(?:researchers|scholars|studies|experts|authors|managers|organizations|"
     r"organisations|companies|firms)\b"),
    ("dated fact", "medium", r"\b(?:in|since|by|during)\s+(?:19|20)\d{2}\b"),
    ("absolute wording", "medium",
     r"\b(?:always|never|undoubtedly|obviously|everyone|nobody|without exception|universally|inevitably|proves?)\b"),
]
CLAIM_COMPILED = [(lab, sev, re.compile(p, re.I)) for lab, sev, p in CLAIM_PATTERNS]

# wording checks: (regex, category, suggestion)
FILLER_RAW = [
    (r"in order to", "wordy", "to"),
    (r"due to the fact that|owing to the fact that", "wordy", "because"),
    (r"in spite of the fact that|despite the fact that", "wordy", "although"),
    (r"at this point in time|at the present time|at this moment in time", "wordy", "now"),
    (r"in the event that", "wordy", "if"),
    (r"for the purpose of", "wordy", "to / for"),
    (r"in the process of", "wordy", "cut it, or use the plain verb"),
    (r"has the ability to|have the ability to|is able to|are able to", "wordy", "can"),
    (r"with regard to|with respect to|in regard to|in relation to", "wordy", "about / on"),
    (r"in terms of", "wordy", "name the thing: for, in, on"),
    (r"the fact that", "wordy", "rewrite without it"),
    (r"subsequent to|prior to", "formal", "after / before"),
    (r"a number of", "vague", "give the number, or say several"),
    (r"a lot of|lots of", "vague", "many / much, or give the amount"),
    (r"it is important to note that|it should be noted that|it is worth noting that|it is interesting to note that|it must be noted that",
     "filler", "cut it and state the point"),
    (r"needless to say|it goes without saying", "filler", "cut it"),
    (r"as a matter of fact", "filler", "cut it"),
    (r"each and every", "filler", "each / every"),
    (r"first and foremost", "filler", "first"),
    (r"last but not least", "filler", "finally"),
    (r"at the end of the day", "cliché", "so / in the end, or cut"),
    (r"plays? (?:a|an) (?:crucial|key|vital|important|significant|pivotal|critical|major) role", "vague",
     "say what it actually does"),
    (r"in today['’]s (?:fast-paced |modern |digital |global |competitive |dynamic )?(?:world|society|era|landscape|age|environment|business environment|economy)",
     "cliché", "open with your actual point"),
    (r"in the modern era|in the digital age", "cliché", "name the period or drop it"),
    (r"in recent years", "vague", "name the years"),
    (r"tip of the iceberg", "cliché", "cut it"),
    (r"paradigm shift", "buzzword", "say what actually changed"),
    (r"game[- ]chang(?:er|ing)", "buzzword", "say what changed and by how much"),
    (r"cutting[- ]edge|state[- ]of[- ]the[- ]art", "buzzword", "name the technology or result"),
    (r"leverag(?:e|es|ed|ing)", "buzzword", "use / apply / build on"),
    (r"synerg(?:y|ies|istic)", "buzzword", "name the combined effect"),
    (r"robust", "buzzword", "say how: tested, reliable, resistant to what"),
    (r"seamless(?:ly)?", "buzzword", "smooth / without gaps, and show it"),
    (r"innovative solutions?", "buzzword", "name the solution"),
    (r"holistic(?:ally)?", "buzzword", "complete / whole, or list what is included"),
    (r"best[- ]in[- ]class|world[- ]class", "buzzword", "give the benchmark"),
    (r"delv(?:e|es|ed|ing)", "overused", "examine / study / look at"),
    (r"tapestry|realm|multifaceted|myriad|plethora", "overused", "plain word: many / complex"),
    (r"pivotal|crucial|vital", "overused", "important, then show why"),
    (r"landscape", "overused", "market / field / industry"),
    (r"utili[sz](?:e|es|ed|ing|ation)", "formal", "use"),
    (r"facilitat(?:e|es|ed|ing)", "formal", "help / enable / allow"),
    (r"commenc(?:e|es|ed|ing)", "formal", "start / begin"),
    (r"ascertain(?:s|ed|ing)?", "formal", "find out"),
    (r"endeavou?r(?:s|ed|ing)?", "formal", "try"),
    (r"very|really|quite|extremely|incredibly|highly|rather|somewhat|fairly", "vague", "cut it or pick a stronger word"),
    (r"basically|essentially|actually|literally|simply", "filler", "cut it"),
    (r"various|numerous|several|multiple|a variety of|a wide range of|a wide variety of", "vague",
     "name them or give the count"),
    (r"things?|stuff", "vague", "name the thing"),
    (r"aspects?", "vague", "name the aspect"),
    (r"significant(?:ly)?", "vague", "if statistical, give the test; otherwise say how much"),
    (r"impacts?|impacted|impacting", "vague", "affects / changes / raises / lowers, and say which"),
    (r"may potentially|could possibly|might perhaps|can potentially", "hedge", "pick one hedge"),
    (r"it can be argued that|it could be argued that|one could argue that|it is argued that", "hedge",
     "say who argues it, or own the claim"),
    (r"overall|in general|generally speaking", "filler", "cut it, or be specific"),
]
FILLER = [(re.compile(r"\b(?:" + p + r")\b", re.I), cat, sug) for p, cat, sug in FILLER_RAW]

_IRREG = (
    "known|made|done|seen|given|taken|shown|found|built|held|led|paid|sold|told|brought|thought|kept|lost|"
    "chosen|driven|written|begun|grown|drawn|thrown|broken|forgotten|hidden|proven|stolen|spoken|undertaken|"
    "forgiven|bought|caught|taught|sent|spent|understood|withdrawn"
)
_NOT_PART = (
    "speed|indeed|bleed|breed|exceed|proceed|succeed|embed|hundred|sacred|interested|excited|concerned|worried|"
    "tired|surprised|pleased|supposed|limited|advanced|experienced|skilled|motivated|committed|dedicated|"
    "qualified|diversified|sophisticated|complicated|detailed|aged|mixed"
)
PASSIVE_RE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\s+"
    r"(?:(?:not|also|often|usually|typically|frequently|then|still|already|always|never|widely|generally|"
    r"commonly|largely|mainly|primarily)\s+)*"
    r"(?:(?!(?:" + _NOT_PART + r")\b)\w{3,}ed|" + _IRREG + r")\b",
    re.I,
)

EVIDENCE_RE = re.compile(
    r"\b(?:for example|for instance|such as|according to|e\.g\.|data|survey|study|studies|results|findings|"
    r"evidence|case|figure|table|percent)\b|%|\d",
    re.I,
)
ANALYSIS_RE = re.compile(
    r"\b(?:therefore|thus|hence|as a result|consequently|this means|this suggests|this shows|this indicates|"
    r"this implies|which suggests|which means|which shows|so that|in other words|this is why|for this reason|"
    r"it follows|this highlights|this demonstrates|this reveals|this explains|accordingly|this matters|"
    r"the implication|the lesson|the takeaway)\b",
    re.I,
)
WEAK_OPENER = re.compile(
    r"^(?:however|moreover|furthermore|additionally|also|and|but|so|thus|therefore|this|these|that|those|it|they)\b",
    re.I,
)


# =============================================================================
# Reading files
# =============================================================================

def _decode_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def _read_docx(data: bytes) -> str:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(io.BytesIO(data))
    parts = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            parts.append(Paragraph(child, doc).text)
        elif tag == "tbl":
            tbl = Table(child, doc)
            for row in tbl.rows:
                cells = []
                for c in row.cells:
                    t = c.text.strip().replace("\n", " ")
                    if not cells or cells[-1] != t:
                        cells.append(t)
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _reflow_pdf(raw: str) -> str:
    lines = [ln.rstrip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if not re.fullmatch(r"\s*\d{1,4}\s*", ln)]  # page numbers
    lens = sorted(len(ln.strip()) for ln in lines if ln.strip())
    if not lens:
        return ""
    ref = lens[max(0, int(len(lens) * 0.9) - 1)]
    paras, buf = [], []
    in_refs = False

    def flush():
        if buf:
            paras.append(" ".join(buf))
            buf.clear()

    for ln in lines:
        s = ln.strip()
        if not s:
            flush()
            continue
        if REF_HEAD.match(s):  # the References heading is its own paragraph
            flush()
            paras.append(s)
            in_refs = True
            continue
        prev_ends = (not buf) or bool(re.search(r"[.!?:\"”)]$", buf[-1]))
        is_heading = (len(s) < 0.5 * ref and len(s.split()) <= 9 and s[:1].isupper()
                      and not re.search(r"[.!?;:,]$", s) and prev_ends)
        if is_heading:
            flush()
            paras.append(s)
            continue
        if in_refs and buf and ENTRY_START.match(s) and buf[-1].endswith("."):
            flush()
        if buf and buf[-1].endswith("-") and s[:1].islower():
            buf[-1] = buf[-1][:-1] + s
        else:
            buf.append(s)
        if re.search(r"[.!?:\"”)]$", s) and len(s) < 0.75 * ref:
            flush()
    flush()
    return "\n\n".join(paras)


def _read_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return _reflow_pdf("\n".join(pages))


def extract_text(name: str, data: bytes) -> tuple[str, str | None]:
    """Return (text, error)."""
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    try:
        if ext == "docx":
            return _read_docx(data), None
        if ext == "pdf":
            txt = _read_pdf(data)
            if len(txt.split()) < 20:
                return "", "That PDF has almost no readable text. It is probably a scan. Export it from Word as PDF, or paste the text."
            return txt, None
        if ext in ("txt", "md", "text", ""):
            return _decode_text(data), None
        if ext == "doc":
            return "", "Old .doc files are not supported. Open it in Word, Save As .docx, and upload that."
        return "", f"Unsupported file type: .{ext}"
    except Exception as exc:  # corrupt file, etc.
        return "", f"Could not read {name}: {exc}"


# =============================================================================
# Text structure helpers
# =============================================================================

def normalize(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ").replace("\ufeff", "")
    t = t.replace("\t", " ")
    t = re.sub(r"[ ]{2,}", " ", t)
    t = "\n".join(line.rstrip() for line in t.split("\n"))
    t = re.sub(r"\n{3,}", "\n\n", t)
    if "\n\n" not in t and t.count("\n") >= 2:  # pasted text with one newline per paragraph
        t = t.replace("\n", "\n\n")
    return t.strip()


def split_references(text: str):
    ms = list(REF_HEAD.finditer(text))
    if not ms:
        return text, "", None
    m = ms[-1]
    refs = text[m.end():].strip()
    if len(refs.split()) < 8:
        return text, "", None
    return text[: m.start()].rstrip(), refs, m.start()


def paragraph_spans(text: str):
    spans, pos = [], 0
    for m in re.finditer(r"\n[ \t]*\n+", text):
        seg = text[pos: m.start()]
        if seg.strip():
            s = pos + (len(seg) - len(seg.lstrip()))
            spans.append((s, pos + len(seg.rstrip())))
        pos = m.end()
    seg = text[pos:]
    if seg.strip():
        s = pos + (len(seg) - len(seg.lstrip()))
        spans.append((s, pos + len(seg.rstrip())))
    return spans


def sentence_spans(text: str):
    prot = ABBR.sub(lambda m: m.group().replace(".", "\x00"), text)
    spans, start = [], 0
    for m in SENT_BOUNDARY.finditer(prot):
        end = m.start() + len(m.group().rstrip())
        spans.append((start, end))
        start = m.end()
    spans.append((start, len(text)))
    out = []
    for s, e in spans:
        seg = text[s:e]
        s2 = s + (len(seg) - len(seg.lstrip()))
        e2 = s + len(seg.rstrip())
        if e2 > s2:
            out.append((s2, e2))
    return out


def classify_paragraph(t: str) -> str:
    wc = len(WORDS.findall(t))
    if " | " in t:
        return "table"
    if "\n" not in t.strip() and wc <= 12 and not re.search(r"[.!?;:,]$", t.strip()):
        return "heading"
    if wc < 3:
        return "heading"
    return "prose"


def parse_document(body: str):
    pars = []
    for s, e in paragraph_spans(body):
        t = body[s:e]
        kind = classify_paragraph(t)
        sents = []
        if kind == "prose":
            for ss, se in sentence_spans(t):
                seg = t[ss:se]
                wc = len(WORDS.findall(seg))
                if wc:
                    sents.append({"start": s + ss, "end": s + se, "text": " ".join(seg.split()), "wc": wc})
        pars.append({"start": s, "end": e, "text": t, "kind": kind, "sents": sents})
    return pars


def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", w)))


def flesch_label(score: float) -> str:
    if score >= 70:
        return "Easy to read"
    if score >= 50:
        return "Fairly readable"
    if score >= 30:
        return "Academic level (normal for papers)"
    return "Very dense"


# =============================================================================
# Citations and references
# =============================================================================

def expand_nums(inner: str) -> set[int]:
    nums: set[int] = set()
    for a, b in re.findall(r"(\d+)\s*[–\-]\s*(\d+)", inner):
        a, b = int(a), int(b)
        if 0 < b - a <= 50:
            nums.update(range(a, b + 1))
    inner2 = re.sub(r"(\d+)\s*[–\-]\s*(\d+)", " ", inner)
    nums.update(int(x) for x in re.findall(r"\d+", inner2))
    return nums


def find_citations(body: str):
    out = []
    for m in PAREN_CIT.finditer(body):
        for chunk in m.group(1).split(";"):
            ym = re.search(r"\b(" + YEAR + r")\b", chunk)
            if not ym:
                continue
            author = None
            for cm in CAP_WORD.finditer(chunk[: ym.start()]):
                if cm.group(1) not in CIT_STOP:
                    author = cm.group(1)
                    break
            if author:
                out.append({"pos": m.start(), "kind": "author-year", "author": author.lower(),
                            "year": ym.group(1), "raw": f"{author} {ym.group(1)}"})
    for m in NARR_CIT.finditer(body):
        if m.group(1) in CIT_STOP:
            continue
        out.append({"pos": m.start(), "kind": "author-year", "author": m.group(1).lower(),
                    "year": m.group(2), "raw": f"{m.group(1)} {m.group(2)}"})
    for m in NUM_CIT.finditer(body):
        out.append({"pos": m.start(), "kind": "numeric", "nums": sorted(expand_nums(m.group())), "raw": m.group()})
    out.sort(key=lambda c: c["pos"])
    return out


def parse_reference_entries(refs_text: str):
    entries = []
    for block in re.split(r"\n\s*\n", refs_text):
        if " | " in block:  # table rows are not references
            continue
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        cur: list[str] = []
        for ln in lines:
            if cur and ENTRY_START.match(ln) and re.search(r"\b" + YEAR + r"\b|\[\d+\]|^\d+[.)]", " ".join(cur)):
                entries.append(" ".join(cur))
                cur = [ln]
            else:
                cur.append(ln)
        if cur:
            entries.append(" ".join(cur))
    return entries


def ref_key(entry: str):
    e = re.sub(r"^\s*(?:\[\d+\]|\d+[.)])\s*", "", entry)
    ym = re.search(r"\b(" + YEAR + r")\b", e)
    year = ym.group(1) if ym else None
    head = e[: ym.start()] if ym else e[:60]
    author = None
    for cm in CAP_WORD.finditer(head):
        if cm.group(1) not in CIT_STOP:
            author = cm.group(1).lower()
            break
    return author, year


def match_references(cites, entries, has_ref_section: bool):
    res = {"has_refs": bool(entries), "n_entries": len(entries), "missing": [], "unused": [],
           "typos": [], "numeric": None, "note": ""}
    ay = [c for c in cites if c["kind"] == "author-year"]
    num = [c for c in cites if c["kind"] == "numeric"]
    res["n_unique_ay"] = len({(c["author"], c["year"][:4]) for c in ay})

    if (ay or num) and not has_ref_section:
        res["note"] = "No reference list found. Add a heading on its own line called References (or Bibliography) above your entries."
        return res
    if not entries:
        return res

    keyed = [(ref_key(e), e) for e in entries]
    ref_map = collections.defaultdict(list)
    for (a, y), e in keyed:
        if a and y:
            ref_map[(a, y[:4])].append(e)

    cited_counts = collections.Counter((c["author"], c["year"][:4]) for c in ay)
    display = {}
    for c in ay:
        display.setdefault((c["author"], c["year"][:4]), c["raw"])
    for key, cnt in cited_counts.items():
        if key in ref_map:
            continue
        sug = None
        for (ra, ry), ents in ref_map.items():
            if ry == key[1] and difflib.SequenceMatcher(None, ra, key[0]).ratio() >= 0.8:
                sug = f"{ra.title()} {ry}"
                break
        if sug:
            res["typos"].append({"cite": display[key], "count": cnt, "maybe": sug})
        else:
            res["missing"].append({"cite": display[key], "count": cnt})
    typo_keys = {t["cite"] for t in res["typos"]}
    for (a, y), e in keyed:
        if a and y:
            if (a, y[:4]) not in cited_counts:
                # skip if a typo suggestion already explains it
                if any(t["maybe"].lower() == f"{a} {y[:4]}" for t in res["typos"]):
                    continue
                res["unused"].append(e)
        elif not num:
            res["unused"].append(e + "   [no author or year detected]")

    if num:
        cited = set()
        for c in num:
            cited.update(c["nums"])
        numbers = []
        for i, e in enumerate(entries):
            m = re.match(r"^\s*(?:\[(\d+)\]|(\d+)[.)])", e)
            numbers.append(int(m.group(1) or m.group(2)) if m else i + 1)
        res["numeric"] = {
            "cited": sorted(cited),
            "out_of_range": sorted(n for n in cited if n not in numbers),
            "unused": sorted(n for n in numbers if n not in cited),
        }
    return res


def find_uncited_claims(pars, cite_positions):
    claims = []

    def has_cit(s):
        i = bisect.bisect_left(cite_positions, s["start"])
        return i < len(cite_positions) and cite_positions[i] < s["end"]

    for pi, p in enumerate(pars):
        sents = p["sents"]
        cited = [has_cit(s) for s in sents]
        for si, s in enumerate(sents):
            if cited[si] or s["wc"] < 5:
                continue
            reasons, sevs = [], []
            for label, sev, rx in CLAIM_COMPILED:
                if rx.search(s["text"]) and label not in reasons:
                    reasons.append(label)
                    sevs.append(sev)
            if not reasons:
                continue
            sev = "high" if "high" in sevs else "medium"
            near = (si > 0 and cited[si - 1]) or (si + 1 < len(sents) and cited[si + 1])
            if near:
                sev = "low"
            claims.append({"para": pi + 1, "start": s["start"], "end": s["end"], "severity": sev,
                           "reasons": ", ".join(reasons), "text": s["text"],
                           "note": "a neighbouring sentence has a citation" if near else ""})
    return claims


# =============================================================================
# Style checks
# =============================================================================

def find_wording(pars):
    hits, grouped = [], {}
    for p in pars:
        for s in p["sents"]:
            for rx, cat, sug in FILLER:
                for m in rx.finditer(s["text"]):
                    key = (rx.pattern, m.group().lower())
                    g = grouped.setdefault(key, {"phrase": m.group().lower(), "category": cat, "fix": sug,
                                                 "count": 0, "example": s["text"]})
                    g["count"] += 1
                    # position in the original body: sentence text is whitespace-normalised, so search the raw slice
    # positions (for highlighting) are computed on raw text below
    rows = sorted(grouped.values(), key=lambda g: (-g["count"], g["phrase"]))
    return rows


def find_wording_spans(body: str, pars):
    spans = []
    for p in pars:
        for s in p["sents"]:
            raw = body[s["start"]: s["end"]]
            for rx, _, _ in FILLER:
                for m in rx.finditer(raw):
                    spans.append((s["start"] + m.start(), s["start"] + m.end()))
    return spans


def find_passive_spans(body: str, pars):
    spans, sent_ids = [], set()
    for pi, p in enumerate(pars):
        for si, s in enumerate(p["sents"]):
            raw = body[s["start"]: s["end"]]
            ms = list(PASSIVE_RE.finditer(raw))
            if ms:
                sent_ids.add((pi, si))
                for m in ms:
                    spans.append((s["start"] + m.start(), s["start"] + m.end()))
    return spans, sent_ids


def find_repetition(pars):
    out = {"openers": [], "runs": [], "words": [], "dupes": [], "phrases": [], "top_words": [], "dupes_skipped": False}
    all_sents = [(pi, s) for pi, p in enumerate(pars) for s in p["sents"]]

    # sentence openers
    opener2 = collections.Counter()
    for _, s in all_sents:
        w = [x.lower() for x in ALPHA_WORDS.findall(s["text"])[:2]]
        if len(w) == 2:
            opener2[" ".join(w)] += 1
    out["openers"] = [{"opener": k, "count": v} for k, v in opener2.most_common() if v >= 3][:15]

    for pi, p in enumerate(pars):
        firsts = []
        for s in p["sents"]:
            w = ALPHA_WORDS.findall(s["text"])
            firsts.append(w[0].lower() if w else "")
        i = 0
        while i < len(firsts):
            j = i
            while j + 1 < len(firsts) and firsts[j + 1] == firsts[i] and firsts[i]:
                j += 1
            if j - i + 1 >= 3:
                out["runs"].append({"paragraph": pi + 1, "word": firsts[i], "sentences": j - i + 1})
            i = j + 1

    # repeated content words inside a paragraph
    for pi, p in enumerate(pars):
        if p["kind"] != "prose":
            continue
        toks = [w.lower() for w in ALPHA_WORDS.findall(p["text"])]
        if len(toks) < 80:
            continue
        cnt = collections.Counter(w for w in toks if len(w) >= 5 and w not in STOP)
        for w, c in cnt.most_common():
            if c >= 4:
                out["words"].append({"paragraph": pi + 1, "word": w, "count": c})
    out["words"] = out["words"][:40]

    # near duplicate sentences
    sets = []
    for pi, s in all_sents:
        cs = {w.lower() for w in ALPHA_WORDS.findall(s["text"]) if w.lower() not in STOP and len(w) > 2}
        sets.append(cs)
    idx = [i for i, cs in enumerate(sets) if len(cs) >= 6]
    if len(idx) > 1500:
        out["dupes_skipped"] = True
    else:
        for a in range(len(idx)):
            sa = sets[idx[a]]
            for b in range(a + 1, len(idx)):
                sb = sets[idx[b]]
                if min(len(sa), len(sb)) / max(len(sa), len(sb)) < 0.6:
                    continue
                inter = len(sa & sb)
                if inter and inter / len(sa | sb) >= 0.6:
                    out["dupes"].append({"score": round(inter / len(sa | sb), 2),
                                         "a": all_sents[idx[a]][1]["text"], "b": all_sents[idx[b]][1]["text"]})
        out["dupes"].sort(key=lambda d: -d["score"])
        out["dupes"] = out["dupes"][:20]

    # repeated 4-word phrases
    toks = [w.lower() for w in ALPHA_WORDS.findall(" ".join(s["text"] for _, s in all_sents))]
    grams = collections.Counter(tuple(toks[i:i + 4]) for i in range(len(toks) - 3))
    for g, c in grams.most_common(400):
        if c < 2:
            break
        if sum(1 for w in g if w not in STOP) >= 2:
            out["phrases"].append({"phrase": " ".join(g), "count": c})
        if len(out["phrases"]) >= 15:
            break

    content = [w for w in toks if w not in STOP and len(w) >= 4]
    total = max(1, len(toks))
    out["top_words"] = [{"word": w, "count": c, "per 1000 words": round(c * 1000 / total, 1)}
                        for w, c in collections.Counter(content).most_common(15)]
    return out


def check_structure(pars, cite_positions):
    rows = []
    for pi, p in enumerate(pars):
        if p["kind"] != "prose":
            continue
        sents = p["sents"]
        if not sents:
            continue
        n = len(sents)
        wc = sum(s["wc"] for s in sents)
        first = sents[0]
        advice = []
        topic = 8 <= first["wc"] <= 40 and not WEAK_OPENER.match(first["text"])
        if not topic:
            if WEAK_OPENER.match(first["text"]):
                advice.append("Open with your own point, not a connector or a bare pronoun.")
            elif first["wc"] < 8:
                advice.append("First sentence is very short. State the paragraph's claim in full.")
            else:
                advice.append("First sentence is too long to work as a clear topic sentence.")
        i = bisect.bisect_left(cite_positions, p["start"])
        has_cit = i < len(cite_positions) and cite_positions[i] < p["end"]
        evidence = has_cit or bool(EVIDENCE_RE.search(p["text"]))
        if not evidence:
            advice.append("No citation, number, or example. Back the claim up.")
        tail = " ".join(s["text"] for s in sents[-2:])
        analysis = bool(ANALYSIS_RE.search(tail))
        if not analysis:
            advice.append("Finish with the so-what: what does the evidence mean for your argument?")
        length_ok = 3 <= n <= 9 and wc <= 220
        if not length_ok:
            advice.append("Too thin (under 3 sentences)." if n < 3 else "Too long. Split it into two paragraphs.")
        rows.append({"paragraph": pi + 1, "starts with": p["text"][:70].replace("\n", " ") + ("..." if len(p["text"]) > 70 else ""),
                     "sentences": n, "words": wc, "topic sentence": topic, "evidence": evidence,
                     "so-what": analysis, "length": length_ok, "fix": " ".join(advice)})
    return rows


def find_monotony(pars):
    runs = []
    for pi, p in enumerate(pars):
        L = [s["wc"] for s in p["sents"]]
        if len(L) < 4:
            continue
        run = [0]
        for i in range(1, len(L) + 1):
            if i < len(L) and abs(L[i] - L[i - 1]) <= 3:
                run.append(i)
            else:
                if len(run) >= 4:
                    runs.append({"paragraph": pi + 1, "sentences": len(run),
                                 "avg words": round(sum(L[k] for k in run) / len(run), 1)})
                run = [i]
    return runs


# =============================================================================
# Main analysis
# =============================================================================

@st.cache_data(show_spinner=False, max_entries=24)
def analyze(raw_text: str, long_threshold: int = 30) -> dict:
    text = normalize(raw_text)
    body, refs_text, refs_pos = split_references(text)
    pars = parse_document(body)
    prose = [p for p in pars if p["kind"] == "prose"]
    sents = [s for p in prose for s in p["sents"]]

    cites = find_citations(body)
    cite_positions = sorted(c["pos"] for c in cites)
    entries = parse_reference_entries(refs_text) if refs_text else []
    refcheck = match_references(cites, entries, bool(refs_text))
    claims = find_uncited_claims(pars, cite_positions)

    wording = find_wording(prose)
    wording_spans = find_wording_spans(body, prose)
    passive_spans, passive_ids = find_passive_spans(body, prose)
    rep = find_repetition(prose)
    structure = check_structure(pars, cite_positions)
    monotony = find_monotony(prose)

    lens = [s["wc"] for s in sents]
    all_words = [w for s in sents for w in WORDS.findall(s["text"])]
    n_words, n_sents = len(all_words), len(sents)
    syl = sum(count_syllables(w) for w in all_words)
    avg_len = n_words / n_sents if n_sents else 0
    flesch = 206.835 - 1.015 * avg_len - 84.6 * (syl / n_words) if n_words else 0
    grade = 0.39 * avg_len + 11.8 * (syl / n_words) - 15.59 if n_words else 0
    long_sents = [s for s in sents if s["wc"] > long_threshold]
    long_spans = [(s["start"], s["end"]) for s in long_sents]
    filler_hits = sum(g["count"] for g in wording)

    metrics = {
        "words": n_words, "sentences": n_sents, "paragraphs": len(prose),
        "avg sentence length": round(avg_len, 1),
        "median sentence length": round(statistics.median(lens), 1) if lens else 0,
        "sentence length spread (stdev)": round(statistics.stdev(lens), 1) if len(lens) > 1 else 0,
        "Flesch reading ease": round(flesch, 1), "grade level": round(grade, 1),
        "passive sentences %": round(100 * len(passive_ids) / n_sents, 1) if n_sents else 0,
        "long sentences": len(long_sents),
        "wording flags": filler_hits,
        "wording flags per 1000 words": round(filler_hits * 1000 / n_words, 1) if n_words else 0,
        "citations found": len(cites),
        "claims needing a source (high)": sum(1 for c in claims if c["severity"] == "high"),
        "claims needing a source (all)": len(claims),
        "cited but not in references": len(refcheck["missing"]) + len(refcheck["typos"])
        + (len(refcheck["numeric"]["out_of_range"]) if refcheck["numeric"] else 0),
        "references never cited": len(refcheck["unused"])
        + (len(refcheck["numeric"]["unused"]) if refcheck["numeric"] else 0),
        "repeated sentence openers": len(rep["openers"]) + len(rep["runs"]),
        "monotone stretches": len(monotony),
        "paragraphs failing a structure check": sum(1 for r in structure if not all(
            [r["topic sentence"], r["evidence"], r["so-what"], r["length"]])),
    }
    return {
        "text": text, "body": body, "refs_text": refs_text, "paragraphs": pars, "sentences": sents,
        "citations": cites, "refcheck": refcheck, "ref_entries": entries, "claims": claims,
        "wording": wording, "wording_spans": wording_spans, "passive_spans": passive_spans,
        "passive_count": len(passive_ids), "long_sentences": sorted(long_sents, key=lambda s: -s["wc"]),
        "long_spans": long_spans, "repetition": rep, "structure": structure, "monotony": monotony,
        "lengths": lens, "metrics": metrics, "long_threshold": long_threshold,
    }


# =============================================================================
# Source overlap
# =============================================================================

_TOK = re.compile(r"[A-Za-z0-9]+")


@st.cache_data(show_spinner=False, max_entries=12)
def overlap_analysis(draft: str, sources: tuple, n: int = 8) -> dict:
    body, _, _ = split_references(normalize(draft))
    dtoks = [(m.group().lower(), m.start(), m.end()) for m in _TOK.finditer(body)]
    words = [t[0] for t in dtoks]
    results, union = [], [False] * len(words)
    for name, stext in sources:
        stoks = [m.group().lower() for m in _TOK.finditer(normalize(stext))]
        if len(stoks) < n:
            results.append({"source": name, "percent": 0.0, "passages": [], "words": 0})
            continue
        grams = {tuple(stoks[i:i + n]) for i in range(len(stoks) - n + 1)}
        covered = [False] * len(words)
        for i in range(len(words) - n + 1):
            if tuple(words[i:i + n]) in grams:
                for k in range(i, i + n):
                    covered[k] = True
        passages, i = [], 0
        while i < len(words):
            if covered[i]:
                j = i
                while j + 1 < len(words) and covered[j + 1]:
                    j += 1
                a, b = dtoks[i][1], dtoks[j][2]
                before = body[max(0, a - 3):a].strip()
                after = body[b:b + 3].strip()
                quoted = bool(re.search(r"[\"“‘']$", before)) or bool(re.match(r"[\"”’']", after))
                ctx = body[a:b + 140]
                cited = bool(PAREN_CIT.search(ctx) or NARR_CIT.search(ctx) or NUM_CIT.search(ctx))
                if quoted and cited:
                    status = "Quoted and cited (fine)"
                elif quoted:
                    status = "Quoted, but no citation nearby"
                elif cited:
                    status = "Cited, but the wording is copied: put it in quotes or rewrite it"
                else:
                    status = "Copied wording, no quote, no citation: rewrite it and cite the source"
                passages.append({"words": j - i + 1, "status": status, "text": " ".join(body[a:b].split())})
                i = j + 1
            else:
                i += 1
        for k, c in enumerate(covered):
            if c:
                union[k] = True
        passages.sort(key=lambda p: -p["words"])
        results.append({"source": name, "percent": round(100 * sum(covered) / max(1, len(words)), 1),
                        "passages": passages, "words": sum(covered)})
    return {"sources": results, "total_percent": round(100 * sum(union) / max(1, len(words)), 1),
            "n": n, "draft_words": len(words)}


# =============================================================================
# Highlighting, diffs, reports
# =============================================================================

F_LONG, F_PASSIVE, F_WORDING, F_CLAIM = 1, 2, 4, 8


def highlight_html(a: dict) -> str:
    body = a["body"]
    flags = bytearray(len(body))
    for s, e in a["long_spans"]:
        for i in range(s, e):
            flags[i] |= F_LONG
    for s, e in a["passive_spans"]:
        for i in range(s, e):
            flags[i] |= F_PASSIVE
    for s, e in a["wording_spans"]:
        for i in range(s, e):
            flags[i] |= F_WORDING
    for c in a["claims"]:
        if c["severity"] in ("high", "medium"):
            for i in range(c["start"], c["end"]):
                flags[i] |= F_CLAIM
    out, i, n = [], 0, len(body)
    while i < n:
        j = i
        while j < n and flags[j] == flags[i]:
            j += 1
        f, chunk = flags[i], html.escape(body[i:j])
        if f:
            style = []
            if f & F_WORDING:
                style.append("background:#ffc9c9")
            elif f & F_PASSIVE:
                style.append("background:#ffd8a8")
            elif f & F_LONG:
                style.append("background:#fff3bf")
            if f & F_CLAIM:
                style.append("border-bottom:2px solid #339af0")
            out.append(f'<span style="{";".join(style)}">{chunk}</span>')
        else:
            out.append(chunk)
        i = j
    return ('<div style="white-space:pre-wrap;line-height:1.75;font-size:15px;background:#fff;color:#1a1a1a;'
            'padding:18px 22px;border-radius:8px;border:1px solid #ddd;font-family:Georgia,serif">'
            + "".join(out) + "</div>")


LEGEND_HTML = (
    '<span style="background:#fff3bf;color:#111;padding:2px 8px;border-radius:4px">long sentence</span> '
    '<span style="background:#ffd8a8;color:#111;padding:2px 8px;border-radius:4px">possible passive voice</span> '
    '<span style="background:#ffc9c9;color:#111;padding:2px 8px;border-radius:4px">filler or buzzword</span> '
    '<span style="border-bottom:2px solid #339af0;padding:2px 8px">claim needing a source</span>'
)


def _word_diff(a: str, b: str) -> str:
    ta, tb = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.append(html.escape(" ".join(ta[i1:i2])))
        else:
            if i2 > i1:
                out.append(f'<del style="background:#ffc9c9;color:#111">{html.escape(" ".join(ta[i1:i2]))}</del>')
            if j2 > j1:
                out.append(f'<ins style="background:#b2f2bb;color:#111;text-decoration:none">{html.escape(" ".join(tb[j1:j2]))}</ins>')
    return " ".join(out)


def diff_html(old: str, new: str) -> str:
    A = [p.strip() for p in re.split(r"\n\s*\n", normalize(old)) if p.strip()]
    B = [p.strip() for p in re.split(r"\n\s*\n", normalize(new)) if p.strip()]
    sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out += [f"<p>{html.escape(p)}</p>" for p in A[i1:i2]]
        elif tag == "delete":
            out += [f'<p><del style="background:#ffc9c9;color:#111">{html.escape(p)}</del></p>' for p in A[i1:i2]]
        elif tag == "insert":
            out += [f'<p><ins style="background:#b2f2bb;color:#111;text-decoration:none">{html.escape(p)}</ins></p>' for p in B[j1:j2]]
        else:
            la, lb = A[i1:i2], B[j1:j2]
            for k in range(max(len(la), len(lb))):
                out.append("<p>" + _word_diff(la[k] if k < len(la) else "", lb[k] if k < len(lb) else "") + "</p>")
    return ('<div style="line-height:1.7;font-size:15px;background:#fff;color:#1a1a1a;padding:14px 20px;'
            'border-radius:8px;border:1px solid #ddd;font-family:Georgia,serif">' + "".join(out) + "</div>")


# direction: -1 lower is better, +1 higher is better, 0 neutral
COMPARE_KEYS = [
    ("words", 0), ("avg sentence length", 0), ("Flesch reading ease", 0), ("grade level", 0),
    ("passive sentences %", -1), ("long sentences", -1), ("wording flags per 1000 words", -1),
    ("claims needing a source (high)", -1), ("claims needing a source (all)", -1),
    ("cited but not in references", -1), ("references never cited", -1),
    ("repeated sentence openers", -1), ("monotone stretches", -1),
    ("paragraphs failing a structure check", -1), ("citations found", 0),
]


def compare_table(m_old: dict, m_new: dict) -> pd.DataFrame:
    rows = []
    for key, direction in COMPARE_KEYS:
        o, n = m_old[key], m_new[key]
        d = round(n - o, 1)
        verdict = "" if d == 0 or direction == 0 else ("better" if d * direction > 0 else "worse")
        rows.append({"measure": key, "earlier": o, "newer": n, "change": d, "verdict": verdict})
    return pd.DataFrame(rows)


def top_fixes(a: dict) -> list[str]:
    m, fixes = a["metrics"], []
    if m["claims needing a source (high)"]:
        fixes.append(f"Add sources to {m['claims needing a source (high)']} claims that cite figures, research, or experts without a citation (Citations tab).")
    if a["refcheck"]["note"]:
        fixes.append(a["refcheck"]["note"])
    if m["cited but not in references"]:
        fixes.append(f"{m['cited but not in references']} in-text citations are missing from your reference list or look misspelled.")
    if m["references never cited"]:
        fixes.append(f"{m['references never cited']} reference "+("entry is" if m['references never cited']==1 else "entries are")+" never cited in the text. Cite or remove.")
    if m["passive sentences %"] > 25:
        fixes.append(f"{m['passive sentences %']}% of sentences look passive. Aim under 25% by naming who does what.")
    if m["long sentences"]:
        fixes.append(f"{m['long sentences']} sentences run past {a['long_threshold']} words. Split the longest first.")
    if m["wording flags per 1000 words"] > 15:
        fixes.append(f"Heavy filler: {m['wording flags per 1000 words']} flags per 1000 words. Work through the Wording tab.")
    if m["repeated sentence openers"]:
        fixes.append("Several sentences start the same way. Vary the openers (Repetition tab).")
    if m["monotone stretches"]:
        fixes.append("Some stretches have sentences of nearly the same length. Mix short and long (Readability tab).")
    if m["paragraphs failing a structure check"]:
        fixes.append(f"{m['paragraphs failing a structure check']} paragraphs miss a topic sentence, evidence, or a closing so-what (Structure tab).")
    if a["repetition"]["dupes"]:
        fixes.append(f"{len(a['repetition']['dupes'])} pairs of sentences say almost the same thing. Merge or cut one.")
    return fixes or ["Nothing major flagged. Read it aloud once for tone."]


def issues_dataframe(a: dict, overlap: dict | None = None) -> pd.DataFrame:
    rows = []
    for c in a["claims"]:
        rows.append(("Claim without citation", c["severity"], c["reasons"], c["text"]))
    for r in a["refcheck"]["missing"]:
        rows.append(("Cited, not in references", "high", f"cited {r['count']}x", r["cite"]))
    for r in a["refcheck"]["typos"]:
        rows.append(("Possible citation typo", "medium", f"maybe {r['maybe']}", r["cite"]))
    for e in a["refcheck"]["unused"]:
        rows.append(("Reference never cited", "medium", "", e))
    for g in a["wording"]:
        rows.append(("Wording: " + g["category"], "low", f"try: {g['fix']} (x{g['count']})", g["phrase"]))
    for s in a["long_sentences"]:
        rows.append(("Long sentence", "low", f"{s['wc']} words", s["text"]))
    for r in a["structure"]:
        if r["fix"]:
            rows.append(("Paragraph structure", "medium", f"paragraph {r['paragraph']}", r["fix"]))
    for d in a["repetition"]["dupes"]:
        rows.append(("Near-duplicate sentences", "medium", f"similarity {d['score']}", d["a"] + "  //  " + d["b"]))
    if overlap:
        for s in overlap["sources"]:
            for p in s["passages"]:
                rows.append(("Source overlap", "high" if "Copied wording" in p["status"] else "medium",
                             f"{s['source']}: {p['status']}", p["text"]))
    return pd.DataFrame(rows, columns=["type", "severity", "detail", "text"])


def build_markdown_report(name: str, a: dict, overlap: dict | None = None) -> str:
    m = a["metrics"]
    L = [f"# Paper Coach report: {name}", f"_Generated {datetime.datetime.now():%Y-%m-%d %H:%M}_", "", "## Priority fixes"]
    L += [f"{i}. {f}" for i, f in enumerate(top_fixes(a), 1)]
    L += ["", "## Measures", "| measure | value |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in m.items()]
    L += ["", f"Reading level: {flesch_label(m['Flesch reading ease'])}"]
    L += ["", "## Claims without a citation"]
    L += [f"- ({c['severity']}) {c['text']}  _[{c['reasons']}]_" for c in a["claims"]] or ["- none"]
    rc = a["refcheck"]
    L += ["", "## Reference check"]
    L += [f"- Cited but not in references: {r['cite']} (x{r['count']})" for r in rc["missing"]]
    L += [f"- Possible typo: {r['cite']} (maybe {r['maybe']})" for r in rc["typos"]]
    L += [f"- Never cited: {e}" for e in rc["unused"]]
    if rc["numeric"]:
        L += [f"- Numbered citations out of range: {rc['numeric']['out_of_range']}",
              f"- Numbered references never cited: {rc['numeric']['unused']}"]
    if not (rc["missing"] or rc["typos"] or rc["unused"]):
        L += ["- no mismatches found"]
    L += ["", "## Wording flags", "| phrase | type | count | try |", "|---|---|---|---|"]
    L += [f"| {g['phrase']} | {g['category']} | {g['count']} | {g['fix']} |" for g in a["wording"]]
    L += ["", f"## Sentences longer than {a['long_threshold']} words"]
    L += [f"- ({s['wc']}) {s['text']}" for s in a["long_sentences"]] or ["- none"]
    L += ["", "## Paragraph structure"]
    L += [f"- Paragraph {r['paragraph']}: {r['fix']}" for r in a["structure"] if r["fix"]] or ["- all paragraphs pass"]
    rep = a["repetition"]
    L += ["", "## Repetition"]
    L += [f"- Opener repeated {o['count']}x: {o['opener']}" for o in rep["openers"]]
    L += [f"- Paragraph {r['paragraph']}: {r['sentences']} sentences in a row start with {r['word']}" for r in rep["runs"]]
    L += [f"- Near-duplicate ({d['score']}): {d['a']}  //  {d['b']}" for d in rep["dupes"]]
    if overlap:
        L += ["", f"## Source overlap (matching runs of {overlap['n']}+ words)",
              f"Total draft covered by sources: {overlap['total_percent']}%"]
        for s in overlap["sources"]:
            L += [f"### {s['source']}: {s['percent']}%"]
            L += [f"- ({p['words']} words) {p['status']}: {p['text']}" for p in s["passages"]]
    return "\n".join(L)


def build_html_report(name: str, a: dict, overlap: dict | None = None) -> str:
    e = html.escape
    m = a["metrics"]
    rows = "".join(f"<tr><td>{e(str(k))}</td><td>{e(str(v))}</td></tr>" for k, v in m.items())
    fixes = "".join(f"<li>{e(f)}</li>" for f in top_fixes(a))
    claims = "".join(f"<li><b>{e(c['severity'])}</b> {e(c['text'])} <i>[{e(c['reasons'])}]</i></li>" for c in a["claims"]) or "<li>none</li>"
    wording = "".join(f"<tr><td>{e(g['phrase'])}</td><td>{e(g['category'])}</td><td>{g['count']}</td><td>{e(g['fix'])}</td></tr>" for g in a["wording"])
    struct = "".join(f"<li>Paragraph {r['paragraph']}: {e(r['fix'])}</li>" for r in a["structure"] if r["fix"]) or "<li>all pass</li>"
    ov = ""
    if overlap:
        ov = f"<h2>Source overlap</h2><p>Total draft covered by sources: {overlap['total_percent']}%</p>"
        for s in overlap["sources"]:
            ov += f"<h3>{e(s['source'])}: {s['percent']}%</h3><ul>" + "".join(
                f"<li>({p['words']} words) <b>{e(p['status'])}</b><br>{e(p['text'])}</li>" for p in s["passages"]) + "</ul>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Paper Coach report</title>
<style>body{{font-family:Georgia,serif;max-width:920px;margin:30px auto;padding:0 16px;color:#1a1a1a;line-height:1.55}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:5px 8px;text-align:left}}h1,h2{{font-family:Arial,sans-serif}}</style>
</head><body><h1>Paper Coach report: {e(name)}</h1><p>Generated {datetime.datetime.now():%Y-%m-%d %H:%M}</p>
<h2>Priority fixes</h2><ol>{fixes}</ol><h2>Measures</h2><table>{rows}</table>
<h2>Claims without a citation</h2><ul>{claims}</ul>
<h2>Wording flags</h2><table><tr><th>phrase</th><th>type</th><th>count</th><th>try</th></tr>{wording}</table>
<h2>Paragraph structure</h2><ul>{struct}</ul>{ov}
<h2>Highlighted draft</h2><p>{LEGEND_HTML}</p>{highlight_html(a)}</body></html>"""


# =============================================================================
# Sample text
# =============================================================================

SAMPLE = """Employee Motivation and Organizational Performance

In today's fast-paced world, employee motivation plays a crucial role in organizational performance. Studies show that motivated employees are 21% more productive than disengaged ones. It is important to note that motivation is a very complex thing and various aspects of the workplace impact it.

Herzberg (1959) argued that hygiene factors prevent dissatisfaction while motivators create satisfaction. This distinction was later challenged by House and Wigdor (1967), who found that the two-factor split was not supported across occupations. The debate matters because managers who rely on pay alone are often disappointed by the results. As a result, modern job design pairs fair pay with meaningful work.

Many managers believe that bonuses always improve performance. Bonuses are given to staff every year. Bonuses are given to staff in every department. The company was found to have spent 4 million dollars on bonuses in 2019. Leadership is also important. Leadership is also a key driver. Leadership is also needed in teams.

Deci and Ryan (2000) showed that autonomy, competence, and relatedness drive intrinsic motivation. Teams that control how they do their work report higher commitment (Gagne & Deci, 2005; Smyth, 2012). The fact that autonomy matters in terms of retention suggests that firms should leverage flexible scheduling to build a robust culture. This means retention policies need to address autonomy first, and pay second.

References

Deci, E. L., & Ryan, R. M. (2000). The what and why of goal pursuits. Psychological Inquiry, 11(4), 227-268.

Gagne, M., & Deci, E. L. (2005). Self-determination theory and work motivation. Journal of Organizational Behavior, 26(4), 331-362.

Herzberg, F. (1959). The motivation to work. John Wiley.

Locke, E. A. (1976). The nature and causes of job satisfaction. Rand McNally.

Smith, J. (2012). Flexible work and retention. Management Review, 5(2), 10-20.
"""


# =============================================================================
# UI
# =============================================================================

def _df(rows, **kw):
    st.dataframe(pd.DataFrame(rows), hide_index=True, **kw)


def _load_text(upload, pasted):
    if upload is not None:
        text, err = extract_text(upload.name, upload.getvalue())
        return text, upload.name, err
    return pasted or "", "pasted text", None


def main():
    st.set_page_config(page_title="Paper Coach", page_icon="📝", layout="wide")
    st.title("📝 Paper Coach")
    st.caption("Citation, structure, and style checks for academic and management papers. "
               "Everything runs on your computer. Nothing is uploaded anywhere.")

    with st.sidebar:
        st.header("Settings")
        long_thr = st.slider("Long sentence limit (words)", 20, 50, 30)
        ngram = st.slider("Source overlap: matching run length (words)", 6, 15, 8,
                          help="Lower finds shorter matches but more false alarms.")
        st.markdown("---")
        st.markdown("**How to use**\n1. Add your draft in the Input tab\n2. Read Summary first\n3. Fix, re-upload, then use Compare to see progress")

    tabs = st.tabs(["Input", "Summary", "Citations", "Readability", "Wording", "Repetition",
                    "Structure", "Highlights", "Source overlap", "Compare drafts", "Export"])
    t_in, t_sum, t_cit, t_read, t_word, t_rep, t_struct, t_hi, t_ov, t_cmp, t_exp = tabs

    # ---------------------------------------------------------------- input
    with t_in:
        c1, c2 = st.columns([3, 2])
        with c1:
            upload = st.file_uploader("Upload your draft (.docx, .pdf, .txt, .md)", type=["docx", "pdf", "txt", "md"], key="draft_up")
            st.text_area("Or paste your text here", height=320, key="paste",
                         placeholder="Paste the paper here. Put your reference list under a heading called References.")
            st.button("Load a sample paper to try it", on_click=lambda: st.session_state.update(paste=SAMPLE))
        with c2:
            src_files = st.file_uploader("Optional: upload sources to check for copied wording",
                                         type=["docx", "pdf", "txt", "md"], accept_multiple_files=True, key="src_up")
            st.info("Tip: for the reference check, keep your list under a heading on its own line: References, Bibliography, or Works Cited. "
                    "Both author-year (Smith, 2020) and numbered [1] styles work.")

    text, name, err = _load_text(upload, st.session_state.get("paste", ""))
    if err:
        st.error(err)
        text = ""
    ready = len(text.split()) >= 40
    a = analyze(text, long_thr) if ready else None

    def need_text():
        st.info("Add at least 40 words in the Input tab to see results here.")

    # -------------------------------------------------------------- summary
    with t_sum:
        if not ready:
            need_text()
        else:
            m = a["metrics"]
            st.subheader(f"Results for {name}")
            c = st.columns(5)
            c[0].metric("Words", m["words"])
            c[1].metric("Sentences", m["sentences"])
            c[2].metric("Avg sentence", m["avg sentence length"])
            c[3].metric("Flesch ease", m["Flesch reading ease"], flesch_label(m["Flesch reading ease"]), delta_color="off")
            c[4].metric("Grade level", m["grade level"])
            c = st.columns(5)
            c[0].metric("Passive sentences", f"{m['passive sentences %']}%")
            c[1].metric("Long sentences", m["long sentences"])
            c[2].metric("Wording flags /1000", m["wording flags per 1000 words"])
            c[3].metric("Claims needing source", m["claims needing a source (high)"])
            c[4].metric("Citations found", m["citations found"])
            st.subheader("Fix these first")
            for i, f in enumerate(top_fixes(a), 1):
                st.markdown(f"**{i}.** {f}")
            st.caption("These checks are rule based. They point at likely problems. You decide what is a real one.")

    # ------------------------------------------------------------ citations
    with t_cit:
        if not ready:
            need_text()
        else:
            rc = a["refcheck"]
            st.subheader("Claims that need a source")
            if a["claims"]:
                sev_order = {"high": 0, "medium": 1, "low": 2}
                rows = sorted(a["claims"], key=lambda c: (sev_order[c["severity"]], c["start"]))
                _df([{"severity": c["severity"], "paragraph": c["para"], "why": c["reasons"],
                      "sentence": c["text"], "note": c["note"]} for c in rows])
            else:
                st.success("No unsupported claims detected.")
            st.subheader("Reference list check")
            if rc["note"]:
                st.warning(rc["note"])
            else:
                c = st.columns(4)
                c[0].metric("Citations in text", a["metrics"]["citations found"])
                c[1].metric("Reference entries", rc["n_entries"])
                c[2].metric("Cited, not listed", a["metrics"]["cited but not in references"])
                c[3].metric("Listed, never cited", a["metrics"]["references never cited"])
                if rc["missing"]:
                    st.markdown("**Cited in the text but missing from the reference list**")
                    _df(rc["missing"])
                if rc["typos"]:
                    st.markdown("**Probably misspelled citations**")
                    _df(rc["typos"])
                if rc["unused"]:
                    st.markdown("**In the reference list but never cited**")
                    for e in rc["unused"]:
                        st.markdown(f"- {e}")
                if rc["numeric"]:
                    st.markdown("**Numbered citations**")
                    st.write(f"Out of range: {rc['numeric']['out_of_range'] or 'none'}. Never cited: {rc['numeric']['unused'] or 'none'}.")
                if not (rc["missing"] or rc["typos"] or rc["unused"] or (rc["numeric"] and (rc["numeric"]["out_of_range"] or rc["numeric"]["unused"]))):
                    st.success("Citations and reference list match.")
            with st.expander("How I read your reference list (check this parsed correctly)"):
                for e in a["ref_entries"]:
                    st.markdown(f"- {e}")

    # ---------------------------------------------------------- readability
    with t_read:
        if not ready:
            need_text()
        else:
            m = a["metrics"]
            c = st.columns(4)
            c[0].metric("Flesch reading ease", m["Flesch reading ease"])
            c[1].metric("Grade level", m["grade level"])
            c[2].metric("Median sentence", m["median sentence length"])
            c[3].metric("Length spread (stdev)", m["sentence length spread (stdev)"])
            st.caption("Most academic writing lands between 30 and 50 on Flesch. A spread under 6 usually means flat rhythm.")
            lens = a["lengths"]
            st.subheader("Sentence length, in order")
            st.bar_chart(pd.DataFrame({"words": lens}))
            short = sum(1 for x in lens if x < 10)
            mid = sum(1 for x in lens if 10 <= x < 25)
            long_ = sum(1 for x in lens if x >= 25)
            tot = max(1, len(lens))
            st.write(f"Short (under 10): **{100 * short // tot}%**   Medium (10 to 24): **{100 * mid // tot}%**   Long (25+): **{100 * long_ // tot}%**")
            if a["monotony"]:
                st.subheader("Flat rhythm")
                st.caption("Four or more sentences in a row with nearly the same length.")
                _df(a["monotony"])
            st.subheader(f"Sentences over {a['long_threshold']} words")
            if a["long_sentences"]:
                _df([{"words": s["wc"], "sentence": s["text"]} for s in a["long_sentences"]])
            else:
                st.success("No sentences over the limit.")
            st.write(f"Possible passive sentences: **{a['passive_count']}** ({a['metrics']['passive sentences %']}%). See them in the Highlights tab.")

    # -------------------------------------------------------------- wording
    with t_word:
        if not ready:
            need_text()
        else:
            if not a["wording"]:
                st.success("No filler or buzzwords flagged.")
            else:
                cats = sorted({g["category"] for g in a["wording"]})
                pick = st.multiselect("Show types", cats, default=cats)
                _df([{"phrase": g["phrase"], "type": g["category"], "count": g["count"], "try instead": g["fix"], "example": g["example"]}
                     for g in a["wording"] if g["category"] in pick])
                st.caption("Some flags are fine in context (a precise use of significant, for example). Judge each one.")

    # ----------------------------------------------------------- repetition
    with t_rep:
        if not ready:
            need_text()
        else:
            rep = a["repetition"]
            st.subheader("Repeated sentence openers")
            if rep["openers"] or rep["runs"]:
                if rep["openers"]:
                    _df(rep["openers"])
                if rep["runs"]:
                    st.markdown("**Runs of sentences starting with the same word**")
                    _df(rep["runs"])
            else:
                st.success("Openers vary well.")
            st.subheader("Words overused inside one paragraph")
            _df(rep["words"]) if rep["words"] else st.success("None found.")
            st.subheader("Near-duplicate sentences")
            if rep["dupes_skipped"]:
                st.info("Document too large for this check.")
            elif rep["dupes"]:
                for d in rep["dupes"]:
                    st.markdown(f"**Similarity {d['score']}**\n\n- {d['a']}\n- {d['b']}")
            else:
                st.success("No near-duplicates.")
            st.subheader("Repeated phrases")
            _df(rep["phrases"]) if rep["phrases"] else st.success("None found.")
            with st.expander("Most used words"):
                _df(rep["top_words"])

    # ------------------------------------------------------------ structure
    with t_struct:
        if not ready:
            need_text()
        else:
            rows = a["structure"]
            if not rows:
                st.info("No prose paragraphs found.")
            else:
                ok = sum(1 for r in rows if not r["fix"])
                st.write(f"**{ok} of {len(rows)}** paragraphs pass all four checks: clear topic sentence, evidence, closing so-what, sensible length.")
                df = pd.DataFrame(rows)
                for col in ("topic sentence", "evidence", "so-what", "length"):
                    df[col] = df[col].map({True: "✓", False: "✗"})
                st.dataframe(df, hide_index=True)

    # ----------------------------------------------------------- highlights
    with t_hi:
        if not ready:
            need_text()
        else:
            st.markdown(LEGEND_HTML, unsafe_allow_html=True)
            st.markdown("")
            st.markdown(highlight_html(a), unsafe_allow_html=True)

    # -------------------------------------------------------------- overlap
    ov = None
    with t_ov:
        if not ready:
            need_text()
        elif not src_files:
            st.info("Upload the sources you used (in the Input tab). This tab then shows any stretch of your draft that copies their wording, "
                    "and whether you quoted and cited it. It only compares against files you give it.")
        else:
            srcs, errs = [], []
            for f in src_files:
                t, e = extract_text(f.name, f.getvalue())
                (errs.append(e) if e else srcs.append((f.name, t)))
            for e in errs:
                st.error(e)
            if srcs:
                ov = overlap_analysis(text, tuple(srcs), ngram)
                st.metric("Draft words inside matching runs", f"{ov['total_percent']}%")
                _df([{"source": s["source"], "% of draft": s["percent"], "matched words": s["words"], "passages": len(s["passages"])} for s in ov["sources"]])
                for s in ov["sources"]:
                    with st.expander(f"{s['source']}: {len(s['passages'])} passages"):
                        for p in s["passages"]:
                            st.markdown(f"**{p['status']}** ({p['words']} words)\n\n> {p['text']}")
                st.caption("Fix by quoting with a citation, or by rewriting the idea in your own sentences and citing the source.")

    # -------------------------------------------------------------- compare
    with t_cmp:
        if not ready:
            need_text()
        else:
            st.write("The draft in the Input tab is the newer version. Add the earlier version here.")
            cu = st.file_uploader("Earlier version (.docx, .pdf, .txt, .md)", type=["docx", "pdf", "txt", "md"], key="old_up")
            cp = st.text_area("Or paste the earlier version", height=180, key="old_paste")
            old_text, old_name, old_err = _load_text(cu, cp)
            if old_err:
                st.error(old_err)
            elif len(old_text.split()) >= 40:
                ao = analyze(old_text, long_thr)
                df = compare_table(ao["metrics"], a["metrics"])
                st.subheader("Progress")
                st.dataframe(df, hide_index=True)
                st.subheader("What changed")
                st.markdown('<span style="background:#ffc9c9;color:#111;padding:2px 8px;border-radius:4px">removed</span> '
                            '<span style="background:#b2f2bb;color:#111;padding:2px 8px;border-radius:4px">added</span>', unsafe_allow_html=True)
                st.markdown(diff_html(old_text, text), unsafe_allow_html=True)

    # --------------------------------------------------------------- export
    with t_exp:
        if not ready:
            need_text()
        else:
            md = build_markdown_report(name, a, ov)
            st.download_button("Download report (Markdown)", md, file_name="paper_coach_report.md", mime="text/markdown")
            st.download_button("Download report (HTML, open and print to PDF)", build_html_report(name, a, ov),
                               file_name="paper_coach_report.html", mime="text/html")
            st.download_button("Download all issues (CSV)", issues_dataframe(a, ov).to_csv(index=False),
                               file_name="paper_coach_issues.csv", mime="text/csv")
            with st.expander("Preview the Markdown report"):
                st.markdown(md)


if __name__ == "__main__":
    main()
