"""Build docs/report/Final_Report.docx in the course format, using DA3_Report.docx as the template.

The title page, introduction, literature survey, architecture figures and references are taken
from the DA3 report (with "planned" statements updated); every other section is regenerated
with the same paragraph, table, caption and figure formatting. Numbers come from results/.

    python scripts/report_figures.py        # figures first
    python scripts/build_final_report.py
"""

from __future__ import annotations

import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

TEMPLATE = Path("docs/report/DA3_Report.docx")
OUT = Path("docs/report/Final_Report.docx")
FIG = Path("docs/figures/final")
FONT = ('<w:rFonts w:ascii="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman" '
        'w:hAnsi="Times New Roman"/>')

# ----------------------------------------------------------------------------- XML helpers


def run(text: str, b=False, i=False, sz=24) -> str:
    fmt = ("<w:b/><w:bCs/>" if b else "") + ("<w:i/><w:iCs/>" if i else "")
    return (f'<w:r><w:rPr>{FONT}{fmt}<w:color w:val="000000"/><w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/></w:rPr>'
            f'<w:t xml:space="preserve">{escape(text)}</w:t></w:r>')


def rich(text: str, sz=24) -> str:
    """Runs from text with **bold** and *italic* marks."""
    out = []
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if part.startswith("**"):
            out.append(run(part[2:-2], b=True, sz=sz))
        elif part.startswith("*") and len(part) > 1:
            out.append(run(part[1:-1], i=True, sz=sz))
        elif part:
            out.append(run(part, sz=sz))
    return "".join(out)


def p(text: str) -> str:
    return ('<w:p><w:pPr><w:spacing w:after="120" w:before="0" w:line="300"/><w:jc w:val="both"/></w:pPr>'
            f"{rich(text)}</w:p>")


def num(n: int, text: str, lead: str | None = None) -> str:
    lead_xml = run(lead, b=True) + run(" ") if lead else ""
    return ('<w:p><w:pPr><w:tabs><w:tab w:val="left" w:pos="567"/></w:tabs><w:spacing w:after="80" w:line="300"/>'
            '<w:ind w:left="567" w:hanging="397"/><w:jc w:val="both"/></w:pPr>'
            f"{run(f'{n}.' + chr(9))}{lead_xml}{rich(text)}</w:p>")


def numbered(items, start=1) -> list[str]:
    out = []
    for k, it in enumerate(items, start):
        out.append(num(k, it[1], it[0]) if isinstance(it, tuple) else num(k, it))
    return out


def bullet(text: str) -> str:
    return ('<w:p><w:pPr><w:pStyle w:val="ListParagraph"/><w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr>'
            '<w:spacing w:after="80" w:line="300"/><w:jc w:val="both"/></w:pPr>' f"{rich(text)}</w:p>")


def h1(text: str, page_break=False) -> str:
    pb = "<w:pageBreakBefore/>" if page_break else ""
    sp = '<w:spacing w:after="160" w:before="0"/>' if page_break else '<w:spacing w:after="160" w:before="360"/>'
    return f'<w:p><w:pPr><w:pStyle w:val="Heading1"/>{pb}{sp}</w:pPr>{run(text, b=True, sz=28)}</w:p>'


def h2(text: str) -> str:
    return ('<w:p><w:pPr><w:pStyle w:val="Heading2"/><w:spacing w:after="120" w:before="240"/></w:pPr>'
            f"{run(text, b=True)}</w:p>")


def h3(text: str) -> str:
    return ('<w:p><w:pPr><w:pStyle w:val="Heading3"/><w:spacing w:after="80" w:before="180"/></w:pPr>'
            f"{run(text, b=True, i=True)}</w:p>")


def formula(text: str) -> str:
    return ('<w:p><w:pPr><w:spacing w:after="120" w:before="60"/><w:jc w:val="center"/></w:pPr>'
            f"{run(text, i=True)}</w:p>")


class Counter:
    table = 0
    figure = 0


def table_caption(text: str) -> str:
    Counter.table += 1
    return ('<w:p><w:pPr><w:keepNext/><w:spacing w:after="80" w:before="160"/><w:jc w:val="center"/></w:pPr>'
            f"{run(f'Table {Counter.table}:', b=True, sz=22)}{run(' ' + text, sz=22)}</w:p>")


def figure_caption(text: str) -> str:
    Counter.figure += 1
    return ('<w:p><w:pPr><w:spacing w:after="200"/><w:jc w:val="center"/></w:pPr>'
            f"{run(f'Figure {Counter.figure}:', b=True, sz=22)}{run(' ' + text, sz=22)}</w:p>")


_BORDER = ('<w:tcBorders><w:top w:val="single" w:color="000000" w:sz="4"/><w:left w:val="single" w:color="000000" '
           'w:sz="4"/><w:bottom w:val="single" w:color="000000" w:sz="4"/><w:right w:val="single" w:color="000000" '
           'w:sz="4"/></w:tcBorders><w:tcMar><w:top w:type="dxa" w:w="50"/><w:left w:type="dxa" w:w="80"/>'
           '<w:bottom w:type="dxa" w:w="50"/><w:right w:type="dxa" w:w="80"/></w:tcMar><w:vAlign w:val="center"/>')


def _cell(text: str, w: int, header: bool, align: str) -> str:
    b = "<w:b/><w:bCs/>" if header else '<w:b w:val="false"/><w:bCs w:val="false"/>'
    runs = "".join(f'<w:r><w:rPr>{FONT}{b}<w:color w:val="000000"/><w:sz w:val="20"/><w:szCs w:val="20"/></w:rPr>'
                   f'<w:t xml:space="preserve">{escape(str(part))}</w:t></w:r>' for part in [text])
    return (f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{w}"/>{_BORDER}</w:tcPr><w:p><w:pPr><w:spacing w:after="0"/>'
            f'<w:jc w:val="{"center" if header else align}"/></w:pPr>{runs}</w:p></w:tc>')


def table(caption: str, header: list[str], rows: list[list], widths: list[float] | None = None,
          align: str | None = None) -> list[str]:
    """Table in the DA3 style. ``widths`` are relative; ``align`` gives l/c per column."""
    n = len(header)
    widths = widths or [1] * n
    total = 9638
    w = [int(total * x / sum(widths)) for x in widths]
    w[-1] = total - sum(w[:-1])
    align = align or ("l" + "c" * (n - 1))
    al = ["left" if a == "l" else "center" for a in align]
    grid = "".join(f'<w:gridCol w:w="{x}"/>' for x in w)
    xml = ('<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="9638"/><w:tblBorders><w:top w:val="single" w:color="auto" '
           'w:sz="4"/><w:left w:val="single" w:color="auto" w:sz="4"/><w:bottom w:val="single" w:color="auto" '
           'w:sz="4"/><w:right w:val="single" w:color="auto" w:sz="4"/><w:insideH w:val="single" w:color="auto" '
           'w:sz="4"/><w:insideV w:val="single" w:color="auto" w:sz="4"/></w:tblBorders><w:tblLayout w:type="fixed"/>'
           f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>')
    xml += "<w:tr><w:trPr><w:tblHeader/></w:trPr>" + "".join(_cell(h, w[k], True, "center")
                                                            for k, h in enumerate(header)) + "</w:tr>"
    for r in rows:
        xml += "<w:tr><w:trPr><w:cantSplit/></w:trPr>" + "".join(_cell(c, w[k], False, al[k])
                                                               for k, c in enumerate(r)) + "</w:tr>"
    xml += "</w:tbl>"
    return [table_caption(caption), xml, '<w:p><w:pPr><w:spacing w:after="120"/></w:pPr></w:p>']


class Media:
    """New images: copied into word/media with relationships."""

    def __init__(self):
        self.items: list[tuple[str, Path]] = []
        self.next_id = 200

    def image(self, path: Path, width_cm: float, descr: str) -> str:
        rid = f"rIdFinal{len(self.items) + 1}"
        self.items.append((rid, path))
        wpx, hpx = Image.open(path).size
        cx = int(width_cm * 360000)
        cy = int(cx * hpx / wpx)
        self.next_id += 1
        return ('<w:p><w:pPr><w:keepNext/><w:spacing w:after="60" w:before="120"/><w:jc w:val="center"/></w:pPr><w:r>'
                f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
                f'<wp:effectExtent t="0" r="0" b="0" l="0"/><wp:docPr id="{self.next_id}" name="{path.stem}" '
                f'descr="{escape(descr)}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks '
                'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/>'
                '</wp:cNvGraphicFramePr><a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic '
                'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr>'
                '<pic:cNvPr id="0" name="" descr=""/><pic:cNvPicPr><a:picLocks noChangeAspect="1" '
                f'noChangeArrowheads="1"/></pic:cNvPicPr></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}" '
                'cstate="none"/><a:srcRect/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr bwMode="auto">'
                f'<a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/>'
                '</a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')


# ----------------------------------------------------------------------------- template handling


def split_body(xml: str):
    start = xml.index("<w:body>") + 8
    end = xml.rindex("</w:body>")
    body, els, i = xml[start:end], [], 0
    pat = re.compile(r"<(w:p|w:tbl|w:sectPr)[ >]")
    while i < len(body):
        m = pat.search(body, i)
        if not m:
            break
        tag, depth, j = m.group(1), 0, None
        for t in re.finditer(r"<(/?)" + tag + r"([ >/])", body[m.start():]):
            pos = m.start() + t.start()
            close = body.index(">", pos)
            if t.group(1) == "/":
                depth -= 1
            elif body[close - 1] == "/":
                continue
            else:
                depth += 1
            if depth == 0:
                j = close + 1
                break
        els.append(body[m.start():j])
        i = j
    return xml[:start], els, xml[end:]


def text_of(e: str) -> str:
    return "".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", e))


def find(els, heading: str, start=0) -> int:
    for k in range(start, len(els)):
        if text_of(els[k]) == escape(heading) and "Heading" in els[k]:
            return k
    raise KeyError(heading)


def find_text(els, prefix: str, start=0) -> int:
    for k in range(start, len(els)):
        if text_of(els[k]).startswith(escape(prefix)):
            return k
    raise KeyError(prefix)


def _forms(s: str) -> list[str]:
    """XML spellings of a text: the template writes apostrophes either plainly or as &apos;."""
    e = escape(s)
    return [e, e.replace("'", "&apos;")] if "'" in s else [e]


def replace_in(e: str, old: str, new: str) -> str:
    for o in _forms(old):
        if e.count(o) == 1:
            return e.replace(o, escape(new))
    raise ValueError(f"expected one occurrence of: {old[:60]}")


def substitute(els: list[str], subs: dict[str, str]) -> list[str]:
    """Apply each text substitution exactly once across ``els``; fail loudly otherwise, so a template
    sentence can never silently stay out of date."""
    els = list(els)
    for old, new in subs.items():
        hits = [k for k, e in enumerate(els) if any(f in e for f in _forms(old))]
        if len(hits) != 1:
            raise ValueError(f"substitution matched {len(hits)} elements: {old[:70]}")
        els[hits[0]] = replace_in(els[hits[0]], old, new)
    return els




# ----------------------------------------------------------------------------- content


def abstract() -> list[str]:
    return [h1("Abstract", page_break=True), *numbered([
        ("Motivation, aim and objective.",
         "Atrial fibrillation (AF) is the most common sustained heart-rhythm disorder; it is often silent, it raises "
         "the risk of stroke about five-fold, and it is treatable once found (Wolf et al., 1991; US Preventive "
         "Services Task Force, 2018). Remote photoplethysmography (rPPG) recovers the pulse from tiny, "
         "heartbeat-driven colour changes of the skin in an ordinary video, so a contactless rhythm check is "
         "possible in principle (Yan et al., 2018). The obstacle is that anything that disturbs beat timing makes "
         "a normal rhythm look irregular, which is what AF looks like, so poor signals produce false AF alarms. "
         "The aim was a machine-vision system that screens for AF from facial RGB video with few false alarms, "
         "flags slow, fast and extra (premature) heartbeats, and says when it cannot be trusted."),
        ("About methodology.",
         "MediaPipe FaceLandmarker gives 478 3D face landmarks and the head pose in every frame. The forehead and "
         "cheeks are divided into up to 25 mesh-based patches, and the POS algorithm turns each patch's colour "
         "trace into a pulse. Every second, each patch is trusted by four checks that do not assume a regular "
         "rhythm (orientation, brightness, consistency with the other patches, beat agreement); trusted patches "
         "are fused, head motion is regressed out, and a rhythm-independent confidence is computed. Beats are "
         "located with sub-frame precision. A random forest on five beat-interval irregularity features, trained "
         "on MIMIC PERform AF with camera-like timing noise, screens 30 s windows for AF after isolated extra beats "
         "are removed by a timing rule tuned on MIT-BIH Arrhythmia."),
        ("Outcome.",
         "The system outputs heart rate, a rhythm result (“regular rhythm”, “possible AF – "
         "see a doctor” or “signal unreliable”), the number of extra beats, slow or fast heart-rate "
         "flags, RMSSD and a confidence, and runs live on a webcam with the patch trust map drawn on the face. "
         "Breathing rate was built and evaluated but is not shown, because it proved unreliable. Only part of "
         "UBFC-rPPG could be downloaded and the AF face-video dataset (OBF) was not available, so the face-video "
         "evaluation uses 12 normal-rhythm videos, and AF detection is evaluated on contact pulse signals."),
        ("Efficiency attained.",
         "On MIMIC PERform AF (35 patients, leave-one-subject-out) the AF classifier reached an AUC of 0.96 on "
         "camera-like pulses, detecting all 19 AF patients and clearing 14 of 16 non-AF patients; confidence "
         "gating raised the AUC to 0.99 while keeping 96% of AF windows, whereas a spectral-sharpness gate would "
         "have discarded 94–100% of them. On the 12 healthy face videos, AF false alarms fell from 11 "
         "(GREEN) and 2 (CHROM) to 1 (POS on one region) and 0 (the proposed method). Heart-rate error on the "
         "face videos was 0.2 BPM, beat-interval error 21 ms, extra-beat removal nearly halved AF false alarms "
         "caused by premature beats on MIT-BIH, the slow/fast flag agreed with the ECG in 95% of segments, and processing "
         "takes about 10 ms per frame on a laptop CPU."),
    ]), ('<w:p><w:pPr><w:spacing w:after="120" w:before="120" w:line="300"/></w:pPr>'
         + run("Keywords: ", b=True) + run("remote photoplethysmography; atrial fibrillation screening; multi-patch "
                                           "signal fusion; signal quality assessment; premature beats; machine "
                                           "vision") + "</w:p>")]


def methodology_new(media: Media, fig4: list[str]) -> list[str]:
    out = [h3("3.8 AF screening"),
           p("Rhythm features are computed from the inter-beat intervals (IBI) of non-overlapping 30 s windows, "
             "after implausible intervals (outside 0.3–2.0 s) and isolated extra beats (Section 3.9) are "
             "removed. Figure 4 shows why they work: a regular rhythm changes its beat gaps smoothly and slightly, "
             "while AF changes them randomly and strongly."),
           *fig4,
           formula("RMSSD = √( mean( (IBI_{i+1} − IBI_i)² ) ),     nRMSSD = RMSSD / mean(IBI)"),
           bullet("**Normalised RMSSD**: beat-to-beat change independent of heart rate."),
           bullet("**Shannon entropy** of the IBI histogram (16 bins, after dropping the 8 shortest and 8 longest "
                  "intervals, as in Dash et al., 2009) and **sample entropy** (m = 2, r = 0.2 SD; Richman & Moorman, "
                  "2000) measure unpredictability."),
           bullet("**Poincaré SD1/SD2**: short- against long-term variability; near 1 for AF, small for a "
                  "normal rhythm."),
           bullet("**Turning-point ratio**: the share of intervals that are a local maximum or minimum; about 2/3 for "
                  "random intervals."),
           p("**Classifier.** A random forest (Breiman, 2001) of 200 trees (minimum 5 windows per leaf, balanced "
             "class weights; scikit-learn, Pedregosa et al., 2011) was chosen over logistic regression after "
             "subject-independent cross-validation. It is trained on MIMIC PERform AF (Charlton et al., 2022) after "
             "resampling the 125 Hz finger PPG to 30 fps and adding 16 ms of random timing noise to every beat, the "
             "beat-timing error measured on the UBFC face videos. Without this, camera timing noise alone made "
             "healthy faces look like AF. Only timing features are used, so the model transfers from contact to "
             "camera pulses."),
           p("**Gating and decision.** A window is judged only when its beat-shape consistency is at least 0.8; "
             "otherwise the output is “signal unreliable”. A video or live window is called “possible "
             "AF” when the mean AF probability of its reliable windows is at least 0.5."),
           h3("3.9 Extra beats and heart-rate flags"),
           p("A beat is flagged as premature when the interval before it is more than 8% shorter than the local "
             "median (8 intervals on each side, leaving out the two around the beat) and the interval after it is "
             "longer than 1.05 times that median:"),
           formula("IBI_before < 0.92 · median,     IBI_after > 1.05 · median"),
           p("The two thresholds were chosen by grid search on half of MIT-BIH Arrhythmia (DS1 of de Chazal et al., "
             "2004) with 16 ms timing noise, and tested on the other half (DS2). Before the AF check, an extra beat "
             "and its two intervals are removed only if it is isolated, that is, if the intervals just outside its "
             "short-long pair are within 20% of the local median and at most 20% of the window's beats are "
             "flagged. This follows the definition of an extra beat inside an otherwise regular rhythm (Chong et "
             "al., 2015) and stops the rule from “cleaning” real AF, whose neighbouring intervals are "
             "irregular too. The setting was chosen on DS1 as the one with the fewest false alarms that loses at "
             "most 2 points of AF sensitivity."),
           p("A slow or fast flag is raised when the median heart rate over at least 2 minutes of confident windows "
             "is below 50 BPM (bradycardia; Kusumoto et al., 2019) or above 100 BPM, and is shown with context "
             "(“can be normal for fit people”)."),
           h3("3.10 Breathing rate, heart rate in irregular rhythms and HRV"),
           p("Breathing modulates the pulse in amplitude (RIAV), beat intervals (RIFV) and intensity (RIIV). Each "
             "beat-by-beat series is resampled to 4 Hz, band-passed to 6–36 breaths/min and its spectral peak "
             "taken; the estimates are averaged when their standard deviation is at most 4 breaths/min and "
             "discarded otherwise (Smart Fusion; Karlen et al., 2013). RIFV is dropped in irregular windows. "
             "Because POS removes changes slower than its 1.6 s window, RIIV is taken from a separate trust-weighted, "
             "inverted green-channel pulse that keeps slow changes."),
           p("Heart rate is the spectral peak of the last 8 s in a regular rhythm and 60 / mean IBI when the rhythm "
             "is irregular (normalised RMSSD above 0.1), because an irregular rhythm has no single spectral peak. A "
             "breathing-aware variant that skips spectral peaks at the breathing rate and its 2nd and 3rd harmonics "
             "was also evaluated. RMSSD and SDNN are computed from the cleaned IBIs."),
           h3("3.11 Comparison variants and baselines"),
           p("To measure what each ingredient contributes, the method runs as seven variants, each adding one "
             "ingredient: single (one skin region, plain POS), patch_mean (patches with equal weights), +facing, "
             "+brightness, +consistency, +beats and full (adding motion removal, the proposed method). Two further "
             "baselines replace POS with GREEN (the inverted, normalised green channel; Verkruysse et al., 2008) or "
             "CHROM (de Haan & Jeanne, 2013) on one skin region. Every variant feeds the same beat detector and the "
             "same AF classifier, so differences come from pulse extraction and fusion alone."),
           h3("3.12 Evaluation against a contact sensor"),
           p("A finger or ear sensor sees each beat at a different time than the face (pulse transit time), and the "
             "camera pulse is inverted relative to a contact PPG (more blood means less reflected light, while "
             "oximeters plot more blood upwards). Neither affects heart rate or beat intervals, but both break "
             "beat-time matching. Before beats are matched, the reference is therefore shifted by the delay "
             "(within ±1 s) and sign that maximise its correlation with the face pulse. The fitted delays were "
             "0.10–0.21 s on every video, and the beat F1 rose from 0.72 to 0.98."),
           h3("3.13 Live demo"),
           p("The live demo processes a webcam (or a video file played as if live). Each frame's landmarks and patch "
             "colours enter a rolling 32 s buffer with their real timestamps, which is resampled to 30 fps. Once a "
             "second, a background thread runs the full method on a snapshot of the buffer, so the video keeps its "
             "frame rate, and the screen shows the trust map, heart rate, confidence, rhythm status with the "
             "number of extra beats, RMSSD and, after 2 minutes of confident readings, the slow/fast flag, always "
             "with the note “screening aid, not a diagnosis”.")]
    return out


def methods_table(old_rows) -> list[str]:
    extra = [["Random forest on IBI features, trained with camera-like timing noise",
              "Explainable features; transfers from contact PPG; robust to camera timing jitter",
              "Trained on 35 ICU patients; contact PPG, not face video"],
             ["Extra-beat rule with neighbour check", "Removes isolated premature beats without erasing AF",
              "Finds ventricular beats well, but most atrial premature beats are too slightly early"],
             ["Breathing from pulse modulations (Smart Fusion)", "No extra sensor; standard method",
              "Unreliable on ICU data and in AF; not shown as an output"]]
    return table("Advantages and disadvantages of the methods used", ["Method", "Advantages", "Disadvantages"],
                 old_rows + extra, [2585, 3526, 3527], "lll")


def implementation() -> list[str]:
    out = [h1("Implementation Details"), h2("1. Hardware / Software")]
    out += table("Hardware and software used", ["Item", "Details"], [
        ["Development and experiments", "Windows 11 laptop, Intel CPU (20 threads), no GPU; earlier phases in a "
                                        "Linux cloud container (4 cores)"],
        ["Camera", "Any RGB webcam or phone camera at about 30 fps; phone videos in H.264"],
        ["Language", "Python 3.12 (3.11–3.13 supported)"],
        ["Face tracking", "MediaPipe 1.0.1, FaceLandmarker task (float16 model, 3.7 MB)"],
        ["Numerical and signal processing", "NumPy 2.4.6, SciPy 1.17.1"],
        ["Video and image processing", "OpenCV 5.0 (headless)"],
        ["Machine learning", "scikit-learn 1.9.1 (k-means for patches; random-forest AF classifier)"],
        ["Data handling and plots", "pandas 3.0.6, Matplotlib 3.11.2"],
        ["ECG tools", "wfdb 4.3.1 (MIT-BIH reading; XQRS R-peak detector for the MIMIC ECG references)"],
        ["Testing", "pytest 9.1.1 (72 tests)"],
        ["Version control", "Git and GitHub (repository mayankoff/vision)"],
        ["Development assistant", "An AI coding assistant, see Prompt Engineering"],
    ], [2, 5], "ll")
    out += [h2("2. Any other"), h3("2.1 Code organisation")]
    out += table("Modules of the implementation (package afscreen)", ["Module", "Role"], [
        ["config.py", "All settings in one place (band, windows, thresholds), so experiments stay comparable"],
        ["face/tracker.py, face/patches.py", "MediaPipe landmarks and head pose; patch definitions, measurement, "
                                             "merging to 12/6/1 patches, level choice"],
        ["traces.py", "Runs tracking and measurement over a video and caches the patch traces"],
        ["signal/pos.py, baselines.py", "POS (vectorised); GREEN and CHROM baselines"],
        ["signal/beats.py, quality.py, spectrum.py", "Beats with sub-frame refinement, beat-shape consistency, "
                                                    "spectrum, plain and breathing-aware FFT heart rate"],
        ["signal/resp.py, hrv.py", "Breathing rate from pulse modulations; RMSSD and SDNN"],
        ["core/trust.py, core/fusion.py", "Trust checks, weighted fusion, motion removal, confidence"],
        ["pipeline.py", "One call per video and variant; the seven variants and two baselines"],
        ["af/features.py, af/experiment.py, af/screen.py", "IBI features; training and cross-validation on MIMIC; "
                                                          "AF screening of face videos with gating"],
        ["flags/ectopy.py, flags/experiment.py", "Extra-beat rule and its removal; MIT-BIH experiments; slow/fast "
                                                "flags"],
        ["vitals.py, eval/*", "Phase 2 experiments; metrics, reference alignment, benchmarks, Phase 5 comparisons"],
        ["live.py", "Live demo (webcam or file), background analysis, on-screen display"],
        ["io/datasets.py, af/mimic.py, flags/mitbih.py", "Readers for UBFC-rPPG (both sets), PURE, synthetic "
                                                        "videos, MIMIC PERform and MIT-BIH, straight from the zips"],
        ["cli.py, scripts/", "Command line (run, extract, evaluate, af-train, af-screen, extra-beats, hr-flags, "
                             "vitals, compare, live); figures, report, synthetic data, Windows setup"],
    ], [2.3, 5], "ll")
    out += [h3("2.2 Synthetic data with known beat times"),
            p("Real datasets give a reference pulse from a contact sensor, but not the exact time of every beat. A "
              "synthetic generator therefore produces regular, AF-like, slow, fast and extra-beat rhythms beat by "
              "beat, and turns them into per-patch colour traces (with optional noisy, shadowed, turned-away "
              "patches and tinted-light head motion) or into pulsing face videos made from a still photo (with "
              "optional small faces and side-to-side sway). It is used for the stress tests in Section Results "
              "1.2 and for end-to-end tests of the live demo."),
            h3("2.3 Testing"),
            p("The implementation has 72 automatic tests (plus 2 optional end-to-end tests that run MediaPipe on "
              "synthetic face videos). They cover filtering, POS and its vectorised form, the baselines, beat "
              "detection and refinement, patches, each trust check, motion removal, the AF features and "
              "classifier, the extra-beat rule, breathing, the dataset readers, reference alignment and the live "
              "monitor. All of them pass, in about 25 s."),
            h3("2.4 Datasets and access")]
    out += table("Datasets: planned and used", ["Dataset", "Contents", "Use", "Status"], [
        ["UBFC-rPPG DATASET_2 and DATASET_1", "Face videos + finger PPG, normal rhythm",
         "Heart rate, beat timing, AF false alarms", "Partly downloaded: 12 videos, 7 with a reference"],
        ["MIMIC PERform AF", "PPG + ECG, 19 AF and 16 non-AF ICU patients, 20 min",
         "Train and validate the AF classifier", "Used (complete)"],
        ["MIMIC PERform Testing (adults)", "PPG + ECG + impedance respiration, 100 ICU adults, 10 min",
         "Breathing rate, heart rate", "Used"],
        ["MIT-BIH Arrhythmia", "ECG with every beat labelled, 48 records", "Extra-beat rule; second AF test",
         "Used (44 non-paced records)"],
        ["Synthetic videos and traces", "Known beat times", "Stress tests, demo checks", "Generated"],
        ["PURE, OBF, COHFACE, UBFC-Phys", "Head motion; AF face videos; breathing belt; talking",
         "Motion removal; AF on faces; breathing", "Not used (access on request or not obtained in time)"],
    ], [2.2, 2.6, 2.3, 2.5], "llll")
    out += [h3("2.5 How to run")]
    out += [bullet("Install with scripts\\windows\\setup.ps1 (or pip), which also downloads the face model and runs "
                   "the tests."),
            bullet("afscreen live starts the webcam demo; afscreen run video.mp4 --debug-video trust.mp4 analyses one "
                   "video."),
            bullet("python scripts/extract_from_zips.py datasets/UBFC caches the face videos; afscreen evaluate, "
                   "af-train, af-screen, extra-beats, hr-flags, vitals and compare reproduce every result in this "
                   "report (see the repository README)."),
            bullet("python scripts/synthetic_stress.py, scripts/report_figures.py and scripts/build_final_report.py "
                   "regenerate the stress tests, the figures and this document.")]
    return out


def results() -> list[str]:
    out = [h1("Results"), h2("1. Results obtained"),
           p("All numbers below were produced by the project code from the datasets in Table 6 and can be "
             "regenerated with the commands in Implementation Details 2.5. The face-video set is small (12 "
             "videos, 7 with a reference), so differences of one video or one window should not be over-read."),
           h3("1.1 Evaluation data and protocol"),
           bullet("**Face videos:** UBFC-rPPG subjects 25 and 26 (DATASET_2) and 5, 8, 10, 11 and 12 (DATASET_1) "
                  "have a finger-PPG reference; five more DATASET_2 videos (1, 4, 5, 27, 48) have none. Every UBFC "
                  "subject has a normal rhythm, so any AF result on them is a false alarm."),
           bullet("**AF classifier:** MIMIC PERform AF, 30 s windows, leave-one-subject-out cross-validation: every "
                  "patient is tested by a model that never saw them."),
           bullet("**Extra beats:** MIT-BIH Arrhythmia beat annotations, tuned on DS1 and tested on DS2, scored "
                  "outside AF episodes; paced records excluded."),
           bullet("**Breathing and heart rate:** MIMIC PERform Testing adults and the AF set, PPG resampled to "
                  "30 fps, against impedance respiration and ECG R-peaks."),
           h3("1.2 Synthetic stress tests: what the trust checks and motion removal do"),
           p("The UBFC videos are still and evenly lit, so the trust checks have nothing to correct there. Six "
             "simulated scenarios of 40 s with 25 patches show what they do when part of the face is bad (mean of "
             "three random seeds)."),
           *table("Beat-interval error (ms) of each variant on synthetic patch traces; lower is better",
                  ["Scenario", "single", "patch mean", "+facing", "+bright-ness", "+consis-tency", "+beats", "full"],
                  [["Clean, regular 72 BPM", "7.8", "4.0", "4.0", "4.0", "4.0", "4.1", "4.1"],
                   ["8 noisy patches", "45.3", "8.9", "8.9", "8.9", "4.9", "4.9", "4.9"],
                   ["Shadowed and turned patches", "41.1", "7.5", "6.8", "4.6", "4.6", "4.6", "4.6"],
                   ["Head motion artifact", "77.3", "76.1", "76.1", "76.1", "76.2", "76.2", "8.2"],
                   ["AF, clean", "8.6", "5.9", "5.9", "5.9", "6.0", "5.9", "5.9"],
                   ["AF, 8 noisy patches", "48.8", "10.6", "10.6", "10.6", "6.3", "6.3", "6.3"]],
                  [3, 1, 1, 1, 1, 1, 1, 1]),
           *table("Effect of motion removal on a regular 72 BPM rhythm with head motion (beat-interval error, ms)",
                  ["Motion frequency", "Artifact (x pulse)", "Without motion removal", "With motion removal"],
                  [["0.8 Hz", "10", "63.6", "7.1"], ["1.0 Hz", "3", "76.2", "8.2"], ["1.0 Hz", "10", "90.9", "14.2"],
                   ["1.3 Hz", "7", "241.1", "14.5"]]),
           h3("1.3 Heart rate and beat timing on the UBFC face videos"),
           *table("Baselines and ablation on the 7 UBFC videos with a reference",
                  ["Variant", "HR MAE (BPM)", "Window HR MAE (BPM)", "IBI error (ms)", "Beat F1", "RMSSD error (ms)"],
                  [["GREEN, one region", "0.58", "2.33", "55.4", "0.84", "157.0"],
                   ["CHROM, one region", "0.23", "1.17", "43.5", "0.94", "89.3"],
                   ["POS, one region (single)", "0.22", "0.43", "21.1", "0.98", "31.4"],
                   ["+ patches, equal weights", "0.22", "0.48", "22.2", "0.98", "39.4"],
                   ["+ facing", "0.23", "0.47", "21.3", "0.98", "41.6"],
                   ["+ brightness", "0.23", "0.46", "21.3", "0.98", "41.5"],
                   ["+ consistency", "0.23", "0.46", "21.1", "0.98", "38.5"],
                   ["+ beat agreement", "0.23", "0.46", "21.8", "0.99", "39.2"],
                   ["+ motion removal (full)", "0.23", "0.46", "21.8", "0.99", "39.2"]],
                  [3, 1.2, 1.4, 1.2, 1, 1.3]),
           h3("1.4 AF classifier on MIMIC PERform AF"),
           *table("AF classifier, leave-one-subject-out (35 patients; 1,399–1,400 windows of 30 s)",
                  ["Signal", "Model", "AUC", "Window sens.", "Window spec.", "AF patients found",
                   "Non-AF patients cleared"],
                  [["Contact PPG, 125 Hz", "Random forest", "0.94", "0.95", "0.86", "19 / 19", "14 / 16"],
                   ["Resampled to 30 fps", "Random forest", "0.96", "0.94", "0.88", "19 / 19", "14 / 16"],
                   ["30 fps + 16 ms timing noise", "Logistic regression", "0.91", "0.87", "0.83", "18 / 19", "14 / 16"],
                   ["30 fps + 16 ms timing noise (final)", "Random forest", "0.96", "0.93", "0.89", "19 / 19",
                    "14 / 16"],
                   ["Same, gated (beat shape ≥ 0.8)", "Random forest", "0.99", "0.96", "0.95", "19 / 19",
                    "14 / 16"]],
                  [2.6, 1.6, 0.8, 1, 1, 1.1, 1.2]),
           *table("Share of windows kept by a quality gate (MIMIC, 30 fps + 16 ms noise)",
                  ["Gate", "Threshold", "AF windows kept", "Non-AF windows kept"],
                  [["Beat-shape consistency (used)", "0.7", "0.99", "0.96"],
                   ["Beat-shape consistency (used)", "0.8", "0.96", "0.91"],
                   ["Beat-shape consistency (used)", "0.9", "0.76", "0.85"],
                   ["Spectral peak sharpness", "0.5", "0.06", "0.86"],
                   ["Spectral peak sharpness", "0.7", "0.00", "0.67"]]),
           h3("1.5 AF false alarms on the face videos"),
           *table("AF screening of the 12 normal-rhythm UBFC videos (same classifier, extra beats removed, gated)",
                  ["Variant", "Windows called AF", "Videos called AF"],
                  [["GREEN, one region", "14 / 17", "11 / 12"], ["CHROM, one region", "4 / 20", "2 / 12"],
                   ["POS, one region", "2 / 20", "1 / 12"], ["POS + patches, equal weights", "1 / 20", "0 / 12"],
                   ["+ facing, + brightness, + consistency", "0 / 20", "0 / 12"],
                   ["+ beat agreement, full method", "1 / 20", "0 / 12"],
                   ["Full method without extra-beat removal", "2 / 20", "1 / 12"],
                   ["Full method, classifier trained without timing noise", "2 / 20", "2 / 12"]],
                  [3.5, 1.5, 1.5]),
           h3("1.6 Extra beats on MIT-BIH Arrhythmia"),
           *table("Extra-beat rule on DS2 (test half), outside AF episodes",
                  ["Beat-timing noise", "Sensitivity", "Precision", "F1", "Ventricular found", "Atrial found",
                   "False flags per 1000 normal beats"],
                  [["0 ms (exact)", "0.54", "0.92", "0.68", "0.82", "0.14", "5.5"],
                   ["16 ms (measured on faces)", "0.54", "0.86", "0.66", "0.81", "0.15", "10.7"],
                   ["33 ms (one frame at 30 fps)", "0.54", "0.61", "0.57", "0.77", "0.20", "43.1"]],
                  [2.2, 1, 1, 0.8, 1.1, 1, 1.4]),
           *table("MIMIC-trained AF classifier on MIT-BIH 30 s windows (16 ms noise): share called AF",
                  ["Windows", "Split", "Number", "No removal", "Plain removal", "Removal of isolated extra beats"],
                  [["AF (sensitivity)", "DS1", "59", "0.97", "0.92", "0.98"],
                   ["AF (sensitivity)", "DS2", "170", "0.90", "0.71", "0.82"],
                   ["Normal rhythm with extra beats", "DS1", "602", "0.13", "0.06", "0.07"],
                   ["Normal rhythm with extra beats", "DS2", "443", "0.22", "0.09", "0.12"],
                   ["Normal rhythm without extra beats", "DS1", "630", "0.002", "0.002", "0.002"],
                   ["Normal rhythm without extra beats", "DS2", "644", "0.003", "0.003", "0.005"]],
                  [2.6, 0.7, 0.9, 1, 1, 1.6]),
           h3("1.7 Slow and fast heart-rate flags"),
           *table("Flag from camera-like PPG beats versus flag from the ECG (MIMIC, 2-minute segments)",
                  ["Reference \\ prediction", "Slow", "Normal", "Fast"],
                  [["Slow (< 50 BPM)", "3", "0", "0"], ["Normal", "0", "223", "2"], ["Fast (> 100 BPM)", "0", "15",
                                                                                    "90"]]),
           p("333 of 350 segments (95%) were scored; agreement was 0.95, Cohen's kappa 0.88 and the heart-rate error "
             "2.8 BPM. Fifteen of the 17 disagreements are AF patients whose ECG rate is fast but whose pulse rate "
             "reads normal."),
           h3("1.8 Breathing rate and heart rate in irregular rhythms"),
           *table("Breathing rate (64 s windows, wide-band pulse, three modulations, Smart Fusion) against impedance "
                  "respiration; “constant guess” always predicts the median rate (17.9 /min)",
                  ["Group", "Windows", "Coverage", "MAE (/min)", "Within 2 /min", "Constant guess: MAE / within 2"],
                  [["ICU non-AF", "179", "0.80", "3.8", "0.43", "3.3 / 0.21"],
                   ["ICU adults (Testing)", "573", "0.79", "5.5", "0.39", "4.1 / 0.26"],
                   ["ICU AF", "142", "0.82", "9.8", "0.07", "5.7 / 0.24"]],
                  [2.2, 1, 1, 1, 1.1, 2]),
           *table("Heart-rate error per 30 s window against the ECG (BPM; share of windows off by more than 10 BPM)",
                  ["Group", "Windows", "FFT peak", "Breathing-aware FFT", "From beats", "Combined"],
                  [["ICU AF", "760", "12.5 (35%)", "12.0 (35%)", "7.4 (19%)", "7.4 (19%)"],
                   ["ICU non-AF", "602", "2.2 (6%)", "2.4 (6%)", "2.6 (6%)", "2.6 (6%)"],
                   ["ICU adults (Testing)", "1989", "8.5 (15%)", "8.5 (16%)", "8.8 (16%)", "9.0 (17%)"]],
                  [2, 1, 1.2, 1.5, 1.2, 1.2]),
           h3("1.9 Beat timing and heart-rate variability on the face videos"),
           *table("Placement of beats between frames (single region, 7 UBFC videos of about 1 minute)",
                  ["Beats placed", "IBI error (ms)", "RMSSD error (ms)", "SDNN error (ms)", "RMSSD camera / reference"],
                  [["Whole frames", "25.4", "36.6", "18.6", "75.6 / 39.0"],
                   ["Parabola through each peak (used)", "21.0", "31.3", "16.8", "70.3 / 39.0"],
                   ["Cubic spline", "21.1", "31.6", "16.9", "70.6 / 39.0"]],
                  [2.6, 1, 1.1, 1.1, 1.6]),
           h3("1.10 Patch count and face size"),
           *table("Full method with a fixed or adaptive number of patches, on the 7 videos re-processed at smaller "
                  "frame sizes (HR MAE in BPM / IBI error in ms)",
                  ["Frame size", "Skin pixels", "6 patches", "12 patches", "25 patches", "Adaptive (chose)"],
                  [["640 x 480", "4696", "0.23 / 21.5", "0.23 / 20.6", "0.22 / 20.3", "0.23 / 21.8 (6 or 12)"],
                   ["320 x 240", "1243", "0.23 / 22.4", "0.22 / 22.0", "0.21 / 22.6", "0.23 / 22.4 (6)"],
                   ["160 x 120", "350", "0.23 / 22.2", "0.22 / 22.6", "0.21 / 21.8", "0.23 / 22.2 (6)"]],
                  [1.3, 1, 1.2, 1.2, 1.2, 1.7]),
           h3("1.11 Live demo and processing speed"),
           *table("Live demo on files played as if live (status after 35 s)",
                  ["Input", "Truth", "Shown"],
                  [["UBFC DATASET_1 subject 5 (real video)", "Oximeter 75–84 BPM, rising",
                    "74–86 BPM, follows the rise; confidence 0.93–0.96; regular rhythm"],
                   ["Synthetic, regular 72 BPM", "Regular", "72 BPM; regular rhythm; 0 extra beats"],
                   ["Synthetic, AF about 86 BPM", "AF", "83–87 BPM from beats; possible AF"],
                   ["Synthetic, regular with extra beats", "Regular, about 4 extra beats per 30 s",
                    "70 BPM; regular rhythm; 4 extra beats"]],
                  [2.5, 2, 3.5], "lll"),
           *table("Processing time on the laptop CPU (300 frames of a UBFC video)",
                  ["Frame size", "Landmarks", "Patches", "Fusion and the rest", "Total", "Max. frame rate"],
                  [["640 x 480", "9.0 ms", "0.7 ms", "0.7 ms", "10.4 ms", "96 fps"],
                   ["320 x 240", "8.5 ms", "0.5 ms", "0.4 ms", "9.4 ms", "107 fps"]]),
           p("In the live demo the full analysis of the 32 s buffer takes 0.2–0.4 s once a second in a "
             "background thread; POS was vectorised for this (identical output, 14 times faster)."),
           h2("2. Interpretation of the obtained results"),
           *numbered([
               ("Beat timing, not heart rate, decides AF false alarms.",
                "All pulse extractors give heart rates within 0.6 BPM on these videos, yet GREEN raises a false "
                "AF alarm on 11 of 12 healthy people, CHROM on 2 and POS on 1, in the same order as their "
                "beat-interval errors (55, 43 and 21 ms). Patches remove the last false alarm."),
               ("The trust checks matter when part of the face is bad.",
                "On the clean UBFC videos the weights stay nearly equal and the checks change nothing. In the "
                "stress tests they cut the beat-interval error from 41–49 ms (single region) to 4.6–6.3 ms, "
                "each check acting on its own problem, and motion removal cuts 64–241 ms to 7–15 ms."),
               ("Rhythm-independent quality keeps AF.",
                "A spectral-sharpness gate would discard 94–100% of AF windows; the beat-shape gate keeps 96% "
                "and raises the AUC from 0.96 to 0.99."),
               ("The classifier transfers.",
                "Trained on contact PPG, it detects 90–97% of AF windows on MIT-BIH ECG beat timing and "
                "flags only 0.2–0.3% of clean normal windows; training with the measured camera timing noise "
                "costs nothing on MIMIC and lowers false alarms on faces."),
               ("Extra beats are the main remaining cause of false alarms,",
                "raising the AF call rate from 0.3% to 13–22% on MIT-BIH. Removing isolated extra beats "
                "nearly halves this at a cost of 8 points of AF sensitivity on DS2; removal without the "
                "neighbour check would have cost 19 points."),
               ("Beat-based heart rate is needed in AF,",
                "where it halves large heart-rate errors; breathing-aware selection is a coin flip because the "
                "breathing estimate itself is unreliable."),
               ("The camera overestimates RMSSD about twofold,",
                "because about 20 ms of timing noise per beat, shared by all patches, adds beat-to-beat variation "
                "that is not there. Heart rate, AF and the flags are robust to it; RMSSD is not."),
               ("Face size and patch count did not matter",
                "on these clean, lightly compressed videos, down to about 14 pixels per patch."),
           ])]
    return out


def analysis(media: Media) -> list[str]:
    out = [h1("Analysis"), h2("1. Analysis of the results obtained"),
           p("The results support the central idea of the project: for rhythm screening the quality of beat "
             "timing matters more than the accuracy of heart rate, and quality must be judged without assuming a "
             "regular rhythm. Beat timing separated the pulse extractors where heart rate could not, the "
             "beat-shape gate kept AF where a spectral gate would have removed it, and training the classifier "
             "with the camera's measured timing noise and removing isolated extra beats each lowered false alarms."),
           p("The results also show what could not be shown. The trust checks and motion removal help strongly "
             "in simulation, but the available real videos have no shadows, shine, turned faces or real head "
             "motion, so their real-world benefit is not demonstrated; without PURE, motion removal never "
             "activated on a real video. AF detection was validated on contact pulses (MIMIC) and ECG timing "
             "(MIT-BIH) with camera-like noise, not on face videos of AF patients, because OBF was not available. "
             "Breathing rate from the pulse failed on intensive-care data, and camera RMSSD is biased upwards."),
           p("Three findings changed the design during the project: the camera pulse is inverted and delayed "
             "relative to a finger sensor (evaluation had to align them); camera timing noise alone makes healthy "
             "faces look like AF (the classifier is now trained with it); and removing every extra beat also "
             "removes AF (only isolated ones are removed now)."),
           h2("2. Comparison with existing results")]
    out += table("Published results and the results of this project", ["Work", "Task and data", "Reported result"], [
        ["POS / CHROM with rPPG-Toolbox (Paruchuri et al., 2024)", "Heart rate, UBFC-rPPG (42 videos)",
         "4.00 / 3.98 BPM MAE"],
        ["This project: POS, one region / full method", "Heart rate, 7 UBFC videos",
         "0.22 / 0.23 BPM MAE (an easier subset)"],
        ["Dash et al. (2009)", "AF from ECG intervals, MIT-BIH AF", "94.4% sensitivity, 95.1% specificity"],
        ["McManus et al. (2013)", "AF from fingertip camera pulse, 76 patients", "96.2% sensitivity, 97.5% specificity"],
        ["Yan et al. (2018)", "AF from face video vs 12-lead ECG", "95% sensitivity, 96% specificity"],
        ["Sun et al. (2022)", "AF from face video (OBF)", "96.0% accuracy (AF vs healthy)"],
        ["This project: AF classifier", "MIMIC PERform AF, camera-like PPG, per patient",
         "100% sensitivity (19/19), 88% specificity (14/16); window AUC 0.96 (0.99 gated)"],
        ["This project: AF screening on faces", "12 normal-rhythm UBFC videos",
         "0 / 12 false alarms (GREEN 11, CHROM 2, POS 1)"],
        ["de Chazal et al. (2004)", "Beat classification from ECG shape and timing, MIT-BIH DS2",
         "Sensitivity 75.9% (supraventricular), 77.7% (ventricular)"],
        ["This project: extra-beat rule", "Beat timing only, MIT-BIH DS2, 16 ms noise",
         "Sensitivity 15% (atrial), 81% (ventricular); precision 86%"],
    ], [3, 3, 3.6], "lll")
    out += [p("The comparison is not like-for-like: the published AF results come from face or fingertip videos of "
              "real AF patients, while ours come from contact pulses with simulated camera noise, and our UBFC "
              "subset is smaller and easier than the full dataset. The extra-beat rule uses timing only, which a "
              "camera can provide, whereas de Chazal et al. also use the ECG waveform shape; this explains its weak "
              "result for atrial premature beats, which are often only slightly early.")]
    out += [h2("3. Efficiency obtained and metrics used")]
    out += table("Metrics used in the project", ["Metric", "Meaning", "Better"], [
        ["MAE, window MAE (BPM)", "Heart-rate error against the reference, whole video and per 8 s window", "Lower"],
        ["Beat-interval error (ms), beat F1", "Error of matched inter-beat intervals; beats found without missing or "
                                              "inventing any", "Lower; higher"],
        ["RMSSD, SDNN error (ms)", "Heart-rate-variability error against the reference", "Lower"],
        ["AUC, sensitivity, specificity", "AF detection per window and per patient", "Higher"],
        ["False-alarm rate, coverage", "AF calls on normal rhythm; share of windows with an answer", "Lower; higher"],
        ["Sensitivity, precision", "Extra-beat detection", "Higher"],
        ["Agreement, Cohen's kappa", "Slow/fast flag against the flag from the ECG", "Higher"],
        ["Breathing MAE, within 2 /min", "Breathing-rate error against impedance respiration", "Lower; higher"],
        ["Processing time (ms per frame)", "Computational efficiency", "Lower"],
    ], [2.5, 5, 1.3], "llc")
    out += [p("Efficiency obtained: AF AUC 0.96 (0.99 gated) with every AF patient detected; no AF false alarm on "
              "the healthy face videos; heart-rate error 0.2 BPM and beat-interval error 21 ms on face videos; "
              "extra-beat precision 86%; slow/fast agreement 95%; and about 10 ms per frame (96 fps) on a laptop "
              "CPU, three times faster than real time.")]
    out += [h2("4. Results represented in graphs with interpretation")]
    figs = [
        ("fig_stress.png", 16, "Beat-interval error of the baseline, equal-weight patches and the proposed method in "
                               "six simulated scenarios",
         "the single region (white) fails whenever part of the face is bad or the head moves; equal-weight patches "
         "(hatched) help except under motion, and the proposed method (dark) stays at 4–8 ms in every scenario, "
         "including AF."),
        ("fig_baselines.png", 16, "Beat-interval error and AF false alarms of the baselines and the proposed method "
                                  "on the UBFC face videos",
         "the ranking of false alarms (11, 2, 1, 0, 0) follows the ranking of beat-timing errors; heart-rate error, "
         "which is 0.2–0.6 BPM for all, would hide these differences."),
        ("fig_gating.png", 15, "Share of AF and non-AF windows kept by the beat-shape gate and by a spectral "
                               "sharpness gate (MIMIC PERform AF)",
         "the beat-shape gate (solid) keeps AF and normal windows alike, while the spectral gate (dashed) removes "
         "nearly all AF windows already at 0.5, which would make AF undetectable."),
        ("fig_removal.png", 15, "Effect of extra-beat removal on the AF classifier on MIT-BIH DS2",
         "plain removal lowers false alarms on extra beats but also erases AF; removing only isolated extra beats "
         "keeps most AF windows while nearly halving the false alarms."),
        ("fig_rmssd.png", 8.5, "RMSSD from the camera against RMSSD from the finger sensor (7 UBFC videos)",
         "every point lies above the identity line: camera timing noise roughly doubles RMSSD, so camera RMSSD is "
         "shown with a warning, and the AF classifier is trained to tolerate the same noise."),
        ("fig_demo.png", 11, "Live demo on a UBFC video played as if live (face blurred for privacy)",
         "the trust map shows the forehead and left cheek trusted (green) and the turned right cheek less trusted "
         "(red); the panel shows heart rate, confidence, rhythm, extra beats and RMSSD."),
    ]
    for name, w, cap, interp in figs:
        out += [media.image(FIG / name, w, cap), figure_caption(cap), p(f"Figure {Counter.figure}: {interp}")]
    return out


def conclusion(old_significance: list[str]) -> list[str]:
    return [h1("Conclusion and Future Work"), h2("1. Significance of the project"), *old_significance,
            h2("2. Results and efficiency obtained"),
            p("The complete system was built: face tracking, 25 mesh-based patches, rhythm-independent trust, "
              "fusion, motion removal, confidence, beat detection, an AF classifier, extra-beat and slow/fast flags, "
              "and a live webcam demo running at about 10 ms per frame on a CPU. Trained on MIMIC PERform AF with "
              "camera-like timing noise, the classifier found all 19 AF patients with a window AUC of 0.96 (0.99 "
              "with confidence gating) and transferred to MIT-BIH ECG timing. On the healthy UBFC face videos it "
              "raised no false alarm with the proposed method, against 11 with GREEN and 2 with CHROM. The "
              "project's main lesson is that beat timing, not heart rate, determines whether a camera can screen "
              "rhythm, and that quality measures must not assume a regular rhythm."),
            h2("3. Difficulties faced in performing this project"),
            *numbered([
                ("Data access:", "OBF (AF face videos), PURE (head motion) and COHFACE (breathing) need access "
                                 "requests that could not be completed in time, and the UBFC-rPPG download failed "
                                 "part-way, leaving 12 videos, 7 with a reference. Two videos had lost their folder "
                                 "names and were identified by heart rate and duration."),
                ("Camera versus contact pulse:", "the camera pulse is inverted and delayed relative to the finger "
                                                 "sensor, which at first made beat matching look poor (F1 0.72) "
                                                 "until the reference was aligned."),
                ("Camera timing noise looks like AF:", "the first classifier flagged healthy faces; training with "
                                                       "the measured timing noise fixed it without losing AF."),
                ("Extra-beat removal also removed AF:", "solved by removing only isolated extra beats, chosen on a "
                                                        "separate data split."),
                ("Breathing from the pulse:", "the estimator was correct on synthetic signals but unreliable on "
                                             "intensive-care patients, so it was dropped as an output."),
                ("Speed of the live demo:", "the analysis grew with the buffer and exceeded 1 s; POS was vectorised "
                                            "and the buffer limited to 32 s."),
                ("Practical issues:", "limited disk space (videos were processed straight from the zip files), "
                                      "Windows setup, and phone videos in HEVC that OpenCV cannot read."),
            ]),
            h2("4. Future work"),
            *numbered([
                "Test AF screening on face videos of AF patients (OBF), and on talking and moving people (PURE, "
                "UBFC-Phys), to measure real sensitivity and the real benefit of trust and motion removal.",
                "Reduce beat-timing noise, for example with a higher frame rate or a better beat fiducial, to make "
                "camera RMSSD usable and lower AF false alarms further.",
                "Use beat shape as well as timing to recognise atrial premature beats, which timing alone misses.",
                "Validate breathing rate on resting, healthy people (COHFACE) before showing it.",
                "Compare with pretrained deep networks (PhysNet, PhysFormer) through rPPG-Toolbox, and test across "
                "skin tones, lighting and phone cameras.",
                "Run a prospective study with clinicians against a 12-lead ECG before any real-world use; the "
                "system is a screening aid, not a diagnosis.",
            ])]


NEW_PROMPTS = [
    "This is the repository of my project, built with an AI coding assistant in the cloud. Pull it into this folder and "
    "understand everything about the project.",
    "I am downloading UBFC-rPPG DATASET_2 and then MIMIC; PURE needs an access request that I have no time for. "
    "Check whether using only these, or other public datasets, will cause problems.",
    "Both datasets are now in the datasets folder as zip files; part of DATASET_2 failed to download. Is this enough "
    "to train the model, or is the complete dataset needed?",
    "I cannot download the rest of the dataset. Do what is possible with the existing data.",
    "I have added the MIT-BIH dataset to the same folder. Start Phase 4.",
    "Commit this and start Phase 2.",
    "Try again (after a failing test of the breathing-aware heart rate).",
    "Commit this and start Phase 5.",
    "Commit this and start Phase 6.",
    "Commit this and start Phase 7: the final project report in the course format (no slides for now).",
]
NEW_KEYWORDS = [
    "MIMIC PERform datasets; impedance pneumography; PPG-derived respiratory rate Smart Fusion",
    "inter-patient heartbeat classification; de Chazal DS1/DS2 split; premature beat detection from RR intervals",
    "sample entropy; random forest; scikit-learn",
    "pulse transit time between face and finger; rPPG polarity",
]


def build() -> None:
    from PIL import Image  # noqa: F401  (needed by Media)

    work = Path(tempfile.mkdtemp(prefix="report_"))
    with zipfile.ZipFile(TEMPLATE) as z:
        z.extractall(work)
    doc = work / "word/document.xml"
    xml = doc.read_text(encoding="utf-8")
    pre, els, post = split_body(xml)
    media = Media()

    # Title page
    title = [e.replace("Digital Assignment 3", "Final Project Report") for e in els[:find(els, "Abstract")]]

    # Introduction (kept; last row of Table 1 updated)
    i_intro, i_lit = find(els, "Introduction"), find(els, "Literature Survey / Related Works")
    intro = substitute(els[i_intro:i_lit], {"real-dataset validation still pending":
                                            "no AF face videos were available for testing"})

    # Literature survey (kept; "planned" statements updated)
    i_arch = find(els, "Overall Architecture")
    lit = els[i_lit:i_arch]
    subs = {
        "These features form the core of the AF feature set planned in this project.":
            "These features form the core of the AF feature set used in this project.",
        "The planned breathing-rate block of this project and its breathing-aware heart-rate selection follow this "
        "approach.": "The breathing-rate block of this project follows this approach; on intensive-care data it "
                     "proved too unreliable to be shown as an output.",
        "It provides the baselines against which this project is compared.":
            "Its GREEN and CHROM implementations are the reference for the baselines re-implemented in this project.",
    }
    lit = substitute(lit, subs)
    Counter.table = 2

    # Architecture (kept; block 8, output and status updated)
    i_meth = find(els, "Proposed Methodology")
    arch = els[i_arch:i_meth]
    arch = substitute(arch, {
        "Breathing rate from the pulse's respiratory modulation is planned for the next phase.":
            "Breathing rate from the pulse's respiratory modulation was built and evaluated but is not shown, "
            "because it proved unreliable.",
        # "Output:" is a separate bold run; only the text after it changes
        " heart rate, breathing rate, RMSSD and SDNN, the rhythm result, the heart-rate flags and the confidence;":
            " heart rate, RMSSD, the rhythm result with the number of extra beats, the heart-rate flags and the "
            "confidence;",
    })
    k = next(n for n, e in enumerate(arch) if text_of(e).startswith("Blocks 1"))
    arch[k] = p("All blocks are implemented and evaluated (Results). Breathing rate is computed but not shown, and "
                "extra beats found by block 11 are removed before the AF check only when they are isolated.")
    Counter.figure = 1

    # Methodology: keep 1-3.7, replace 3.8-3.10, regenerate Table 3, keep the justification
    i_38 = find(els, "3.8 AF screening features (next phase)")
    i_tab3 = find(els, "4. Advantages and disadvantages of the methods used")
    i_just = find(els, "5. Justification for choosing these methods")
    i_impl = find(els, "Implementation Details")
    meth = substitute(els[i_meth:i_38], {
        "compute heart rate and beat intervals, and, in the next phases, run AF screening and the heart-rate flags "
        "on confident windows.": "compute heart rate and beat intervals, and run AF screening (after removing "
                                 "isolated extra beats) and the heart-rate flags on confident windows."})
    Counter.figure = 3
    fig4 = [e for e in els[i_38:i_tab3] if "<w:drawing>" in e]
    fig4_cap = [e for e in els[i_38:i_tab3] if text_of(e).startswith("Figure 4:")]
    old_tab = next(e for e in els[i_tab3:i_just] if e.startswith("<w:tbl>"))
    old_rows = [[text_of(c).replace("&amp;", "&") for c in re.findall(r"<w:tc>.*?</w:tc>", r, re.S)]
                for r in re.findall(r"<w:tr>.*?</w:tr>", old_tab, re.S)][1:]
    meth += methodology_new(media, fig4 + fig4_cap)
    Counter.figure = 4
    meth += [els[i_tab3]] + methods_table(old_rows) + els[i_just:i_impl]

    impl = implementation()
    res = results()
    ana = analysis(media)

    # Conclusion: keep the significance paragraph
    i_concl = find(els, "Conclusion and Future Work")
    i_sig = find(els, "1. Significance of the project", i_concl)
    i_res2 = find(els, "2. Results and efficiency obtained", i_concl)
    concl = conclusion(els[i_sig + 1:i_res2])

    # Prompt engineering: kept, extended
    i_prompt = find(els, "Prompt Engineering")
    i_refs = find(els, "References")
    prompt = els[i_prompt:i_refs]
    k = next(n for n, e in enumerate(prompt) if text_of(e).startswith("The prompts were written by the team"))
    prompt[k] = p("The prompts were written by the team and given to an AI coding assistant (an agent that can "
                  "read the repository, run code and search the web): first in "
                  "the cloud (prompts 1–35) and then in the desktop app on the team's laptop with the datasets "
                  "(prompts 36–45). It was used to plan the phases, explain concepts, research the literature, "
                  "draw the architecture diagrams, write and test the code, run the experiments and draft the "
                  "reports. The team made the project decisions (scope, outputs, which datasets to use, what to keep "
                  "or drop), and every result was produced by running the project code.")
    k35 = next(n for n, e in enumerate(prompt) if text_of(e).startswith("35."))
    prompt = prompt[:k35 + 1] + [num(36 + n, t) for n, t in enumerate(NEW_PROMPTS)] + prompt[k35 + 1:]
    k_kw = max(n for n, e in enumerate(prompt) if 'w:numId w:val="5"' in e and "ListParagraph" in e
               and n < next(m for m, e2 in enumerate(prompt) if text_of(e2) == "4. Any other"))
    kw_tpl = prompt[k_kw]
    kw_new = [re.sub(r"<w:t xml:space=\"preserve\">[^<]*</w:t>", f'<w:t xml:space="preserve">{escape(t)}</w:t>',
                     kw_tpl, count=1) for t in NEW_KEYWORDS]
    prompt = prompt[:k_kw + 1] + kw_new + prompt[k_kw + 1:]
    prompt = [replace_in(e, "the team should check the references and the code before final submission.",
                         "the team checked the code by running the tests and experiments, and should check the "
                         "references before final submission.") if "before final submission" in e else e
              for e in prompt]

    # References: kept, with new entries in alphabetical order
    refs = els[i_refs:-1]
    rels_new = {}

    def ref_para(parts: list[tuple[str, bool]], url: str) -> str:
        rid = f"rIdRefNew{len(rels_new) + 1}"
        rels_new[rid] = url
        runs = "".join(run(t, i=it) for t, it in parts)
        return ('<w:p><w:pPr><w:spacing w:after="140" w:line="276"/><w:ind w:left="567" w:hanging="567"/>'
                f'<w:jc w:val="left"/></w:pPr>{runs}{run(" ")}<w:hyperlink w:history="1" r:id="{rid}">'
                f"{run(url)}</w:hyperlink></w:p>")

    new_refs = {
        "Bobbia": ref_para([("Breiman, L. (2001). Random forests. ", False), ("Machine Learning, 45", True),
                            ("(1), 5–32.", False)], "https://doi.org/10.1023/A:1010933404324"),
        "Dash": ref_para([("de Chazal, P., O'Dwyer, M., & Reilly, R. B. (2004). Automatic classification of "
                           "heartbeats using ECG morphology and heartbeat interval features. ", False),
                          ("IEEE Transactions on Biomedical Engineering, 51", True), ("(7), 1196–1206.", False)],
                         "https://doi.org/10.1109/TBME.2004.827359"),
        "Orphanidou": ref_para([("Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., "
                                 "et al. (2011). Scikit-learn: Machine learning in Python. ", False),
                                ("Journal of Machine Learning Research, 12", True), (", 2825–2830.", False)],
                               "https://jmlr.org/papers/v12/pedregosa11a.html"),
        "Poh": ref_para([("Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using "
                          "approximate entropy and sample entropy. ", False),
                         ("American Journal of Physiology – Heart and Circulatory Physiology, 278", True),
                         ("(6), H2039–H2049.", False)], "https://doi.org/10.1152/ajpheart.2000.278.6.H2039"),
    }
    out_refs = []
    for e in refs:
        out_refs.append(e)
        for key, new in new_refs.items():
            if text_of(e).startswith(key):
                out_refs.append(new)
    mimic_web = ref_para([("Charlton, P. H. (2022). MIMIC PERform datasets [Data set]. Zenodo.", False)],
                         "https://zenodo.org/record/6950488")
    k_web = next(n for n, e in enumerate(out_refs) if text_of(e).startswith("Charlton, P. H. (n.d.)"))
    out_refs.insert(k_web + 1, mimic_web)

    body = (title + abstract() + intro + lit + arch + meth + impl + res + ana + concl + prompt + out_refs
            + [els[-1]])
    doc.write_text(pre + "".join(body) + post, encoding="utf-8")

    # relationships and media
    rels = work / "word/_rels/document.xml.rels"
    r = rels.read_text(encoding="utf-8")
    add = "".join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
                  f'relationships/hyperlink" Target="{escape(u)}" TargetMode="External"/>' for rid, u in rels_new.items())
    for rid, path in media.items:
        target = f"media/final_{path.name}"
        shutil.copy(path, work / "word" / target)
        add += (f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
                f'image" Target="{target}"/>')
    rels.write_text(r.replace("</Relationships>", add + "</Relationships>"), encoding="utf-8")

    # document title in the properties
    core = work / "docProps/core.xml"
    if core.exists():
        c = core.read_text(encoding="utf-8")
        core.write_text(c.replace("Digital Assignment 3", "Final Project Report"), encoding="utf-8")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(work.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(work).as_posix())
    shutil.rmtree(work, ignore_errors=True)
    print(f"written {OUT}: {Counter.table} tables, {Counter.figure} figures")


if __name__ == "__main__":
    build()
