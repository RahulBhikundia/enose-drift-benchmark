"""Builds reports/final_report.pdf, the CSL7620 submission report (2026-11-26).
Run: PYTHONPATH=<dir with reportlab> .venv/Scripts/python.exe scripts/report/make_final_report.py
Every number is taken from results/ (see the comment next to each block)."""
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures"
OUT = ROOT / "reports"; OUT.mkdir(exist_ok=True)
PDF = OUT / "final_report.pdf"

fd = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(fd / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(fd / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVI", str(fd / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")

INK = colors.HexColor("#1f1f1e"); MUTED = colors.HexColor("#6b6a64"); BLUE = colors.HexColor("#2a78d6")
RULE = colors.HexColor("#d8d7d0"); TINT = colors.HexColor("#f3f2ee"); GOOD = colors.HexColor("#e6f2ea")
WARN = colors.HexColor("#fbf1dc"); BAD = colors.HexColor("#f8e4e0")

S = {
    "title": ParagraphStyle("title", fontName="DVB", fontSize=20, leading=25, textColor=INK, spaceAfter=6),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=10.5, leading=14, textColor=MUTED, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=14, leading=18, textColor=INK, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=11, leading=14, textColor=INK, spaceBefore=10, spaceAfter=4),
    "p": ParagraphStyle("p", fontName="DV", fontSize=9.3, leading=13.2, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    "b": ParagraphStyle("b", fontName="DV", fontSize=9.3, leading=13, textColor=INK, leftIndent=12, bulletIndent=2, bulletFontName="DV",
                        spaceAfter=2.5),
    "cap": ParagraphStyle("cap", fontName="DV", fontSize=8, leading=10.5, textColor=MUTED, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8, leading=10.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DVB", fontSize=8, leading=10.2, textColor=INK),
    "box": ParagraphStyle("box", fontName="DV", fontSize=9.3, leading=13.2, textColor=INK),
}
story = []


def P(t, s="p"):
    story.append(Paragraph(t, S[s]))


def H1(t):
    story.append(CondPageBreak(5 * cm)); story.append(Paragraph(t, S["h1"]))


def H2(t):
    story.append(CondPageBreak(6.5 * cm)); story.append(Paragraph(t, S["h2"]))


def B(items):
    for t in items:
        story.append(Paragraph(t, S["b"], bulletText="•"))
    story.append(Spacer(1, 3))


def T(rows, widths, header=True, shade=None):
    data = [[Paragraph(str(c), S["cellb" if (header and i == 0) else "cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), TINT), ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED)]
    for (r, c) in (shade or []):
        st.append(("BACKGROUND", (0, r), (-1, r), c))
    t.setStyle(TableStyle(st))
    story.append(KeepTogether([t])); story.append(Spacer(1, 8))


def BOX(t, bg=TINT):
    tb = Table([[Paragraph(t, S["box"])]], colWidths=[17 * cm])
    tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LEFTPADDING", (0, 0), (-1, -1), 9),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 7),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("LINEBEFORE", (0, 0), (0, -1), 2.5, BLUE)]))
    story.append(tb); story.append(Spacer(1, 9))


def FIGURE(name, cap, width=17.0):
    p = FIG / name
    from PIL import Image as PI
    w, h = PI.open(p).size
    img = Image(str(p), width=width * cm, height=width * cm * h / w)
    story.append(KeepTogether([img, Paragraph(cap, S["cap"])]))


S["h1"].keepWithNext = 1; S["h2"].keepWithNext = 1
S["ref"] = ParagraphStyle("ref", fontName="DV", fontSize=8, leading=10.5, textColor=INK, leftIndent=14,
                          firstLineIndent=-14, spaceAfter=2.5)
S["auth"] = ParagraphStyle("auth", fontName="DV", fontSize=10, leading=14, textColor=INK, spaceAfter=10)


def footer(c, doc):
    c.saveState(); c.setFont("DV", 7.5); c.setFillColor(MUTED)
    c.drawString(2 * cm, 1.2 * cm, "CSL7620 Machine Learning — course project report")
    c.drawRightString(19 * cm, 1.2 * cm, f"{doc.page}")
    c.restoreState()


exec((ROOT / "scripts" / "report" / "final_report_body.py").read_text(encoding="utf-8"))   # report text (kept separate)

doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="Does Flatness Help Gas Sensors Survive Drift?",
                        author="CSL7620 project")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(PDF)

