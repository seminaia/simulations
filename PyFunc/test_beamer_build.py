"""Build test for Beamer slide-deck support in DocumentBuilder.

Generates a small presentation exercising every content type
(paragraphs, bullets, equations, tables, figures, raw tex, mixed paragraphs)
and writes .tex + .pdf output.
"""

from pathlib import Path
import sys, os

# ensure the parent directory is on the path so we can import doc_builder
sys.path.insert(0, os.path.dirname(__file__))

from doc_builder import DocumentBuilder

OUT_DIR = Path(__file__).parent / "test_beamer_output"
OUT_DIR.mkdir(exist_ok=True)
base = str(OUT_DIR / "test_slides")
db = DocumentBuilder(
    base_name=base,
    title="DocumentBuilder Beamer Test",
    author="Test Runner",
    subtitle="Verifying slide-deck generation",
)
db.beamer_theme("Madrid", color_theme="whale")
db.toc(True)

# -- Section 1: text content ---------------------------------------------
db.section("Introduction")

db.slide("Welcome Slide")
db.p("This is a paragraph inside a Beamer frame.")
db.line("This is a single line.")
db.end_slide()

db.slide("Bullet Points")
db.bullets([
    "First bullet",
    "Second bullet with some detail",
    "Third bullet",
])
db.end_slide()

db.slide("Numbered List")
db.numbered([
    "Step one",
    "Step two",
    "Step three",
])
db.end_slide()

# -- Section 2: math content ---------------------------------------------
db.section("Mathematics")

db.slide("Display Equation")
db.p("Here is the Schrödinger equation:")
db.eq(r"i\hbar \frac{\partial}{\partial t}\Psi = \hat{H}\Psi")
db.end_slide()

db.slide("Aligned Equations")
db.align(
    r"a &= b + c",
    r"  &= d + e + f",
)
db.end_slide()

db.slide("Mixed Paragraph with Inline Math")
db.px(
    "The energy is ",
    db.im(r"E = mc^{2}"),
    " where ",
    db.im(r"m"),
    " is mass.",
)
db.end_slide()

# -- Section 3: tables ---------------------------------------------------
db.section("Data")

db.slide("Sample Table")
db.table(
    headers=["Element", "Z", "Mass (amu)"],
    rows=[
        ["H", 1, 1.008],
        ["He", 2, 4.0026],
        ["Li", 3, 6.941],
    ],
    caption="Periodic table excerpt",
    float_fmt=".4f",
)
db.end_slide()

# -- Section 4: raw LaTeX ------------------------------------------------
db.section("Extras")

db.slide("Raw LaTeX")
db.raw_tex(r"\centering \textbf{Bold centered text via raw\_tex}")
db.end_slide()

db.slide("Vertical Space")
db.p("Before space")
db.vspace("1cm")
db.p("After space")
db.end_slide()

# -- Section 5: auto-wrap (no explicit slide) -----------------------------
db.section("Auto-Wrapped Content")
db.p("This paragraph was NOT placed inside a slide() call.")
db.p("It should be auto-wrapped in a frame titled after the section.")

# -- Section 6: fragile frame --------------------------------------------
db.section("Fragile Frame")
db.slide("Verbatim-Friendly", fragile=True)
db.raw_tex(r"\texttt{some\_monospace\_code()}")
db.end_slide()

# -- Section 7: subtitle on a slide --------------------------------------
db.section("Subtitle Demo")
db.slide("Main Title", subtitle="A clarifying subtitle")
db.p("This slide has both a title and a subtitle.")
db.end_slide()

# ── save ────────────────────────────────────────────────────────────────
print("=" * 60)
print("BEAMER BUILD TEST")
print("=" * 60)

# 1) .tex only
tex_path = db.save_beamer_tex()
assert Path(tex_path).exists(), f"FAIL: {tex_path} not created"
print(f"[PASS] .tex generated: {tex_path}")

# 2) quick sanity: check the .tex contains Beamer markers
tex_src = Path(tex_path).read_text()
checks = {
    r"\documentclass{beamer}":       "document class",
    r"\usetheme{Madrid}":            "theme",
    r"\usecolortheme{whale}":        "color theme",
    r"\begin{frame}":                "frame environment",
    r"\end{frame}":                  "end frame",
    r"\titlepage":                    "title page",
    r"\tableofcontents":             "table of contents",
    r"\section{":                     "section command",
    r"\framesubtitle{":              "frame subtitle",
    r"[fragile]":                     "fragile option",
    r"\begin{align*}":               "align environment",
    r"\begin{itemize}":              "itemize (bullets)",
    r"\begin{enumerate}":            "enumerate (numbered)",
}
all_ok = True
for marker, label in checks.items():
    if marker in tex_src:
        print(f"  [PASS] {label:25s} → found")
    else:
        print(f"  [FAIL] {label:25s} → MISSING")
        all_ok = False

# 3) try PDF compilation
pdf_path = db.save_beamer_pdf(runs=2)
if pdf_path and Path(pdf_path).exists():
    size_kb = Path(pdf_path).stat().st_size / 1024
    print(f"[PASS] .pdf generated: {pdf_path} ({size_kb:.0f} KB)")
else:
    print("[SKIP] PDF not generated (pdflatex may not be installed)")

# 4) also test save_all(beamer=True) round-trip
db2 = DocumentBuilder(base_name=str(OUT_DIR / "test_save_all"), title="save_all test")
db2.section("Hello")
db2.slide("Slide 1")
db2.p("Content")
db2.end_slide()
txt2, tex2, pdf2 = db2.save_all(beamer=True)
assert Path(txt2).exists(), f"FAIL: {txt2}"
assert Path(tex2).exists(), f"FAIL: {tex2}"
print(f"[PASS] save_all(beamer=True) → .txt={txt2}, .tex={tex2}, .pdf={pdf2}")

print("=" * 60)
if all_ok:
    print("ALL TEX CHECKS PASSED")
else:
    print("SOME TEX CHECKS FAILED — review output above")
print("=" * 60)
