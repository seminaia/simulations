#!/usr/bin/env python3
"""
simulations_ppt.py
==================
Generates the "Introduction to Atomistic and Molecular Simulations" Beamer
slide deck using DocumentBuilder and matplotlib (for graphs).

Usage
-----
    conda activate myGPAWenv
    python simulations_ppt.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# ── make sure PyFunc is importable ──────────────────────────────────────────
sys.path.insert(0, str(object=Path(__file__).resolve().parent / "PyFunc"))
from doc_builder import DocumentBuilder


# ============================================================================
# 1.  Generate the matplotlib figure  (Time Scales vs Length Scales)
# ============================================================================
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

SCATTER_PATH = FIG_DIR / "timescale_vs_lengthscale.pdf"


def make_loglog_figure(outpath: Path) -> None:
    """Produce the Time-Scales vs Length-Scales log-log scatter plot."""
    lengths = [1e-10, 1e-9, 1e-7, 1e-6, 1e-2]
    times   = [1e-12, 1e-9, 1e-6, 1e-3, 1e0 ]
    labels  = [
        "Quantum/DFT",
        "Classical/MD",
        "Nanostructures",
        "Microstructures",
        "Macrostructures",
    ]
    offsets_x = [0.3, 0.3, 0.3, 0.3, 0.3]   # log-scale text offsets (multipliers)
    offsets_y = [5.0, 5.0, 5.0, 5.0, 5.0]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-12, 1e0)
    ax.set_ylim(1e-12, 1e2)
    ax.set_xlabel("Length Scale (m)", fontsize=12)
    ax.set_ylabel("Time Scale (s)", fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.5)

    ax.scatter(lengths, times, s=60, color="black", zorder=5)

    for x, y, lbl, ox, oy in zip(lengths, times, labels, offsets_x, offsets_y):
        ax.annotate(
            text=lbl,
            xy=(x, y),
            xytext=(x * ox, y * oy),
            fontsize=10,
            ha="center",
        )

    fig.tight_layout()
    fig.savefig(str(object=outpath), dpi=300)
    plt.close(fig)
    print(f"[OK] Saved figure → {outpath}")


# ============================================================================
# 2.  Build the slide deck
# ============================================================================
def build_deck() -> None:
    """Construct and compile the Beamer presentation."""

    # -- generate the matplotlib figure first --------------------------------
    make_loglog_figure(SCATTER_PATH)

    # -- builder -------------------------------------------------------------
    db = DocumentBuilder(
        base_name="simulations_ppt",
        title="Introduction to Atomistic and Molecular Simulations",
        author="Soknarith Sem",
        institute="Worcester Polytechnic Institute",
        document_options="aspectratio=169,10pt",
    )

    # -- theme ---------------------------------------------------------------
    db.beamer_theme(
        theme="metropolis",
        theme_options="numbering=none,progressbar=frametitle",
    )

    # -- raw preamble (custom packages, colours, backgrounds) -----------------
    db.raw_preamble(r"\metroset{block=fill, titleformat frame=smallcaps}")
    db.raw_preamble(r"\setbeamertemplate{footline}{}")
    db.raw_preamble(r"\setbeamertemplate{navigation symbols}{}")

    # extra packages
    db.raw_preamble(r"\usepackage[english]{babel}")
    db.raw_preamble(r"\usepackage{lmodern}")
    db.raw_preamble(r"\usepackage{microtype}")
    db.raw_preamble(r"\usepackage{csquotes}")
    db.raw_preamble(r"\usepackage{siunitx}")
    db.raw_preamble(r"\sisetup{detect-all}")
    db.raw_preamble(r"\usepackage{appendixnumberbeamer}")
    db.raw_preamble(r"\usepackage{chemformula}")
    db.raw_preamble(r"\usepackage{subcaption}")
    db.raw_preamble(r"\usepackage{adjustbox}")

    # WPI colours
    db.raw_preamble(r"\definecolor{WPIred}{HTML}{A41034}")
    db.raw_preamble(r"\definecolor{WPIgray}{HTML}{6B6F72}")
    db.raw_preamble(r"\definecolor{WPIlight}{HTML}{F2F2F2}")
    db.raw_preamble(r"\definecolor{Accent}{HTML}{E67E22}")

    db.raw_preamble(r"\setbeamercolor{frametitle}{bg=WPIred, fg=white}")
    db.raw_preamble(r"\setbeamercolor{progress bar}{fg=WPIred!80!black, bg=WPIgray!30}")
    db.raw_preamble(r"\setbeamercolor{title separator}{fg=WPIred}")
    db.raw_preamble(r"\setbeamercolor{alerted text}{fg=Accent}")
    db.raw_preamble(r"\setbeamercolor{block title}{bg=WPIred!15, fg=WPIred!80!black}")
    db.raw_preamble(r"\setbeamercolor{block body}{bg=WPIlight}")

    # backgrounds
    db.raw_preamble(r"\newcommand{\TitleBG}{~/Templates/background1.pdf}")
    db.raw_preamble(r"\newcommand{\BodyBG}{~/Templates/background2.pdf}")
    db.raw_preamble(r"\newcommand{\EndBG}{~/Templates/background3.pdf}")
    db.raw_preamble(
        r"\setbeamertemplate{background canvas}{"
        r"\includegraphics[width=\paperwidth,height=\paperheight,keepaspectratio]{\BodyBG}}"
    )

    # bibliography (empty .bib for now)
    db.raw_preamble(r"\usepackage[backend=biber,style=ieee,sorting=none]{biblatex}")
    db.raw_preamble(r"\addbibresource{}")

    # structureimg helper
    db.raw_preamble(
        r"\newcommand{\structureimg}[2][]{"
        r"\adjincludegraphics[#1,width=0.45\linewidth,"
        r"height=0.32\textheight,keepaspectratio]{#2}}"
    )

    # -- custom title slide (scoped background) --------------------------
    db.maketitle(False)  # disable auto \titlepage
    db.raw_body(r"{")
    db.raw_body(r"\setbeamertemplate{background canvas}{")
    db.raw_body(
        r"  \includegraphics[width=\paperwidth,height=\paperheight,"
        r"keepaspectratio]{\TitleBG}"
    )
    db.raw_body(r"}")
    db.raw_body(r"\begin{frame}[plain]")
    db.raw_body(r"  \vspace{0.20\textheight}")
    db.raw_body(r"  \titlepage")
    db.raw_body(r"\end{frame}")
    db.raw_body(r"}")

    # =====================================================================
    # SECTION : Background
    # =====================================================================
    db.section("Background")

    # -- Time Scales vs Length Scales (matplotlib figure) ------------------
    db.slide("Time Scales vs Length Scales")
    db.figure(str(SCATTER_PATH), width=r"0.85\textwidth")
    db.end_slide()

    # -- Quantum vs Classical Simulations ---------------------------------
    db.slide("Quantum vs Classical Simulations")
    db.table(
        headers=["Schrödinger's Equation", "Newton's Equation"],
        rows=[],
        alignment="c|c",
    )
    db.end_slide()

    # -- Jacob's Ladder (placeholder) -------------------------------------
    db.slide("Jacob's Ladder")
    db.end_slide()

    # =====================================================================
    # SECTION : Conclusion
    # =====================================================================
    db.section("Conclusion")

    db.slide("Summary")
    db.end_slide()

    # =====================================================================
    # Acknowledgments (no section header — just a frame)
    # =====================================================================
    db.slide("Acknowledgments")
    db.p("WPI Turing Computing Cluster, University of Texas, TACC, and Prof. Yu Zhong")
    db.end_slide()

    # =====================================================================
    # References (placeholder)
    # =====================================================================
    db.slide("References")
    db.end_slide()

    # =====================================================================
    # Final "Thank You" slide with scoped EndBG background
    # =====================================================================
    # Requires LaTeX group scoping {…} around the background override +
    # frame.  raw_tail() emits after all sections/frames, so this ends
    # up right before \end{document} — exactly where we need it.
    db.raw_tail(r"{")
    db.raw_tail(r"\setbeamertemplate{background canvas}{")
    db.raw_tail(
        r"  \includegraphics[width=\paperwidth,"
        r"height=\paperheight,keepaspectratio]{\EndBG}"
    )
    db.raw_tail(r"}")
    db.raw_tail(r"\begin{frame}[plain]")
    db.raw_tail(r"  \centering")
    db.raw_tail(r"  {\LARGE\textbf{Thank You!}}\\[2em]")
    db.raw_tail(r"  \large Questions?\\[1.5em]")
    db.raw_tail(r"  \small")
    db.raw_tail(r"  \href{mailto:ssem@wpi.edu}{ssem@wpi.edu}")
    db.raw_tail(r"\end{frame}")
    db.raw_tail(r"}")

    # =====================================================================
    # Save
    # =====================================================================
    tex_path = db.save_beamer_tex()
    print(f"[OK] .tex written → {tex_path}")

    # Uncomment to also compile PDF (requires pdflatex + Metropolis theme):
    # pdf_path = db.save_beamer_pdf(runs=2, clean=True)
    # print(f"[OK] .pdf written → {pdf_path}")


if __name__ == "__main__":
    build_deck()
