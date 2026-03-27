from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import shutil
import subprocess

from pylatex import (
    Document, Section, Subsection, Figure, Tabular, LongTable,
    NoEscape, Command
)
from pylatex.utils import escape_latex


def _txt_safe(text: str) -> str:
    replacements = {
        "—": "--",
        "–": "-",
        "−": "-",
        "━": "-",
        "─": "-",
        "═": "=",
        "│": "|",
        "•": "*",
        "·": "*",
        "×": "x",
        "²": "^2",
        "³": "^3",
        "₀": "_0",
        "₁": "_1",
        "₂": "_2",
        "₃": "_3",
        "₄": "_4",
        "₅": "_5",
        "₆": "_6",
        "₇": "_7",
        "₈": "_8",
        "₉": "_9",
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ε": "epsilon",
        "θ": "theta",
        "λ": "lambda",
        "μ": "mu",
        "π": "pi",
        "Π": "Pi",
        "σ": "sigma",
        "τ": "tau",
        "φ": "phi",
        "ω": "omega",
        "√": "sqrt",
        "≤": "<=",
        "≥": ">=",
        "≠": "!=",
        "≈": "~=",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _fmt_cell(val: Any, float_fmt: str = ".4f") -> str:
    if isinstance(val, float):
        return format(val, float_fmt)
    return str(val)


class WorkLog:
    def __init__(
        self,
        base_name: str,
        title: str,
        author: str | None = None,
        date_tex: str = r"\today",
    ):
        self.base_name = base_name
        self.title = title
        self.author = author
        self.date_tex = date_tex

        self.txt_lines: list[str] = []

        self.doc = Document(base_name, geometry_options={"margin": "1in"})
        self.doc.packages.append(Command("usepackage", "amsmath"))
        self.doc.packages.append(Command("usepackage", "amssymb"))
        self.doc.packages.append(Command("usepackage", "booktabs"))
        self.doc.packages.append(Command("usepackage", "longtable"))
        self.doc.packages.append(Command("usepackage", "float"))

        self.doc.preamble.append(Command("title", title))
        if author:
            self.doc.preamble.append(Command("author", author))
        self.doc.preamble.append(Command("date", NoEscape(date_tex)))
        self.doc.append(NoEscape(r"\maketitle"))

        self._container_stack = [self.doc]

    @property
    def current(self):
        return self._container_stack[-1]

    # ---------- structure ----------

    def section(self, title: str):
        self.txt_lines.append("")
        self.txt_lines.append(title)
        self.txt_lines.append("-" * max(20, len(title)))

        sec = Section(title)
        self.current.append(sec)
        self._container_stack = [self.doc, sec]

    def subsection(self, title: str):
        self.txt_lines.append("")
        self.txt_lines.append(title)
        self.txt_lines.append("-" * max(8, len(title)))

        sub = Subsection(title)
        self.current.append(sub)
        self._container_stack = self._container_stack[:2] + [sub]

    # ---------- text ----------

    def text(self, text: str = ""):
        s = str(text)
        self.txt_lines.append(s)

        if s.strip():
            self.current.append(escape_latex(_txt_safe(s)))
        self.current.append(NoEscape(r"\par"))

    def raw_tex(self, tex: str):
        self.current.append(NoEscape(tex))

    def inline_math(self, latex: str) -> str:
        return f"${latex}$"

    # ---------- math ----------

    def math(self, latex: str):
        latex = latex.strip()
        self.txt_lines.append(f"[MATH] {latex}")
        self.current.append(NoEscape(r"\["))
        self.current.append(NoEscape(latex))
        self.current.append(NoEscape(r"\]"))

    def align(self, *lines: str):
        cleaned = [line.strip() for line in lines if str(line).strip()]
        self.txt_lines.append("[ALIGN]")
        self.txt_lines.extend(cleaned)

        self.current.append(NoEscape(r"\begin{align*}"))
        for i, line in enumerate(cleaned):
            end = r" \\" if i < len(cleaned) - 1 else ""
            self.current.append(NoEscape(line + end))
        self.current.append(NoEscape(r"\end{align*}"))

    # ---------- tables ----------

    def table(
        self,
        headers: list[str],
        rows: Iterable[Iterable[Any]],
        caption: str | None = None,
        label: str | None = None,
        float_fmt: str = ".4f",
    ):
        rows = [list(r) for r in rows]

        # txt version
        txt_headers = [_txt_safe(h) for h in headers]
        txt_rows = [[_fmt_cell(v, float_fmt) for v in row] for row in rows]

        widths = [len(h) for h in txt_headers]
        for row in txt_rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        def fmt_row(vals: list[str]) -> str:
            return " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals)))

        self.txt_lines.append(fmt_row(txt_headers))
        self.txt_lines.append("-+-".join("-" * w for w in widths))
        for row in txt_rows:
            self.txt_lines.append(fmt_row(row))
        if caption:
            self.txt_lines.append(f"Caption: {caption}")
        self.txt_lines.append("")

        # latex version
        self.current.append(NoEscape(r"\begin{table}[H]"))
        self.current.append(NoEscape(r"\centering"))

        tab = Tabular(" ".join(["c"] * len(headers)))
        tab.add_hline()
        tab.add_row(headers)
        tab.add_hline()
        for row in rows:
            formatted = [
                _fmt_cell(v, float_fmt) if isinstance(v, float) else str(v)
                for v in row
            ]
            tab.add_row(formatted)
        tab.add_hline()
        self.current.append(tab)

        if caption:
            self.current.append(NoEscape(rf"\caption{{{escape_latex(caption)}}}"))
        if label:
            self.current.append(NoEscape(rf"\label{{{label}}}"))

        self.current.append(NoEscape(r"\end{table}"))

    # ---------- figures ----------

    def figure(
        self,
        path: str,
        caption: str | None = None,
        label: str | None = None,
        width: str = r"0.95\textwidth",
        position: str = "H",
    ):
        self.txt_lines.append(f"[FIGURE] {path}")
        if caption:
            self.txt_lines.append(f"Caption: {caption}")
        self.txt_lines.append("")

        with self.current.create(Figure(position=position)) as fig:
            fig.add_image(path, width=NoEscape(width))
            if caption:
                fig.add_caption(caption)
            if label:
                fig.append(NoEscape(rf"\label{{{label}}}"))

    # ---------- output ----------

    def save_txt(self) -> str:
        path = f"{self.base_name}.txt"
        Path(path).write_text("\n".join(self.txt_lines) + "\n", encoding="utf-8")
        return path

    def save_tex(self) -> str:
        path = f"{self.base_name}.tex"
        self.doc.generate_tex(self.base_name)
        return path

    def save_pdf(self, clean_tex: bool = False) -> str | None:
        try:
            self.doc.generate_pdf(self.base_name, clean_tex=clean_tex,
                                  compiler="pdflatex",compiler_args=["-interaction=nonstopmode"])
            return f"{self.base_name}.pdf"
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return None
        
    def save_all(self, clean_tex: bool = False) -> tuple[str, str, str | None]:
        txt = self.save_txt()
        tex = self.save_tex()
        try:
            pdf = self.save_pdf(clean_tex=clean_tex)
        except Exception as e:
            print(f"Error generating PDF: {e}")
            pdf = None
        return txt, tex, pdf