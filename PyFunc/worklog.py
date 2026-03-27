from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal
import shutil
import subprocess


# ============================================================================
# Helpers
# ============================================================================

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


def _tex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _fmt_cell(val: Any, float_fmt: str = ".4f") -> str:
    if isinstance(val, float):
        return format(val, float_fmt)
    return str(val)


def _is_separator_line(text: str) -> bool:
    s = text.strip()
    return bool(s) and len(set(s)) == 1 and s[0] in "-=*_~|"


# ============================================================================
# Block model
# ============================================================================

@dataclass
class Block:
    kind: str


@dataclass
class ParagraphBlock(Block):
    text: str

    def __init__(self, text: str):
        super().__init__("paragraph")
        self.text = text


@dataclass
class LineBlock(Block):
    text: str

    def __init__(self, text: str):
        super().__init__("line")
        self.text = text


@dataclass
class EquationBlock(Block):
    latex: str

    def __init__(self, latex: str):
        super().__init__("equation")
        self.latex = latex


@dataclass
class AlignBlock(Block):
    lines: list[str]

    def __init__(self, lines: list[str]):
        super().__init__("align")
        self.lines = lines


@dataclass
class BulletListBlock(Block):
    items: list[str]

    def __init__(self, items: list[str]):
        super().__init__("bullet_list")
        self.items = items


@dataclass
class NumberedListBlock(Block):
    items: list[str]

    def __init__(self, items: list[str]):
        super().__init__("numbered_list")
        self.items = items


@dataclass
class TableBlock(Block):
    headers: list[str]
    rows: list[list[Any]]
    caption: str | None = None
    label: str | None = None
    float_fmt: str = ".4f"
    longtable: bool = False
    alignment: str | None = None

    def __init__(
        self,
        headers: list[str],
        rows: list[list[Any]],
        caption: str | None = None,
        label: str | None = None,
        float_fmt: str = ".4f",
        longtable: bool = False,
        alignment: str | None = None,
    ):
        super().__init__("table")
        self.headers = headers
        self.rows = rows
        self.caption = caption
        self.label = label
        self.float_fmt = float_fmt
        self.longtable = longtable
        self.alignment = alignment


@dataclass
class FigureBlock(Block):
    path: str
    caption: str | None = None
    label: str | None = None
    width: str = r"0.95\textwidth"
    position: str = "H"

    def __init__(
        self,
        path: str,
        caption: str | None = None,
        label: str | None = None,
        width: str = r"0.95\textwidth",
        position: str = "H",
    ):
        super().__init__("figure")
        self.path = path
        self.caption = caption
        self.label = label
        self.width = width
        self.position = position


@dataclass
class RawTexBlock(Block):
    tex: str

    def __init__(self, tex: str):
        super().__init__("raw_tex")
        self.tex = tex


@dataclass
class HorizontalRuleBlock(Block):
    def __init__(self):
        super().__init__("horizontal_rule")


@dataclass
class PageBreakBlock(Block):
    def __init__(self):
        super().__init__("page_break")


@dataclass
class VSpaceBlock(Block):
    amount: str

    def __init__(self, amount: str):
        super().__init__("vspace")
        self.amount = amount


@dataclass
class SectionNode:
    title: str
    level: Literal[1, 2, 3]
    blocks: list[Block] = field(default_factory=list)
    children: list["SectionNode"] = field(default_factory=list)


# ============================================================================
# Main API
# ============================================================================

class WorkLog:
    """
    Structured document logger with:
      - txt output
      - tex output
      - pdf output

    Design rules:
      - prose goes in p()/line()
      - LaTeX math goes in eq()/align()
      - structured data goes in table()/figure()
      - document hierarchy goes in section()/subsection()/subsubsection()
    """

    def __init__(
        self,
        base_name: str,
        title: str,
        author: str | None = None,
        subtitle: str | None = None,
        date_tex: str = r"\today",
    ):
        self.base_name = base_name
        self.title = title
        self.author = author
        self.subtitle = subtitle
        self.date_tex = date_tex

        self.root = SectionNode(title="__root__", level=1)
        self._section_stack: list[SectionNode] = [self.root]

        self._make_title = True
        self._toc = False

    # ---------------------------------------------------------------------
    # document-level controls
    # ---------------------------------------------------------------------

    def maketitle(self, enabled: bool = True) -> None:
        self._make_title = enabled

    def toc(self, enabled: bool = True) -> None:
        self._toc = enabled

    def inline_math(self, latex: str) -> str:
        return f"${latex}$"

    # ---------------------------------------------------------------------
    # structure
    # ---------------------------------------------------------------------

    def section(self, title: str) -> None:
        node = SectionNode(title=title, level=1)
        self.root.children.append(node)
        self._section_stack = [self.root, node]

    def subsection(self, title: str) -> None:
        if len(self._section_stack) < 2:
            self.section("Untitled Section")
        parent = self._section_stack[1]
        node = SectionNode(title=title, level=2)
        parent.children.append(node)
        self._section_stack = [self.root, parent, node]

    def subsubsection(self, title: str) -> None:
        if len(self._section_stack) < 3:
            self.subsection("Untitled Subsection")
        parent = self._section_stack[2]
        node = SectionNode(title=title, level=3)
        parent.children.append(node)
        self._section_stack = [self.root, self._section_stack[1], parent, node]

    @property
    def current(self) -> SectionNode:
        return self._section_stack[-1]

    # ---------------------------------------------------------------------
    # prose / layout
    # ---------------------------------------------------------------------

    def p(self, text: str = "") -> None:
        self.current.blocks.append(ParagraphBlock(str(text)))

    def line(self, text: str = "") -> None:
        self.current.blocks.append(LineBlock(str(text)))

    def bullets(self, items: Iterable[str]) -> None:
        self.current.blocks.append(BulletListBlock([str(x) for x in items]))

    def numbered(self, items: Iterable[str]) -> None:
        self.current.blocks.append(NumberedListBlock([str(x) for x in items]))

    def hline(self) -> None:
        self.current.blocks.append(HorizontalRuleBlock())

    def page_break(self) -> None:
        self.current.blocks.append(PageBreakBlock())

    def vspace(self, amount: str = "1em") -> None:
        self.current.blocks.append(VSpaceBlock(amount))

    # ---------------------------------------------------------------------
    # math
    # ---------------------------------------------------------------------

    def eq(self, latex: str) -> None:
        self.current.blocks.append(EquationBlock(latex.strip()))

    def align(self, *lines: str) -> None:
        cleaned = [str(line).strip() for line in lines if str(line).strip()]
        self.current.blocks.append(AlignBlock(cleaned))

    # ---------------------------------------------------------------------
    # tables / figures / raw tex
    # ---------------------------------------------------------------------

    def table(
        self,
        headers: list[str],
        rows: Iterable[Iterable[Any]],
        caption: str | None = None,
        label: str | None = None,
        float_fmt: str = ".4f",
        longtable: bool = False,
        alignment: str | None = None,
    ) -> None:
        self.current.blocks.append(
            TableBlock(
                headers=list(headers),
                rows=[list(r) for r in rows],
                caption=caption,
                label=label,
                float_fmt=float_fmt,
                longtable=longtable,
                alignment=alignment,
            )
        )

    def figure(
        self,
        path: str,
        caption: str | None = None,
        label: str | None = None,
        width: str = r"0.95\textwidth",
        position: str = "H",
    ) -> None:
        self.current.blocks.append(
            FigureBlock(path, caption=caption, label=label, width=width, position=position)
        )

    def raw_tex(self, tex: str) -> None:
        self.current.blocks.append(RawTexBlock(tex))

    # =========================================================================
    # TXT rendering
    # =========================================================================

    def _render_txt_block(self, block: Block, out: list[str]) -> None:
        if block.kind == "paragraph":
            text = _txt_safe(block.text)
            out.append(text)
            out.append("")
        elif block.kind == "line":
            out.append(_txt_safe(block.text))
        elif block.kind == "equation":
            out.append(f"[MATH] {block.latex}")
        elif block.kind == "align":
            out.append("[ALIGN]")
            out.extend(block.lines)
        elif block.kind == "bullet_list":
            for item in block.items:
                out.append(f"- {_txt_safe(item)}")
            out.append("")
        elif block.kind == "numbered_list":
            for i, item in enumerate(block.items, start=1):
                out.append(f"{i}. {_txt_safe(item)}")
            out.append("")
        elif block.kind == "table":
            headers = [_txt_safe(h) for h in block.headers]
            rows = [[_fmt_cell(v, block.float_fmt) for v in row] for row in block.rows]
            widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    widths[i] = max(widths[i], len(cell))

            def fmt_row(vals: list[str]) -> str:
                return " | ".join(vals[i].ljust(widths[i]) for i in range(len(vals)))

            out.append(fmt_row(headers))
            out.append("-+-".join("-" * w for w in widths))
            for row in rows:
                out.append(fmt_row(row))
            if block.caption:
                out.append(f"Caption: {_txt_safe(block.caption)}")
            if block.label:
                out.append(f"Label: {block.label}")
            out.append("")
        elif block.kind == "figure":
            out.append(f"[FIGURE] {block.path}")
            if block.caption:
                out.append(f"Caption: {_txt_safe(block.caption)}")
            if block.label:
                out.append(f"Label: {block.label}")
            out.append("")
        elif block.kind == "raw_tex":
            out.append("[RAW_TEX]")
            out.append(block.tex)
            out.append("")
        elif block.kind == "horizontal_rule":
            out.append("-" * 60)
        elif block.kind == "page_break":
            out.append("")
            out.append("[PAGE BREAK]")
            out.append("")
        elif block.kind == "vspace":
            out.append("")

    def _render_txt_node(self, node: SectionNode, out: list[str]) -> None:
        if node.title != "__root__":
            if node.level == 1:
                out.append("")
                out.append(node.title)
                out.append("=" * max(20, len(node.title)))
                out.append("")
            elif node.level == 2:
                out.append(node.title)
                out.append("-" * max(8, len(node.title)))
                out.append("")
            elif node.level == 3:
                out.append(node.title)
                out.append("~" * max(6, len(node.title)))
                out.append("")

        for block in node.blocks:
            self._render_txt_block(block, out)

        for child in node.children:
            self._render_txt_node(child, out)

    def save_txt(self) -> str:
        out: list[str] = []
        out.append(_txt_safe(self.title))
        if self.subtitle:
            out.append(_txt_safe(self.subtitle))
        if self.author:
            out.append(_txt_safe(self.author))
        out.append("=" * 80)
        self._render_txt_node(self.root, out)
        path = f"{self.base_name}.txt"
        Path(path).write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
        return path

    # =========================================================================
    # TEX rendering
    # =========================================================================

    def _tex_preamble(self) -> list[str]:
        lines = [
            r"\documentclass[12pt]{article}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{amsmath}",
            r"\usepackage{amssymb}",
            r"\usepackage{booktabs}",
            r"\usepackage{longtable}",
            r"\usepackage{graphicx}",
            r"\usepackage{float}",
            r"\usepackage{caption}",
            r"\usepackage{hyperref}",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\parskip}{0.6em}",
            "",
            rf"\title{{{_tex_escape(self.title)}}}",
            rf"\date{{{self.date_tex}}}",
        ]
        if self.author:
            lines.append(rf"\author{{{_tex_escape(self.author)}}}")
        lines.extend(["", r"\begin{document}", ""])
        if self._make_title:
            lines.append(r"\maketitle")
            lines.append("")
        if self.subtitle:
            lines.append(rf"\begin{{center}}\large {_tex_escape(self.subtitle)}\end{{center}}")
            lines.append("")
        if self._toc:
            lines.append(r"\tableofcontents")
            lines.append(r"\newpage")
            lines.append("")
        return lines

    def _render_tex_block(self, block: Block, out: list[str]) -> None:
        if block.kind == "paragraph":
            text = str(block.text)
            if not text.strip():
                out.append(r"\par")
            else:
                out.append(_tex_escape(text))
                out.append(r"\par")
            out.append("")
        elif block.kind == "line":
            text = str(block.text)
            if not text.strip():
                out.append(r"\par")
            elif _is_separator_line(text):
                out.append(r"\hrule")
            else:
                out.append(_tex_escape(text) + r"\\")
            out.append("")
        elif block.kind == "equation":
            out.append(r"\[")
            out.append(block.latex)
            out.append(r"\]")
            out.append("")
        elif block.kind == "align":
            out.append(r"\begin{align*}")
            for i, line in enumerate(block.lines):
                suffix = r" \\" if i < len(block.lines) - 1 else ""
                out.append(line + suffix)
            out.append(r"\end{align*}")
            out.append("")
        elif block.kind == "bullet_list":
            out.append(r"\begin{itemize}")
            for item in block.items:
                out.append(rf"\item {_tex_escape(item)}")
            out.append(r"\end{itemize}")
            out.append("")
        elif block.kind == "numbered_list":
            out.append(r"\begin{enumerate}")
            for item in block.items:
                out.append(rf"\item {_tex_escape(item)}")
            out.append(r"\end{enumerate}")
            out.append("")
        elif block.kind == "table":
            colspec = block.alignment or ("c" * len(block.headers))
            if block.longtable:
                out.append(rf"\begin{{longtable}}{{{colspec}}}")
                if block.caption:
                    if block.label:
                        out.append(rf"\caption{{{_tex_escape(block.caption)}}}\label{{{block.label}}}\\")
                    else:
                        out.append(rf"\caption{{{_tex_escape(block.caption)}}}\\")
                out.append(r"\toprule")
                out.append(" & ".join(_tex_escape(h) for h in block.headers) + r" \\")
                out.append(r"\midrule")
                out.append(r"\endfirsthead")
                out.append(r"\toprule")
                out.append(" & ".join(_tex_escape(h) for h in block.headers) + r" \\")
                out.append(r"\midrule")
                out.append(r"\endhead")
                out.append(r"\bottomrule")
                out.append(r"\endfoot")
                for row in block.rows:
                    cells = []
                    for val in row:
                        if isinstance(val, float):
                            cells.append(_fmt_cell(val, block.float_fmt))
                        else:
                            cells.append(_tex_escape(str(val)))
                    out.append(" & ".join(cells) + r" \\")
                out.append(r"\end{longtable}")
                out.append("")
            else:
                out.append(r"\begin{table}[H]")
                out.append(r"\centering")
                out.append(rf"\begin{{tabular}}{{{colspec}}}")
                out.append(r"\toprule")
                out.append(" & ".join(_tex_escape(h) for h in block.headers) + r" \\")
                out.append(r"\midrule")
                for row in block.rows:
                    cells = []
                    for val in row:
                        if isinstance(val, float):
                            cells.append(_fmt_cell(val, block.float_fmt))
                        else:
                            cells.append(_tex_escape(str(val)))
                    out.append(" & ".join(cells) + r" \\")
                out.append(r"\bottomrule")
                out.append(r"\end{tabular}")
                if block.caption:
                    out.append(rf"\caption{{{_tex_escape(block.caption)}}}")
                if block.label:
                    out.append(rf"\label{{{block.label}}}")
                out.append(r"\end{table}")
                out.append("")
        elif block.kind == "figure":
            out.append(rf"\begin{{figure}}[{block.position}]")
            out.append(r"\centering")
            out.append(rf"\includegraphics[width={block.width}]{{{block.path}}}")
            if block.caption:
                out.append(rf"\caption{{{_tex_escape(block.caption)}}}")
            if block.label:
                out.append(rf"\label{{{block.label}}}")
            out.append(r"\end{figure}")
            out.append("")
        elif block.kind == "raw_tex":
            out.append(block.tex)
            out.append("")
        elif block.kind == "horizontal_rule":
            out.append(r"\hrule")
            out.append("")
        elif block.kind == "page_break":
            out.append(r"\newpage")
            out.append("")
        elif block.kind == "vspace":
            out.append(rf"\vspace{{{block.amount}}}")
            out.append("")

    def _render_tex_node(self, node: SectionNode, out: list[str]) -> None:
        if node.title != "__root__":
            if node.level == 1:
                out.append(rf"\section*{{{_tex_escape(node.title)}}}")
                out.append("")
            elif node.level == 2:
                out.append(rf"\subsection*{{{_tex_escape(node.title)}}}")
                out.append("")
            elif node.level == 3:
                out.append(rf"\subsubsection*{{{_tex_escape(node.title)}}}")
                out.append("")

        for block in node.blocks:
            self._render_tex_block(block, out)

        for child in node.children:
            self._render_tex_node(child, out)

    def save_tex(self) -> str:
        out = self._tex_preamble()
        self._render_tex_node(self.root, out)
        out.extend([r"\end{document}", ""])
        path = f"{self.base_name}.tex"
        Path(path).write_text("\n".join(out), encoding="utf-8")
        return path

    # =========================================================================
    # PDF build
    # =========================================================================

    def save_pdf(self, runs: int = 1) -> str | None:
        tex_path = self.save_tex()
        engine = shutil.which("pdflatex")
        if engine is None:
            print("pdflatex not found; wrote .tex only")
            return None

        pdf_path = f"{self.base_name}.pdf"
        ok = True

        for _ in range(max(1, runs)):
            result = subprocess.run(
                [engine, "-interaction=nonstopmode", tex_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                ok = False
                print("pdflatex failed; check the .log file")
                print(result.stdout[-3000:])
                break

        return pdf_path if ok and Path(pdf_path).exists() else None

    def save_all(self, runs: int = 1) -> tuple[str, str, str | None]:
        txt = self.save_txt()
        tex = self.save_tex()
        pdf = self.save_pdf(runs=runs)
        return txt, tex, pdf