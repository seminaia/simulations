from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal
import shutil
import subprocess

from pylatex import (
    Document,
    Section,
    Subsection,
    Subsubsection,
    Figure,
    Tabular,
    LongTable,
    Itemize,
    Enumerate,
    NoEscape,
    Command,
)
from pylatex.utils import escape_latex


# =============================================================================
# Helpers
# =============================================================================

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


# =============================================================================
# Block model
# =============================================================================

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
class InlineMathToken:
    latex: str


@dataclass
class MixedParagraphBlock(Block):
    parts: list[Any]

    def __init__(self, parts: list[Any]):
        super().__init__("mixed_paragraph")
        self.parts = parts


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


# =============================================================================
# WorkLog
# =============================================================================

class WorkLog:
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

        self.root = SectionNode("__root__", level=1)
        self._stack: list[SectionNode] = [self.root]

        self._make_title = True
        self._toc = False

    # -------------------------------------------------------------------------
    # Document controls
    # -------------------------------------------------------------------------

    def maketitle(self, enabled: bool = True) -> None:
        self._make_title = enabled

    def toc(self, enabled: bool = True) -> None:
        self._toc = enabled

    # -------------------------------------------------------------------------
    # Structure
    # -------------------------------------------------------------------------

    def section(self, title: str) -> None:
        node = SectionNode(title, level=1)
        self.root.children.append(node)
        self._stack = [self.root, node]

    def subsection(self, title: str) -> None:
        if len(self._stack) < 2:
            self.section("Untitled Section")
        parent = self._stack[1]
        node = SectionNode(title, level=2)
        parent.children.append(node)
        self._stack = [self.root, parent, node]

    def subsubsection(self, title: str) -> None:
        if len(self._stack) < 3:
            self.subsection("Untitled Subsection")
        parent = self._stack[2]
        node = SectionNode(title, level=3)
        parent.children.append(node)
        self._stack = [self.root, self._stack[1], parent, node]

    @property
    def current(self) -> SectionNode:
        return self._stack[-1]

    # -------------------------------------------------------------------------
    # Prose
    # -------------------------------------------------------------------------

    def p(self, text: str = "") -> None:
        self.current.blocks.append(ParagraphBlock(str(text)))

    def line(self, text: str = "") -> None:
        self.current.blocks.append(LineBlock(str(text)))

    def im(self, latex: str) -> InlineMathToken:
        return InlineMathToken(latex)

    def px(self, *parts: Any) -> None:
        self.current.blocks.append(MixedParagraphBlock(list(parts)))

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

    # -------------------------------------------------------------------------
    # Math
    # -------------------------------------------------------------------------

    def eq(self, latex: str) -> None:
        self.current.blocks.append(EquationBlock(latex.strip()))

    def align(self, *lines: str) -> None:
        cleaned = [str(line).strip() for line in lines if str(line).strip()]
        self.current.blocks.append(AlignBlock(cleaned))

    # -------------------------------------------------------------------------
    # Tables / figures / raw tex
    # -------------------------------------------------------------------------

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

    def long_table(
        self,
        headers: list[str],
        rows: Iterable[Iterable[Any]],
        caption: str | None = None,
        label: str | None = None,
        float_fmt: str = ".4f",
        alignment: str | None = None,
    ) -> None:
        self.table(
            headers=headers,
            rows=rows,
            caption=caption,
            label=label,
            float_fmt=float_fmt,
            longtable=True,
            alignment=alignment,
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

    # =============================================================================
    # TXT rendering
    # =============================================================================

    def _render_txt_block(self, block: Block, out: list[str]) -> None:
        if block.kind == "paragraph":
            out.append(_txt_safe(block.text))
            out.append("")
        elif block.kind == "line":
            out.append(_txt_safe(block.text))
        elif block.kind == "mixed_paragraph":
            parts = []
            for part in block.parts:
                if isinstance(part, InlineMathToken):
                    parts.append(part.latex)
                else:
                    parts.append(_txt_safe(str(part)))
            out.append("".join(parts))
            out.append("")
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

    # =============================================================================
    # PyLaTeX rendering
    # =============================================================================

    def _new_document(self) -> Document:
        doc = Document(self.base_name, geometry_options={"margin": "1in"})
        doc.preamble.append(Command("usepackage", "amsmath"))
        doc.preamble.append(Command("usepackage", "amssymb"))
        doc.preamble.append(Command("usepackage", "booktabs"))
        doc.preamble.append(Command("usepackage", "longtable"))
        doc.preamble.append(Command("usepackage", "float"))
        doc.preamble.append(Command("usepackage", "caption"))
        doc.preamble.append(Command("usepackage", "hyperref"))
        doc.preamble.append(Command("title", self.title))
        if self.author:
            doc.preamble.append(Command("author", self.author))
        doc.preamble.append(Command("date", NoEscape(self.date_tex)))
        doc.append(NoEscape(r"\setlength{\parindent}{0pt}"))
        doc.append(NoEscape(r"\setlength{\parskip}{0.6em}"))
        if self._make_title:
            doc.append(NoEscape(r"\maketitle"))
        if self.subtitle:
            doc.append(NoEscape(rf"\begin{{center}}\large {escape_latex(self.subtitle)}\end{{center}}"))
        if self._toc:
            doc.append(NoEscape(r"\tableofcontents"))
            doc.append(NoEscape(r"\newpage"))
        return doc

    def _append_tex_block(self, container: Any, block: Block) -> None:
        if block.kind == "paragraph":
            if block.text.strip():
                container.append(escape_latex(block.text))
            container.append(NoEscape(r"\par"))
        elif block.kind == "line":
            if block.text.strip():
                container.append(escape_latex(block.text))
                container.append(NoEscape(r"\\"))
            else:
                container.append(NoEscape(r"\par"))
        elif block.kind == "mixed_paragraph":
            for part in block.parts:
                if isinstance(part, InlineMathToken):
                    container.append(NoEscape(f"${part.latex}$"))
                else:
                    container.append(escape_latex(str(part)))
            container.append(NoEscape(r"\par"))
        elif block.kind == "equation":
            container.append(NoEscape(r"\["))
            container.append(NoEscape(block.latex))
            container.append(NoEscape(r"\]"))
        elif block.kind == "align":
            container.append(NoEscape(r"\begin{align*}"))
            for i, line in enumerate(block.lines):
                suffix = r" \\" if i < len(block.lines) - 1 else ""
                container.append(NoEscape(line + suffix))
            container.append(NoEscape(r"\end{align*}"))
        elif block.kind == "bullet_list":
            with container.create(Itemize()) as itemize:
                for item in block.items:
                    itemize.add_item(item)
        elif block.kind == "numbered_list":
            with container.create(Enumerate()) as enum:
                for item in block.items:
                    enum.add_item(item)
        elif block.kind == "table":
            colspec = block.alignment or ("c" * len(block.headers))
            if block.longtable:
                lt = LongTable(colspec)
                lt.add_hline()
                lt.add_row(block.headers)
                lt.add_hline()
                lt.end_table_header()
                for row in block.rows:
                    lt.add_row([
                        _fmt_cell(v, block.float_fmt) if isinstance(v, float) else str(v)
                        for v in row
                    ])
                lt.add_hline()
                container.append(lt)
                if block.caption:
                    container.append(NoEscape(rf"\captionof{{table}}{{{escape_latex(block.caption)}}}"))
                if block.label:
                    container.append(NoEscape(rf"\label{{{block.label}}}"))
            else:
                container.append(NoEscape(r"\begin{table}[H]"))
                container.append(NoEscape(r"\centering"))
                tab = Tabular(colspec)
                tab.add_hline()
                tab.add_row(block.headers)
                tab.add_hline()
                for row in block.rows:
                    tab.add_row([
                        _fmt_cell(v, block.float_fmt) if isinstance(v, float) else str(v)
                        for v in row
                    ])
                tab.add_hline()
                container.append(tab)
                if block.caption:
                    container.append(NoEscape(rf"\caption{{{escape_latex(block.caption)}}}"))
                if block.label:
                    container.append(NoEscape(rf"\label{{{block.label}}}"))
                container.append(NoEscape(r"\end{table}"))
        elif block.kind == "figure":
            with container.create(Figure(position=block.position)) as fig:
                fig.add_image(block.path, width=NoEscape(block.width))
                if block.caption:
                    fig.add_caption(block.caption)
                if block.label:
                    fig.append(NoEscape(rf"\label{{{block.label}}}"))
        elif block.kind == "raw_tex":
            container.append(NoEscape(block.tex))
        elif block.kind == "horizontal_rule":
            container.append(NoEscape(r"\hrule"))
        elif block.kind == "page_break":
            container.append(NoEscape(r"\newpage"))
        elif block.kind == "vspace":
            container.append(NoEscape(rf"\vspace{{{block.amount}}}"))

    def _append_node(self, parent: Any, node: SectionNode) -> None:
        if node.title == "__root__":
            for child in node.children:
                self._append_node(parent, child)
            return

        if node.level == 1:
            env = Section(node.title, numbering=False)
        elif node.level == 2:
            env = Subsection(node.title, numbering=False)
        else:
            env = Subsubsection(node.title, numbering=False)

        with parent.create(env) as section_container:
            for block in node.blocks:
                self._append_tex_block(section_container, block)
            for child in node.children:
                self._append_node(section_container, child)

    def save_tex(self) -> str:
        doc = self._new_document()
        self._append_node(doc, self.root)
        doc.generate_tex(self.base_name)
        return f"{self.base_name}.tex"

    # =============================================================================
    # PDF
    # =============================================================================

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