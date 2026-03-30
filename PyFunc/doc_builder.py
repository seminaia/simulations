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
    Alignat,
    NoEscape,
    Command,
)
from pylatex.utils import escape_latex
    
"""DocumentBuilder is a Module that provides a class for building documents
programmatically using PyLaTeX. It allows the user to construct a report
with sections, subsections, equations, tables, figures, and other content
blocks, and then render the report to plain text, LaTeX source, or PDF."""

# =============================================================================
# Helpers
# =============================================================================

def _txt_safe(text: str) -> str:
    """_summary_
    Simple function to replace certain characters with safer alternatives for plain text output.
    This is used when rendering the report to .txt format, to avoid issues with characters that may not display well in plain text.
    
    Args:
        text (str): The text to be made safe for plain text output.

    Returns:
        str: The text with certain characters replaced by safer alternatives.
    """
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
    """_summary_
    Simple function to format a cell value for display in a table.
    If the value is a float, it is formatted according to the specified float format.
    Otherwise, the value is converted to a string.

    Args:
        val (Any): The value to be formatted.
        float_fmt (str, optional): The format string to use for floats. Defaults to ".4f".

    Returns:
        str: The formatted value.
    """
    if isinstance(val, float):
        return format(val, float_fmt)
    return str(val)


# =============================================================================
# Block model
# =============================================================================

@dataclass
class Block:
    """Base class for all content blocks in the worklog.
    Block is the base class for all content blocks in the report. 
    Each block has a kind that determines how it should be rendered.
    
    Args:
        kind (str): The type of the block, which determines how it should be rendered.
    """
    kind: str

@dataclass
class ParagraphBlock(Block):
    """ParagraphBlock represents a block of text that should be rendered as a paragraph. 
    It can contain multiple lines of text, and will be separated from other blocks by vertical space when rendered.
    
    Args:
        Block (_type_): The base Block class that this inherits from.
    """
    text: str
    def __init__(self, text: str):
        super().__init__("paragraph")
        self.text = text


@dataclass
class LineBlock(Block):
    """LineBlock represents a single line of text that should be rendered with a line break after it. 
        It is similar to ParagraphBlock but is intended for cases where you want to control line breaks more explicitly.
        
    Args:
        Block (_type_): The base Block class that this inherits from.
    """
    text: str
    def __init__(self, text: str):
        super().__init__("line")
        self.text = text


@dataclass
class InlineMathToken:
    """InlineMathToken represents a piece of inline math that can be included within a MixedParagraphBlock. 
    It is not a Block itself, but rather a token that can be included in the parts list of a MixedParagraphBlock to indicate that this part should be rendered as inline math.
    """
    latex: str


@dataclass
class MixedParagraphBlock(Block):
    """MixedParagraphBlock represents a block of text that can contain a mix of regular text and inline math. 
    The parts list can contain strings (for regular text) and InlineMathToken instances (for inline math). When rendered, the strings will be escaped for LaTeX and the InlineMathTokens will be rendered as inline math.

    Args:
        Block (_type_): The base Block class that this inherits from.
    """
    parts: list[Any]
    def __init__(self, parts: list[Any]):
        super().__init__("mixed_paragraph")
        self.parts = parts


@dataclass
class EquationBlock(Block):
    f"""EquationBlock represents a block of display math.
    The latex string should contain the LaTeX code for the math, without the surrounding {']'} {']'} or $$ delimiters. 
    When rendered, it will be wrapped in display math delimiters.
    Args:
        Block (_type_): The base Block class that this inherits from.
    """
    latex: str
    def __init__(self, latex: str):
        super().__init__("equation")
        self.latex = latex


@dataclass
class AlignBlock(Block):
    """AlignBlock represents a block of aligned equations. Each line in the lines list should contain a single equation, and the equations will be aligned at the equal signs when rendered.
    Args:
        Block (_type_): The base Block class that this inherits from.
    """
    lines: list[str]
    def __init__(self, lines: list[str]):
        super().__init__("align")
        self.lines = lines


@dataclass
class BulletListBlock(Block):
    """BulletListBlock represents a block of bulleted list items. Each item in the items list will be rendered as a bullet point in the output.
    Args:
        Block (_type_): The base Block class that this inherits from.
    """
    items: list[str]
    def __init__(self, items: list[str]):
        super().__init__("bullet_list")
        self.items = items


@dataclass
class NumberedListBlock(Block):
    """NumberedListBlock represents a block of numbered list items. Each item in the items list will be rendered as a numbered point in the output.
    Args:
        Block (_type_): The base Block class that this inherits from.
    """
    items: list[str]

    def __init__(self, items: list[str]):
        super().__init__("numbered_list")
        self.items = items


@dataclass
class TableBlock(Block):
    """TableBlock represents a bolck of tabular data. It contains the headers and rows of the table, as well as optional caption and label for referencing. The float_fmt is used to format any float values in the table when rendering to text or LaTeX. The longtable flag indicates whether to use the longtable environment in LaTeX, which allows tables to span multiple pages. The alignment string can specify the column alignment for LaTeX rendering (e.g., "lcr" for left, center, right).
    Args:
        Block (_type_): The base Block class that this inherits from.
    """
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
    """FigureBlock represents a block containing a figure. It includes the path to the image file, as well as optional caption and label for referencing. The width and position parameters control how the figure is displayed in the LaTeX output.
    
    Args:
        Block (_type_): The base Block class that this inherits from.
    """
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
    """RawTexBlock represents a block of raw LaTeX code that will be included in the output without any modification. This can be used for cases where you want to include custom LaTeX that doesn't fit into the other block types, or for including LaTeX commands that affect formatting rather than content.
    Args:
        Block (_type_): The base Block class that this inherits from.
    """
    tex: str
    def __init__(self, tex: str):
        super().__init__("raw_tex")
        self.tex = tex


@dataclass
class HorizontalRuleBlock(Block):
    """HorizontalRuleBlock represents a horizontal rule (line) that can be used to visually separate sections of the report. When rendered, it will produce a horizontal line across the page.
    Args:
        Block (_type_): The base Block class that this inherits from.
    """
    def __init__(self):
        super().__init__("horizontal_rule")


@dataclass
class PageBreakBlock(Block):
    """PageBreakBlock represents a page break in the document. When rendered, it will start a new page.
    Args:
        Block (_type_): The base Block class that this inherits from.
    """
    def __init__(self):
        super().__init__("page_break")


@dataclass
class VSpaceBlock(Block):
    """VSpaceBlock represents a vertical space in the document. The amount parameter specifies how much vertical space to include, and can be specified in any units that LaTeX accepts (e.g., "1em", "0.5in", "10pt").
    Args:
        Block (_type_): The base Block class that this inherits from.
    """
    amount: str
    def __init__(self, amount: str):
        super().__init__("vspace")
        self.amount = amount


@dataclass
class SectionNode:
    """SectionNode represents a section of the report. It has a title, a level (1 for section, 2 for subsection, 3 for subsubsection), a list of blocks that belong to this section, and a list of child sections. The blocks are the content of the section, while the children are subsections that belong to this section.
    Args:
        title (str): The title of the section.
        level (Literal[1, 2, 3]): The level of the section (1 for section, 2 for subsection, 3 for subsubsection).
        blocks (list[Block], optional): The list of content blocks that belong to this section. Defaults to an empty list.
        children (list["SectionNode"], optional): The list of child sections that belong to this section. Defaults to an empty list.
    """
    title: str
    level: Literal[1, 2, 3]
    blocks: list[Block] = field(default_factory=list)
    children: list["SectionNode"] = field(default_factory=list)


# =============================================================================
# DocumentBuilder
# =============================================================================

class DocumentBuilder:
    """
    DocumentBuilder is a simple document building class that allows you to build up a report
    using a simple API. It supports sections, subsections, paragraphs, equations, tables, figures, and more.
    It can render to plain text, LaTeX source, and PDF (via pdflatex). It uses the PyLaTeX library for LaTeX document building.
    The idea is to have a single source of truth for the report content, and be able to easily generate different output formats from it.
    The API is designed to be simple and flexible, allowing you to build up the report in a natural way as you compute results and generate content.
    The internal structure is a tree of sections, each containing a list of blocks (paragraphs, equations, tables, etc.) and child sections. 
    The rendering functions traverse this tree to produce the desired output format.
    
    Args:
        base_name (str): The base name for the output files (without extension).
        title (str): The title of the report.
        author (str | None, optional): The author of the report. Defaults to None.
        subtitle (str | None, optional): The subtitle of the report. Defaults to None.
        date_tex (str, optional): The LaTeX code for the date to be displayed in the report. Defaults to r"\today".
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

        self.root = SectionNode("__root__", level=1)
        self._stack: list[SectionNode] = [self.root]

        self._make_title = True
        self._toc = False

    # -------------------------------------------------------------------------
    # Document controls
    # -------------------------------------------------------------------------

    def maketitle(self, enabled: bool = True) -> None:
        """Enable or disable the title page.

        Args:
            enabled (bool, optional): Whether to include the title page. Defaults to True.
        """
        self._make_title = enabled

    def toc(self, enabled: bool = True) -> None:
        """Enable or disable the table of contents.

        Args:
            enabled (bool, optional): Whether to include the table of contents. Defaults to True.
        """
        self._toc = enabled

    # -------------------------------------------------------------------------
    # Structure
    # -------------------------------------------------------------------------

    def section(self, title: str) -> None:
        """Start a new section.

        Args:
            title (str): The title of the section.
        """
        node = SectionNode(title, level=1)
        self.root.children.append(node)
        self._stack = [self.root, node]

    def subsection(self, title: str) -> None:
        """Start a new subsection.

        Args:
            title (str): The title of the subsection.
        """
        if len(self._stack) < 2:
            self.section("Untitled Section")
        parent = self._stack[1]
        node = SectionNode(title, level=2)
        parent.children.append(node)
        self._stack = [self.root, parent, node]

    def subsubsection(self, title: str) -> None:
        """Start a new subsubsection.

        Args:
            title (str): The title of the subsubsection.
        """
        if len(self._stack) < 3:
            self.subsection("Untitled Subsection")
        parent = self._stack[2]
        node = SectionNode(title, level=3)
        parent.children.append(node)
        self._stack = [self.root, self._stack[1], parent, node]

    @property
    def current(self) -> SectionNode:
        """Get the current section node.

        Returns:
            SectionNode: The current section node.
        """
        return self._stack[-1]

    # -------------------------------------------------------------------------
    # Prose
    # -------------------------------------------------------------------------

    def p(self, text: str = "") -> None:
        self.current.blocks.append(ParagraphBlock(str(text)))

    def line(self, text: str = "") -> None:
        """Add a line of text.

        Args:
            text (str, optional): The text to add. Defaults to "".
        """
        self.current.blocks.append(LineBlock(str(text)))

    def im(self, latex: str) -> InlineMathToken:
        """Add an inline math expression.
        Must Be used inside a mixed paragraph block.

        Args:
            latex (str): The LaTeX code for the inline math expression.

        Returns:
            InlineMathToken: The inline math token.
        """
        return InlineMathToken(latex)

    def px(self, *parts: Any) -> None:
        """Add a mixed paragraph.

        Args:
            parts (Any): The parts of the mixed paragraph.
        """
        self.current.blocks.append(MixedParagraphBlock(list(parts)))

    def bullets(self, items: Iterable[str]) -> None:
        """Add a bullet list.

        Args:
            items (Iterable[str]): The items of the bullet list.
        """
        self.current.blocks.append(BulletListBlock([str(x) for x in items]))

    def numbered(self, items: Iterable[str]) -> None:
        """Add a numbered list.

        Args:
            items (Iterable[str]): The items of the numbered list.
        """
        self.current.blocks.append(NumberedListBlock([str(x) for x in items]))

    def hline(self) -> None:
        """Add a horizontal line.
        """
        self.current.blocks.append(HorizontalRuleBlock())

    def page_break(self) -> None:
        """Add a page break."""
        self.current.blocks.append(PageBreakBlock())
    
    def vspace(self, amount: str = "1em") -> None:
        """Add vertical space.

        Args:
            amount (str, optional): The amount of vertical space. Defaults to "1em".
        """
        self.current.blocks.append(VSpaceBlock(amount))

    # -------------------------------------------------------------------------
    # Math
    # -------------------------------------------------------------------------

    def eq(self, latex: str) -> None:
        """Add an equation.

        Args:
            latex (str): The LaTeX code for the equation.
        """
        self.current.blocks.append(EquationBlock(latex.strip()))

    def align(self, *lines: str) -> None:
        """Add an aligned equation.

        Args:
            lines (str): The lines of the aligned equation.
        """
        cleaned = [str(object=line).strip() for line in lines if str(line).strip()]
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
        """Add a table.

        Args:
            headers (list[str]): The headers of the table.
            rows (Iterable[Iterable[Any]]): The rows of the table.
            caption (str | None, optional): The caption of the table. Defaults to None.
            label (str | None, optional): The label of the table. Defaults to None.
            float_fmt (str, optional): The format for floating-point numbers. Defaults to ".4f".
            longtable (bool, optional): Whether to use a longtable. Defaults to False.
            alignment (str | None, optional): The alignment of the table. Defaults to None.
        """
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
        """Add a long table (a table that can span multiple pages in LaTeX).

        Args:
            headers (list[str]): The headers of the table.
            rows (Iterable[Iterable[Any]]): The rows of the table.
            caption (str | None, optional): The caption of the table. Defaults to None.
            label (str | None, optional): The label of the table. Defaults to None.
            float_fmt (str, optional): The format for floating-point numbers. Defaults to ".4f".
            alignment (str | None, optional): The alignment of the table. Defaults to None.
        """
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
        """Add a figure to the document.
        
        Args:
            path (str): The path to the image file to include in the document.
            caption (str | None, optional): The caption of the figure. Defaults to None.
            label (str | None, optional): The label of the figure. Defaults to None.
            width (str, optional): The width of the figure. Defaults to r"0.95\textwidth".
            position (str, optional): The LaTeX figure placement specifier (e.g., "H", "t", "b", "p"). Defaults to "H".
        """
        self.current.blocks.append(
            FigureBlock(path, caption=caption, label=label, width=width, position=position)
        )

    def raw_tex(self, tex: str) -> None:
        """Add raw LaTeX code to the document.

        Args:
            tex (str): The raw LaTeX code to include in the document.
        """
        self.current.blocks.append(RawTexBlock(tex))

    # =============================================================================
    # TXT rendering
    # =============================================================================

    def _render_txt_block(self, block: Block, out: list[str]) -> None:
        """Render a single block to plain text format.
        This function handles the different block types and appends the appropriate text representation to the output list.
        
        Args:
            block (Block): The block to render.
            out (list[str]): The list of strings to which the rendered text will be appended.
        """

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
        """Render a section node and its children to plain text format. 
        This function handles rendering the section title according to its level and then recursively
        renders all blocks and child sections contained within the node.


        Args:
            node (SectionNode): _description_
            out (list[str]): _description_
        """
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
    def _render_table_cell(self, val:Any, float_fmt: str=".4f") -> Any:
        if isinstance(val, float):
            return format(val, float_fmt)
        if isinstance(val, InlineMathToken):
            return NoEscape(f"${val.latex}$")
        return str(val)
    
    def _new_document(self) -> Document:
        """Create a new PyLaTeX Document object with the preamble configured for this worklog.
        This includes loading common packages, setting the title, author, date, and page geometry,
        and optionally including the title and table of contents in the document body.

        Returns:
            Document: A configured PyLaTeX Document object ready for content to be appended.
        """
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
        """Append a single block to a PyLaTeX container.
        This function handles the different block types and appends the appropriate LaTeX
        representation to the given container, which can be a Document, Section, or other
        PyLaTeX environment.

        Args:
            container (Any): The PyLaTeX container to append content to.
            block (Block): The block to render into LaTeX and append.
        """
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
<<<<<<< Updated upstream
elif block.kind == "table":
    colspec = block.alignment or ("c" * len(block.headers))
    if block.longtable:
        lt = LongTable(colspec, escape=False)
        lt.add_hline()
        lt.add_row([_render_table_cell(h, block.float_fmt) for h in block.headers])
        lt.add_hline()
        lt.end_table_header()
        for row in block.rows:
            lt.add_row([_render_table_cell(v, block.float_fmt) for v in row])
        lt.add_hline()
        container.append(lt)
        if block.caption:
            container.append(NoEscape(rf"\captionof{{table}}{{{escape_latex(block.caption)}}}"))
        if block.label:
            container.append(NoEscape(rf"\label{{{block.label}}}"))
    else:
        container.append(NoEscape(r"\begin{table}[H]"))
        container.append(NoEscape(r"\centering"))
        tab = Tabular(colspec, escape=False)
        tab.add_hline()
        tab.add_row([_render_table_cell(h, block.float_fmt) for h in block.headers])
        tab.add_hline()
        for row in block.rows:
            tab.add_row([_render_table_cell(v, block.float_fmt) for v in row])
        tab.add_hline()
        container.append(tab)
        if block.caption:
            container.append(NoEscape(rf"\caption{{{escape_latex(block.caption)}}}"))
        if block.label:
            container.append(NoEscape(rf"\label{{{block.label}}}"))
        container.append(NoEscape(r"\end{table}"))
=======
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
                        self._render_table_cell(v, block.float_fmt) for v in row
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
                        self._render_table_cell(v, block.float_fmt) for v in row
                    ])
                tab.add_hline()
                container.append(tab)
                if block.caption:
                    container.append(NoEscape(rf"\caption{{{escape_latex(block.caption)}}}"))
                if block.label:
                    container.append(NoEscape(rf"\label{{{block.label}}}"))
                container.append(NoEscape(r"\end{table}"))
>>>>>>> Stashed changes
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
        """Append a section node and all of its content recursively to a PyLaTeX container.
        This function handles the special root node case, creates the appropriate Section,
        Subsection, or Subsubsection environment based on the node's level, appends all
        blocks in the node, and then recursively appends all child nodes.

        Args:
            parent (Any): The PyLaTeX container to append the section to.
            node (SectionNode): The section node to render into LaTeX and append.
        """

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
        """Generate the LaTeX .tex file for this worklog.
        This function creates a new PyLaTeX Document, appends the entire section tree
        starting from the root node, generates the .tex file, and returns the path
        to the generated .tex file.

        Returns:
            str: The path to the generated .tex file.
        """
        doc = self._new_document()
        self._append_node(doc, self.root)
        doc.generate_tex(self.base_name)
        return f"{self.base_name}.tex"

    # =============================================================================
    # PDF
    # =============================================================================

    def save_pdf(self, runs: int = 1) -> str | None:
        """Generate the PDF for this worklog by first generating the .tex file and then
        invoking pdflatex the specified number of times. If pdflatex is not found,
        only the .tex file is written and None is returned.

        Args:
            runs (int, optional): The number of times to run pdflatex to resolve
                references. Defaults to 1.

        Returns:
            str | None: The path to the generated PDF if successful, otherwise None.
        """
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
        """ Save .pdf, .tex, .txt versions of the worklog.

        Args:
            runs (int, optional): The number of times to run pdflatex to resolve
                references when generating the PDF. Defaults to 1.

        Returns:
            tuple[str, str, str | None]: A tuple containing the paths to the generated
                .txt, .tex, and .pdf files, respectively. The PDF path will be None
                if PDF generation failed or pdflatex was not found.
        """
        txt = self.save_txt()
        tex = self.save_tex()
        pdf = self.save_pdf(runs=runs)
        return txt, tex, pdf