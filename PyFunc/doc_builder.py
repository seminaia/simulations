from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
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
    Math,
)
from pylatex.utils import escape_latex


class DocumentBuilder:
    """
    PyLaTeX-first document builder.

    Main idea:
        - Build the LaTeX document directly using PyLaTeX objects.
        - Keep a parallel plain-text log.
        - Support both article reports and Beamer slide decks.
    """

    def __init__(
        self,
        base_name: str,
        title: str,
        author: str | None = None,
        subtitle: str | None = None,
        institute: str | None = None,
        date_tex: str = r"\today",
        document_options: str | None = None,
        documentclass: str = "article",
        margin: str = "1in",
    ):
        self.base_name = base_name
        self.title = title
        self.author = author
        self.subtitle = subtitle
        self.institute = institute
        self.date_tex = date_tex
        self.document_options = document_options
        self.documentclass = documentclass
        self.margin = margin

        self.is_beamer = documentclass == "beamer"

        self.text_lines: list[str] = []
        self._make_title = True
        self._toc = False

        self._raw_preamble: list[str] = []
        self._raw_body: list[str] = []
        self._raw_tail: list[str] = []

        self._beamer_theme = "Madrid"
        self._beamer_color_theme: str | None = None
        self._beamer_theme_options: str | None = None

        self.doc = self._new_document()
        self._build_preamble_and_title()
        
        self.current = self.doc
        self.stack: list[Any] = [self.doc]

        self._active_frame: list[Any] | None = None
        self._active_frame_title: str | None = None
        self._active_frame_fragile: bool = False

    # =========================================================================
    # Configuration
    # =========================================================================

    def maketitle(self, enabled: bool = True) -> None:
        self._make_title = enabled

    def toc(self, enabled: bool = True) -> None:
        self._toc = enabled

    def beamer_theme(
        self,
        theme: str = "Madrid",
        color_theme: str | None = None,
        theme_options: str | None = None,
    ) -> None:
        self._beamer_theme = theme
        self._beamer_color_theme = color_theme
        self._beamer_theme_options = theme_options

    def raw_preamble(self, tex: str) -> None:
        self._raw_preamble.append(tex)

    def raw_body(self, tex: str) -> None:
        self._raw_body.append(tex)

    def raw_tail(self, tex: str) -> None:
        self._raw_tail.append(tex)

    # =========================================================================
    # Document creation
    # =========================================================================

    def _new_document(self) -> Document:
        if self.is_beamer:
            doc = Document(
                self.base_name,
                documentclass="beamer",
                document_options=self.document_options,
                page_numbers=False,
            )
        else:
            doc = Document(
                self.base_name,
                documentclass="article",
                document_options=self.document_options,
                geometry_options={"margin": self.margin},
            )

        return doc

    def _build_preamble_and_title(self) -> None:
        doc = self.doc

        if self.is_beamer:
            if self._beamer_theme_options:
                doc.preamble.append(
                    NoEscape(
                        rf"\usetheme[{self._beamer_theme_options}]{{{self._beamer_theme}}}"
                    )
                )
            else:
                doc.preamble.append(NoEscape(rf"\usetheme{{{self._beamer_theme}}}"))

            if self._beamer_color_theme:
                doc.preamble.append(
                    NoEscape(rf"\usecolortheme{{{self._beamer_color_theme}}}")
                )

        doc.preamble.append(Command("usepackage", "amsmath"))
        doc.preamble.append(Command("usepackage", "amssymb"))
        doc.preamble.append(Command("usepackage", "booktabs"))
        doc.preamble.append(Command("usepackage", "longtable"))
        doc.preamble.append(Command("usepackage", "graphicx"))
        doc.preamble.append(Command("usepackage", "float"))
        doc.preamble.append(Command("usepackage", "caption"))
        doc.preamble.append(Command("usepackage", "hyperref"))

        for line in self._raw_preamble:
            doc.preamble.append(NoEscape(line))

        doc.preamble.append(Command("title", self.title))

        if self.subtitle and self.is_beamer:
            doc.preamble.append(Command("subtitle", self.subtitle))

        if self.author:
            doc.preamble.append(Command("author", self.author))

        if self.institute:
            doc.preamble.append(Command("institute", self.institute))

        doc.preamble.append(Command("date", NoEscape(self.date_tex)))

        if not self.is_beamer:
            doc.append(NoEscape(r"\setlength{\parindent}{0pt}"))
            doc.append(NoEscape(r"\setlength{\parskip}{0.6em}"))

        if self._make_title:
            if self.is_beamer:
                doc.append(NoEscape(r"\begin{frame}"))
                doc.append(NoEscape(r"\titlepage"))
                doc.append(NoEscape(r"\end{frame}"))
            else:
                doc.append(NoEscape(r"\maketitle"))

        if self.subtitle and not self.is_beamer:
            doc.append(
                NoEscape(
                    rf"\begin{{center}}\large {escape_latex(self.subtitle)}\end{{center}}"
                )
            )

        for line in self._raw_body:
            doc.append(NoEscape(line))

        if self._toc:
            if self.is_beamer:
                doc.append(NoEscape(r"\begin{frame}{Outline}"))
                doc.append(NoEscape(r"\tableofcontents"))
                doc.append(NoEscape(r"\end{frame}"))
            else:
                doc.append(NoEscape(r"\tableofcontents"))
                doc.append(NoEscape(r"\newpage"))

    # =========================================================================
    # Helpers
    # =========================================================================

    def _target(self) -> Any:
        if self._active_frame is not None:
            return self._active_frame
        return self.current

    def _append(self, obj: Any) -> None:
        target = self._target()
        if isinstance(target, list):
            target.append(obj)
        else:
            target.append(obj)

    def _txt(self, text: str = "") -> None:
        self.text_lines.append(str(text))

    def latex(self, tex: str) -> NoEscape:
        return NoEscape(tex)

    def math(self, latex: str) -> NoEscape:
        return NoEscape(rf"${latex}$")

    def ref(self, label: str) -> NoEscape:
        return NoEscape(rf"\ref{{{label}}}")

    def figref(self, label: str) -> NoEscape:
        return NoEscape(rf"Figure~\ref{{{label}}}")

    def tabref(self, label: str) -> NoEscape:
        return NoEscape(rf"Table~\ref{{{label}}}")

    def eqref(self, label: str) -> NoEscape:
        return NoEscape(rf"Eq.~\eqref{{{label}}}")

    def im(self, latex: str) -> NoEscape:
        return self.math(latex)

    def append(self, obj: Any) -> None:
        self._append(obj)

    def raw_tex(self, tex: str) -> None:
        self._append(NoEscape(tex))
        self._txt(f"[RAW TEX] {tex}")

    # =========================================================================
    # Structure
    # =========================================================================

    def section(self, title: str) -> None:
        if self.is_beamer:
            self.doc.append(NoEscape(rf"\section{{{escape_latex(title)}}}"))
            self.current = self.doc
            self.stack = [self.doc]
        else:
            sec = Section(title, numbering=False)
            self.doc.append(sec)
            self.current = sec
            self.stack = [self.doc, sec]

        self._txt("")
        self._txt(title)
        self._txt("=" * max(20, len(title)))

    def subsection(self, title: str) -> None:
        if self.is_beamer:
            self.doc.append(NoEscape(rf"\subsection{{{escape_latex(title)}}}"))
            self.current = self.doc
            self.stack = [self.doc]
        else:
            parent = self.stack[1] if len(self.stack) >= 2 else self.doc
            sub = Subsection(title, numbering=False)
            parent.append(sub)
            self.current = sub
            self.stack = [self.doc, parent, sub]

        self._txt("")
        self._txt(title)
        self._txt("-" * max(8, len(title)))

    def subsubsection(self, title: str) -> None:
        if self.is_beamer:
            self.doc.append(NoEscape(rf"\subsubsection{{{escape_latex(title)}}}"))
            self.current = self.doc
            self.stack = [self.doc]
        else:
            parent = self.stack[2] if len(self.stack) >= 3 else self.current
            subsub = Subsubsection(title, numbering=False)
            parent.append(subsub)
            self.current = subsub
            self.stack = [self.doc, *self.stack[1:3], subsub]

        self._txt("")
        self._txt(title)
        self._txt("~" * max(6, len(title)))

    # =========================================================================
    # Prose
    # =========================================================================

    def p(self, text: str = "") -> None:
        if text:
            self._append(escape_latex(str(text)))
        self._append(NoEscape(r"\par"))
        self._txt(str(text))
        self._txt("")

    def line(self, text: str = "") -> None:
        if text:
            self._append(escape_latex(str(text)))
            self._append(NoEscape(r"\\"))
        else:
            self._append(NoEscape(r"\par"))
        self._txt(str(text))

    def px(self, *parts: Any) -> None:
        txt_parts = []

        for part in parts:
            if isinstance(part, NoEscape):
                self._append(part)
                txt_parts.append(str(part))
            else:
                self._append(escape_latex(str(part)))
                txt_parts.append(str(part))

        self._append(NoEscape(r"\par"))
        self._txt("".join(txt_parts))
        self._txt("")

    def bullets(self, items: Iterable[str]) -> None:
        with self._target().create(Itemize()) as itemize:
            for item in items:
                itemize.add_item(str(item))

        for item in items:
            self._txt(f"- {item}")
        self._txt("")

    def numbered(self, items: Iterable[str]) -> None:
        with self._target().create(Enumerate()) as enum:
            for item in items:
                enum.add_item(str(item))

        for i, item in enumerate(items, start=1):
            self._txt(f"{i}. {item}")
        self._txt("")

    def hline(self) -> None:
        self._append(NoEscape(r"\hrule"))
        self._txt("-" * 60)

    def page_break(self) -> None:
        self._append(NoEscape(r"\newpage"))
        self._txt("")
        self._txt("[PAGE BREAK]")
        self._txt("")

    def vspace(self, amount: str = "1em") -> None:
        self._append(NoEscape(rf"\vspace{{{amount}}}"))
        self._txt("")

    # =========================================================================
    # Math
    # =========================================================================

    def eq(self, latex: str, label: str | None = None) -> None:
        if label:
            self._append(NoEscape(r"\begin{equation}"))
            self._append(NoEscape(latex))
            self._append(NoEscape(rf"\label{{{label}}}"))
            self._append(NoEscape(r"\end{equation}"))
        else:
            self._append(Math(data=[NoEscape(latex)], escape=False))

        self._txt(f"[MATH] {latex}")

    def align(self, *lines: str, label: str | None = None) -> None:
        if label:
            self._append(NoEscape(r"\begin{align}"))
        else:
            self._append(NoEscape(r"\begin{align*}"))

        for i, line in enumerate(lines):
            suffix = r" \\" if i < len(lines) - 1 else ""
            self._append(NoEscape(line + suffix))

        if label:
            self._append(NoEscape(rf"\label{{{label}}}"))
            self._append(NoEscape(r"\end{align}"))
        else:
            self._append(NoEscape(r"\end{align*}"))

        self._txt("[ALIGN]")
        for line in lines:
            self._txt(line)

    # =========================================================================
    # Tables
    # =========================================================================

    def _fmt_cell(self, val: Any, float_fmt: str = ".4f") -> Any:
        if isinstance(val, float):
            return format(val, float_fmt)
        if isinstance(val, NoEscape):
            return val
        return str(val)

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
        rows = [list(r) for r in rows]
        colspec = alignment or ("c" * len(headers))

        if longtable:
            lt = LongTable(colspec)
            lt.add_hline()
            lt.add_row(headers)
            lt.add_hline()
            lt.end_table_header()

            for row in rows:
                lt.add_row([self._fmt_cell(v, float_fmt) for v in row])
                lt.add_hline()

            self._append(lt)

            if caption:
                self._append(NoEscape(rf"\captionof{{table}}{{{escape_latex(caption)}}}"))
            if label:
                self._append(NoEscape(rf"\label{{{label}}}"))

        else:
            tab = Tabular(colspec)
            tab.add_hline()
            tab.add_row(headers)
            tab.add_hline()

            for row in rows:
                tab.add_row([self._fmt_cell(v, float_fmt) for v in row])
                tab.add_hline()

            if self.is_beamer:
                self._append(NoEscape(r"\centering"))
                self._append(tab)
                if caption:
                    self._append(NoEscape(rf"\captionof{{table}}{{{escape_latex(caption)}}}"))
                if label:
                    self._append(NoEscape(rf"\label{{{label}}}"))
            else:
                self._append(NoEscape(r"\begin{table}[H]"))
                self._append(NoEscape(r"\centering"))
                self._append(tab)
                if caption:
                    self._append(NoEscape(rf"\caption{{{escape_latex(caption)}}}"))
                if label:
                    self._append(NoEscape(rf"\label{{{label}}}"))
                self._append(NoEscape(r"\end{table}"))

        self._txt("[TABLE]")
        self._txt(" | ".join(headers))
        for row in rows:
            self._txt(" | ".join(str(self._fmt_cell(v, float_fmt)) for v in row))
        if caption:
            self._txt(f"Caption: {caption}")
        if label:
            self._txt(f"Label: {label}")
        self._txt("")

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

    # =========================================================================
    # Figures
    # =========================================================================

    def figure(
        self,
        path: str,
        caption: str | None = None,
        label: str | None = None,
        width: str | NoEscape = r"0.95\textwidth",
        height: str | NoEscape | None = None,
        position: str = "h",
    ) -> None:
        path = str(path)

        width_tex = str(width)
        height_tex = str(height) if height is not None else None

        if self.is_beamer:
            self._append(NoEscape(r"\centering"))

            if height_tex is not None:
                self._append(
                    NoEscape(
                        rf"\includegraphics[width={width_tex},height={height_tex},keepaspectratio]{{{path}}}"
                    )
                )
            else:
                self._append(NoEscape(rf"\includegraphics[width={width_tex}]{{{path}}}"))

            if caption:
                self._append(NoEscape(rf"\captionof{{figure}}{{{escape_latex(caption)}}}"))
            if label:
                self._append(NoEscape(rf"\label{{{label}}}"))

        else:
            with self._target().create(Figure(position=position)) as fig:
                if height_tex is not None:
                    fig.append(
                        NoEscape(
                            rf"\includegraphics[width={width_tex},height={height_tex},keepaspectratio]{{{path}}}"
                        )
                    )
                else:
                    fig.add_image(path, width=NoEscape(width_tex))

                if caption:
                    fig.add_caption(caption)
                if label:
                    fig.append(NoEscape(rf"\label{{{label}}}"))

        self._txt(f"[FIGURE] {path}")
        if caption:
            self._txt(f"Caption: {caption}")
        if label:
            self._txt(f"Label: {label}")
        self._txt("")

    # =========================================================================
    # Beamer slides
    # =========================================================================

    def slide(
        self,
        title: str,
        subtitle: str | None = None,
        fragile: bool = False,
    ) -> None:
        if not self.is_beamer:
            raise RuntimeError("slide() requires documentclass='beamer'")

        self.end_slide()

        self._active_frame = []
        self._active_frame_title = title
        self._active_frame_fragile = fragile

        if subtitle:
            self._active_frame.append(NoEscape(rf"\framesubtitle{{{escape_latex(subtitle)}}}"))

        self._txt("")
        self._txt(f"[SLIDE] {title}")
        self._txt("-" * max(8, len(title)))

    def end_slide(self) -> None:
        if self._active_frame is None:
            return

        fragile = "[fragile]" if self._active_frame_fragile else ""
        title = escape_latex(self._active_frame_title or "")

        self.doc.append(NoEscape(rf"\begin{{frame}}{fragile}{{{title}}}"))

        for obj in self._active_frame:
            self.doc.append(obj)

        if self._active_frame_fragile:
            self.doc.append(NoEscape("\\end{frame}\n"))
        else:
            self.doc.append(NoEscape(r"\end{frame}"))

        self._active_frame = None
        self._active_frame_title = None
        self._active_frame_fragile = False

    # =========================================================================
    # Save methods
    # =========================================================================

    def save_txt(self) -> str:
        path = f"{self.base_name}.txt"

        header = []
        header.append(self.title)
        if self.subtitle:
            header.append(self.subtitle)
        if self.author:
            header.append(self.author)
        header.append("=" * 80)
        header.append("")

        Path(path).write_text(
            "\n".join(header + self.text_lines).rstrip() + "\n",
            encoding="utf-8",
        )
        return path

    def save_tex(self) -> str:
        self.end_slide()

        for line in self._raw_tail:
            self.doc.append(NoEscape(line))

        self.doc.generate_tex(self.base_name)
        return f"{self.base_name}.tex"

    _AUX_EXTENSIONS = (
        ".aux",
        ".log",
        ".out",
        ".toc",
        ".nav",
        ".snm",
        ".vrb",
        ".synctex.gz",
        ".fls",
        ".fdb_latexmk",
        ".bbl",
        ".blg",
    )

    def _cleanup_aux(self) -> None:
        for ext in self._AUX_EXTENSIONS:
            p = Path(f"{self.base_name}{ext}")
            if p.exists():
                p.unlink()

    def save_pdf(self, runs: int = 2, clean: bool = True) -> str | None:
        tex_path = self.save_tex()

        engine = shutil.which("pdflatex")
        if engine is None:
            print("pdflatex not found; wrote .tex only")
            return None

        pdf_path = f"{self.base_name}.pdf"
        ok = True

        for _ in range(max(1, runs)):
            result = subprocess.run(
                [
                    engine,
                    "-interaction=nonstopmode",
                    tex_path,
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                ok = False
                print("pdflatex failed. Last output:")
                print(result.stdout[-3000:])
                break

        if ok and clean:
            self._cleanup_aux()

        return pdf_path if ok and Path(pdf_path).exists() else None

    def save_all(
        self,
        runs: int = 2,
        clean: bool = True,
    ) -> tuple[str, str, str | None]:
        txt = self.save_txt()
        tex = self.save_tex()
        pdf = self.save_pdf(runs=runs, clean=clean)
        return txt, tex, pdf