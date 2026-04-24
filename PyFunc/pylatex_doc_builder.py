from pathlib import Path
import shutil
import subprocess

from pylatex import Document, Section, Subsection, Subsubsection, Figure, Math, NoEscape, Command
from pylatex.utils import escape_latex


class PyLatexDocumentBuilder:
    def __init__(self, base_name, title, author=None):
        self.base_name = base_name
        self.title = title
        self.author = author

        self.doc = Document(base_name, geometry_options={"margin": "1in"})
        self.doc.preamble.append(Command("usepackage", "amsmath"))
        self.doc.preamble.append(Command("usepackage", "amssymb"))
        self.doc.preamble.append(Command("usepackage", "graphicx"))
        self.doc.preamble.append(Command("usepackage", "float"))
        self.doc.preamble.append(Command("usepackage", "hyperref"))

        self.doc.preamble.append(Command("title", title))
        if author:
            self.doc.preamble.append(Command("author", author))
        self.doc.preamble.append(Command("date", NoEscape(r"\today")))

        self.doc.append(NoEscape(r"\maketitle"))
        self.doc.append(NoEscape(r"\setlength{\parindent}{0pt}"))
        self.doc.append(NoEscape(r"\setlength{\parskip}{0.6em}"))

        self.current = self.doc

    def section(self, title):
        sec = Section(title, numbering=False)
        self.doc.append(sec)
        self.current = sec

    def subsection(self, title):
        sub = Subsection(title, numbering=False)
        self.current.append(sub)
        self.current = sub

    def p(self, text):
        self.current.append(escape_latex(str(text)))
        self.current.append(NoEscape(r"\par"))

    def rawp(self, tex):
        self.current.append(NoEscape(tex))
        self.current.append(NoEscape(r"\par"))

    def eq(self, latex):
        self.current.append(Math(data=[NoEscape(latex)], escape=False))

    def figure(self, path, caption=None, label=None, width=r"0.95\textwidth"):
        with self.current.create(Figure(position="H")) as fig:
            fig.add_image(path, width=NoEscape(width))
            if caption:
                fig.add_caption(caption)
            if label:
                fig.append(NoEscape(rf"\label{{{label}}}"))

    def ref(self, label):
        return NoEscape(rf"\ref{{{label}}}")

    _AUX_EXTENSIONS = (
        ".aux", ".nav", ".snm", ".toc",
        ".vrb", ".synctex.gz", ".fls", ".fdb_latexmk", ".bbl", ".blg",
    )

    def _cleanup_aux(self) -> None:
        """Remove auxiliary files produced by pdflatex / biber."""
        for ext in self._AUX_EXTENSIONS:
            p = Path(f"{self.base_name}{ext}")
            if p.exists():
                p.unlink()

    def save_beamer_pdf(self, runs: int = 1, clean: bool = True) -> str | None:
        """Generate a Beamer PDF presentation.

        Writes the .tex first, then compiles with pdflatex.

        Args:
            runs: Number of pdflatex passes (useful for TOC / references).
            clean: Remove auxiliary files (.aux, .log, .nav, etc.) after
                a successful build.  Defaults to True.

        Returns:
            Path to the generated PDF, or *None* if compilation fails.
        """
        tex_path = self.save_beamer_tex()
        engine = shutil.which("pdflatex")
        if engine is None:
            print("pdflatex not found; wrote .tex only")
            return None

        output_dir = str(Path(tex_path).parent)
        pdf_path = f"{self.base_name}.pdf"
        ok = True
        for _ in range(max(1, runs)):
            result = subprocess.run(
                [engine, "-interaction=nonstopmode",
                 f"-output-directory={output_dir}", tex_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                ok = False
                print("pdflatex failed; check the .log file")
                print(result.stdout[-3000:])
                break
        if ok and clean:
            self._cleanup_aux()
        return pdf_path if ok and Path(pdf_path).exists() else None
    def save_tex(self):
        self.doc.generate_tex(self.base_name)
        return f"{self.base_name}.tex"

    def save_pdf(self, runs=2):
        tex_path = self.save_tex()
        engine = shutil.which("pdflatex")
        if engine is None:
            print("pdflatex not found; wrote .tex only")
            return None

        for _ in range(runs):
            subprocess.run(
                [engine, "-interaction=nonstopmode", tex_path],
                check=False,
                capture_output=True,
                text=True,
            )

        pdf_path = f"{self.base_name}.pdf"
        return pdf_path if Path(pdf_path).exists() else None

    def save_all(self, runs: int = 1, beamer: bool = False) -> tuple[str, str, str | None]:
        """Save .txt, .tex, and .pdf versions of the document.

        When *beamer* is True the LaTeX / PDF output uses Beamer
        (slide-deck format) instead of article format.

        Args:
            runs (int, optional): The number of times to run pdflatex to resolve
                references when generating the PDF. Defaults to 1.
            beamer (bool, optional): If *True* produce a Beamer presentation
                instead of an article. Defaults to False.

        Returns:
            tuple[str, str, str | None]: Paths to the .txt, .tex, and .pdf
                files respectively. The PDF path will be None if generation
                failed or pdflatex was not found.
        """
        txt = self.save_txt()
        if beamer:
            tex = self.save_beamer_tex()
            pdf = self.save_beamer_pdf(runs=runs)
        else:
            tex = self.save_tex()
            pdf = self.save_pdf(runs=runs)
        return txt, tex, pdf