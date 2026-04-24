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