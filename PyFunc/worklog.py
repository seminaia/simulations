from pathlib import Path
from pylatex import Document, Section, Subsection, Figure, NoEscape, Command
from pylatex.utils import escape_latex

def latex_safe_text(text: str) -> str:
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
        "²": "^2",
        "³": "^3",
        "₀": "_0",
        "₁": "_1",
        "₂": "_2",
        "₃": "_3",
        "√": "sqrt",
        "Π": "Pi",
        "π": "pi",
        "≤": "<=",
        "≥": ">=",
        "≠": "!=",
        "×": "x",
        }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

class WorkLog:


    def __init__(self, base_name: str, title: str):
        self.base_name = base_name
        self.title_text = title

        self.txt_lines = []
        self.tex_lines = []

        self._init_tex()

    def _init_tex(self):
        self.tex_lines.extend([
            r"\documentclass[12pt]{article}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{amsmath}",
            r"\usepackage{graphicx}",
            r"\usepackage{float}",
            "",
            rf"\title{{{escape_latex(self.title_text)}}}",
            r"\date{}",
            "",
            r"\begin{document}",
            r"\maketitle",
            ""
        ])

    def title(self, text: str):
        self.txt_lines.append(text)
        self.txt_lines.append("=" * 80)

        self.tex_lines.append(rf"\section*{{{escape_latex(text)}}}")
        self.tex_lines.append("")

    def section(self, text: str):
        self.txt_lines.append("")
        self.txt_lines.append(text)
        self.txt_lines.append("-" * 80)
        safe = escape_latex(latex_safe_text(text))
        self.tex_lines.append(rf"\section*{{{safe}}}")
        self.tex_lines.append("")

    def subsection(self, text: str):
        self.txt_lines.append("")
        self.txt_lines.append(text)
        self.txt_lines.append("-" * len(text))

        self.tex_lines.append(rf"\subsection*{{{escape_latex(text)}}}")
        self.tex_lines.append("")
        
    def text(self, text: str = ""):
        s = str(text)
        self.txt_lines.append(s)

        s_tex = latex_safe_text(s)
        if s_tex.strip():
            self.tex_lines.append(escape_latex(s_tex) + r"\\")
        else:
            self.tex_lines.append("")

    def math(self, latex: str):
        self.txt_lines.append(f"[MATH] {latex}")

        self.tex_lines.append(r"\[")
        self.tex_lines.append(latex)
        self.tex_lines.append(r"\]")
        self.tex_lines.append("")

    def figure(self, path: str, caption: str = ""):
        self.txt_lines.append(f"[FIGURE] {path}")
        if caption:
            self.txt_lines.append(f"Caption: {caption}")

        self.tex_lines.extend([
            r"\begin{figure}[H]",
            r"\centering",
            rf"\includegraphics[width=0.95\textwidth]{{{path}}}",
        ])
        if caption:
            self.tex_lines.append(rf"\caption{{{escape_latex(caption)}}}")
        self.tex_lines.extend([
            r"\end{figure}",
            ""
        ])

    def save_txt(self):
        txt_path = f"{self.base_name}.txt"
        Path(txt_path).write_text("\n".join(self.txt_lines) + "\n", encoding="utf-8")
        return txt_path

    def save_tex(self):
        tex_path = f"{self.base_name}.tex"
        tex_full = self.tex_lines + [r"\end{document}", ""]
        Path(tex_path).write_text("\n".join(tex_full), encoding="utf-8")
        return tex_path

    def save_pdf(self, clean_tex=False):
        doc = Document(self.base_name, geometry_options={"margin": "1in"})
        doc.preamble.append(Command("title", self.title_text))
        doc.preamble.append(Command("date", ""))
        doc.append(NoEscape(r"\maketitle"))

        # Reuse the generated LaTeX body
        body = "\n".join(self.tex_lines[self.tex_lines.index(r"\begin{document}") + 2:])
        doc.append(NoEscape(body))
        doc.generate_pdf(clean_tex=clean_tex)

    def save_all(self, clean_tex=False):
        txt_path = self.save_txt()
        tex_path = self.save_tex()
        import shutil
        import subprocess

        engine = shutil.which("pdflatex")
        if engine is not None:
            result = subprocess.run([engine, "-interaction=nonstopmode", f"{self.base_name}.tex"])
        if result.returncode != 0:
            print("pdflatex returned a nonzero exit code; check the .log file")
        else:
            print("pdflatex not found; wrote .txt and .tex only")

        return txt_path, tex_path, f"{self.base_name}.pdf"