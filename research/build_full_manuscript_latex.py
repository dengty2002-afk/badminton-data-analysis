from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research_outputs" / "shi_player_profile_v2" / "FULL_MANUSCRIPT_v2.md"
OUT_DIR = ROOT / "output" / "pdf" / "shi_player_profile_v2"
TEX = OUT_DIR / "shi_player_profile_v2.tex"


CITATIONS = [
    (r"Torres-Luque等人（2020）", r"\citet{TorresLuque2020}"),
    (r"Galeano等人（2021）", r"\citet{Galeano2021}"),
    (r"Valldecabres等人（2020）", r"\citet{Valldecabres2020}"),
    (r"Zhao等人（2025）", r"\citet{Zhao2025}"),
    (r"Hammes和Link（2024）", r"\citet{Hammes2024}"),
    (r"Aitchison（1982）", r"\citet{Aitchison1982}"),
    (r"Casals和Daunis-i-Estadella（2023）", r"\citet{Casals2023}"),
    (r"（Wang et al., 2023）", r"\citep{Wang2023}"),
    (r"(Wang et al., 2023)", r"\citep{Wang2023}"),
    (r"（Torres-Luque et al., 2020）", r"\citep{TorresLuque2020}"),
    (r"(Torres-Luque et al., 2020)", r"\citep{TorresLuque2020}"),
    (r"（Galeano et al., 2021）", r"\citep{Galeano2021}"),
    (r"(Galeano et al., 2021)", r"\citep{Galeano2021}"),
    (r"（Hammes & Link, 2024）", r"\citep{Hammes2024}"),
    (r"(Hammes & Link, 2024)", r"\citep{Hammes2024}"),
    (r"（Casals & Daunis-i-Estadella, 2023）", r"\citep{Casals2023}"),
    (r"(Casals & Daunis-i-Estadella, 2023)", r"\citep{Casals2023}"),
    (r"（Aitchison, 1982）", r"\citep{Aitchison1982}"),
    (r"(Aitchison, 1982)", r"\citep{Aitchison1982}"),
]


def escape_plain(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def inline(text: str) -> str:
    tokens: list[str] = []

    def hold(value: str) -> str:
        token = f"@@LATEX{len(tokens)}@@"
        tokens.append(value)
        return token

    for visible, command in CITATIONS:
        text = text.replace(visible, hold(command))

    def code_span(value: str) -> str:
        math_spans = {
            "M = 24": r"\(M=24\)",
            "d_m = x_{Shi,m} - x_{Opponent,m}": r"\(d_m=x_{\mathrm{Shi},m}-x_{\mathrm{Opponent},m}\)",
        }
        if value in math_spans:
            return math_spans[value]
        escaped = escape_plain(value).replace(r"\_", r"\_\allowbreak{}")
        return r"\texttt{" + escaped + "}"

    text = re.sub(r"`([^`]+)`", lambda m: hold(code_span(m.group(1))), text)
    text = re.sub(
        r"\*\*([^*]+)\*\*",
        lambda m: hold(r"\textbf{" + inline(m.group(1)) + "}"),
        text,
    )
    text = re.sub(
        r"(?<!\*)\*([^*]+)\*(?!\*)",
        lambda m: hold(r"\textit{" + inline(m.group(1)) + "}"),
        text,
    )
    text = escape_plain(text)
    for idx, value in enumerate(tokens):
        text = text.replace(f"@@LATEX{idx}@@", value)
    return text


def strip_heading_number(text: str) -> str:
    return re.sub(r"^\d+(?:\.\d+)*\s*", "", text).strip()


def table_widths(ncols: int) -> list[float]:
    if ncols == 2:
        return [0.28, 0.66]
    if ncols == 3:
        return [0.20, 0.39, 0.35]
    if ncols == 6:
        return [0.14, 0.17, 0.12, 0.22, 0.10, 0.13]
    return [0.94 / ncols] * ncols


def render_table(rows: list[list[str]], caption: str | None, index: int) -> str:
    ncols = len(rows[0])
    widths = table_widths(ncols)
    total = sum(widths)
    proportions = [w / total for w in widths]
    gap_count = (ncols - 1) * 2
    spec = "@{}" + "".join(
        f">{{\\raggedright\\arraybackslash}}p{{(\\linewidth - {gap_count}\\tabcolsep) * \\real{{{p:.4f}}}}}"
        for p in proportions
    ) + "@{}"
    default_caption = "复现资产与文件清单" if index >= 5 else f"数据表{index}"
    cap = re.sub(r"^表\s*\d+\s*", "", caption or default_caption).strip()
    output = [r"\begin{singlespace}", r"\small", rf"\begin{{longtable}}{{{spec}}}"]
    output.append(rf"\caption{{{inline(cap)}}}\label{{tab:{index}}}\\")
    output.append(r"\toprule")
    output.append(" & ".join(r"\textbf{" + inline(cell) + "}" for cell in rows[0]) + r" \\")
    output.append(r"\midrule")
    output.append(r"\endfirsthead")
    output.append(rf"\multicolumn{{{ncols}}}{{l}}{{\small\itshape 表\thetable（续）}}\\")
    output.append(r"\toprule")
    output.append(" & ".join(r"\textbf{" + inline(cell) + "}" for cell in rows[0]) + r" \\")
    output.append(r"\midrule")
    output.append(r"\endhead")
    output.append(r"\bottomrule")
    output.append(r"\endfoot")
    for row in rows[1:]:
        output.append(" & ".join(inline(cell) for cell in row) + r" \\")
    output.extend([r"\end{longtable}", r"\end{singlespace}"])
    return "\n".join(output)


PREAMBLE = r"""\documentclass[12pt,a4paper]{article}
\usepackage[margin=2.54cm,headheight=15pt]{geometry}
\usepackage{fontspec}
\usepackage{xeCJK}
\IfFontExistsTF{Times New Roman}{\setmainfont{Times New Roman}}{\setmainfont{TeX Gyre Termes}}
\IfFontExistsTF{SimSun}{\setCJKmainfont[AutoFakeBold=2.5,AutoFakeSlant=0.2]{SimSun}}{\setCJKmainfont{Microsoft YaHei}}
\IfFontExistsTF{Microsoft YaHei}{\setCJKsansfont{Microsoft YaHei}}{\setCJKsansfont{SimSun}}
\setmonofont{Courier New}
\IfFontExistsTF{SimSun}{\setCJKmonofont{SimSun}}{\setCJKmonofont{Microsoft YaHei}}
\usepackage{setspace}
\doublespacing
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs,longtable,array,calc}
\usepackage{float}
\usepackage{caption}
\usepackage{enumitem}
\usepackage[round,authoryear]{natbib}
\bibliographystyle{apalike}
\usepackage{xurl}
\usepackage{hyperref}
\usepackage{fancyhdr}
\usepackage{microtype}
\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue,pdfauthor={待补充},pdftitle={基于自动逐拍数据的石宇奇男子单打打法风格与技战术过程画像}}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small 石宇奇男子单打打法风格与技战术过程画像}
\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\setlength{\parindent}{2em}
\setlength{\parskip}{0pt}
\setlength{\emergencystretch}{2em}
\setlist{nosep,leftmargin=2.5em}
\captionsetup{font=small,labelfont=bf,labelsep=quad}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}
\renewcommand{\refname}{参考文献}
\DeclareRobustCommand{\doi}[1]{\href{https://doi.org/#1}{https://doi.org/#1}}
\graphicspath{{figures/}}
\begin{document}
\begin{titlepage}
\thispagestyle{empty}
\centering
\vspace*{2.2cm}
{\LARGE\bfseries 基于自动逐拍数据的石宇奇男子单打打法风格与技战术过程画像\par}
\vspace{0.5cm}
{\Large 24场比赛的回顾性个案研究\par}
\vspace{1.0cm}
{\large\itshape Playing Style and Tactical Process Profile of Shi Yuqi:\par}
{\large\itshape A 24-Match Stroke-Level Case Study\par}
\vfill
\begin{tabular}{rl}
作者： & 待补充 \\
单位： & 待补充 \\
通讯作者： & 待补充 \\
稿件类型： & 探索性回顾性个案研究 \\
引用格式： & APA第7版 \\
\end{tabular}
\vfill
{\large 2026年8月22日\par}
\end{titlepage}
"""


def section_between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def convert_body(text: str, table_start: int = 0, figure_start: int = 0) -> str:
    lines = text.splitlines()
    output: list[str] = []
    paragraph: list[str] = []
    pending_caption: str | None = None
    table_index = table_start
    figure_index = figure_start
    in_math = False
    in_list = False
    list_type = ""

    def flush_paragraph() -> None:
        nonlocal paragraph, pending_caption
        if not paragraph:
            return
        joined = " ".join(x.strip() for x in paragraph).strip()
        paragraph = []
        match = re.fullmatch(r"\*\*(表\s*\d+\s+.+)\*\*", joined)
        if match:
            pending_caption = match.group(1)
            return
        output.append(inline(joined) + "\n")

    def close_list() -> None:
        nonlocal in_list, list_type
        if in_list:
            output.append(rf"\end{{{list_type}}}")
            in_list = False
            list_type = ""

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if stripped == r"\[":
            flush_paragraph(); close_list(); in_math = True; output.append(r"\["); i += 1; continue
        if in_math:
            output.append(line)
            if stripped == r"\]": in_math = False
            i += 1; continue
        if stripped.startswith("|"):
            flush_paragraph(); close_list()
            raw_rows: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw_rows.append(lines[i].strip()); i += 1
            parsed = [[c.strip() for c in row.strip("|").split("|")] for row in raw_rows]
            parsed = [row for row in parsed if not all(re.fullmatch(r":?-{3,}:?", c) for c in row)]
            table_index += 1
            output.append(render_table(parsed, pending_caption, table_index))
            pending_caption = None
            continue
        image = re.fullmatch(r"!\[(.+)]\((.+)\)", stripped)
        if image:
            flush_paragraph(); close_list(); figure_index += 1
            caption = re.sub(r"^图\s*\d+\s*", "", image.group(1)).strip()
            filename = Path(image.group(2)).name
            output.extend([
                r"\begin{figure}[H]", r"\centering",
                rf"\includegraphics[width=0.92\textwidth]{{{filename}}}",
                rf"\caption{{{inline(caption)}}}\label{{fig:{figure_index}}}", r"\end{figure}",
            ])
            i += 1; continue
        if stripped.startswith("### "):
            flush_paragraph(); close_list()
            output.append(rf"\subsection{{{inline(strip_heading_number(stripped[4:]))}}}")
            i += 1; continue
        if stripped.startswith("## "):
            flush_paragraph(); close_list()
            title = stripped[3:].strip()
            if title.startswith("附录A"):
                output.append(r"\appendix")
                output.append(r"\section{复现资产与图表}")
            elif title in {"数据可得性声明", "伦理声明", "作者贡献（CRediT）", "利益冲突声明", "经费声明", "AI使用声明"}:
                output.append(rf"\section*{{{inline(title)}}}")
                output.append(rf"\addcontentsline{{toc}}{{section}}{{{inline(title)}}}")
            else:
                output.append(rf"\section{{{inline(strip_heading_number(title))}}}")
            i += 1; continue
        numbered = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        bullet = re.match(r"^-\s+(.+)$", stripped)
        if numbered or bullet:
            flush_paragraph()
            needed = "enumerate" if numbered else "itemize"
            if not in_list or list_type != needed:
                close_list(); output.append(rf"\begin{{{needed}}}"); in_list = True; list_type = needed
            output.append(r"\item " + inline((numbered or bullet).group(2 if numbered else 1)))
            i += 1; continue
        if not stripped or stripped == "---":
            flush_paragraph(); close_list(); i += 1; continue
        paragraph.append(stripped)
        i += 1
    flush_paragraph(); close_list()
    return "\n".join(output)


def main() -> None:
    md = SOURCE.read_text(encoding="utf-8")
    zh = section_between(md, "## 中文摘要", "## English Abstract")
    en = section_between(md, "## English Abstract", "## 1 引言")
    body = section_between(md, "## 1 引言", "## 参考文献")
    appendix = md.split("## 附录A：复现资产与图表", 1)[1].strip()

    zh_text, zh_keywords = zh.split("**关键词：**", 1)
    en_text, en_keywords = en.split("**Keywords:**", 1)

    parts = [PREAMBLE]
    parts.extend([
        r"\section*{中文摘要}", r"\addcontentsline{toc}{section}{中文摘要}",
        r"\begin{singlespace}", r"\noindent " + inline(zh_text.strip()),
        r"\par\vspace{0.5em}\noindent\textbf{关键词：}" + inline(zh_keywords.strip()), r"\end{singlespace}",
        r"\newpage", r"\section*{English Abstract}", r"\addcontentsline{toc}{section}{English Abstract}",
        r"\begin{singlespace}", r"\noindent " + inline(en_text.strip()),
        r"\par\vspace{0.5em}\noindent\textbf{Keywords: }" + inline(en_keywords.strip()), r"\end{singlespace}",
        r"\newpage", convert_body("## 1 引言\n" + body),
        r"\clearpage", r"\phantomsection", r"\addcontentsline{toc}{section}{参考文献}",
        r"\begin{singlespace}", r"\bibliography{references}", r"\end{singlespace}",
        r"\clearpage", convert_body("## 附录A：复现资产与图表\n" + appendix, table_start=4, figure_start=4),
        r"\end{document}",
    ])
    TEX.write_text("\n".join(parts), encoding="utf-8", newline="\n")
    print(TEX)


if __name__ == "__main__":
    main()
