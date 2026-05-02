"""Render the paper abstract with all \\newcommand macros expanded.

Produces two files:

  outputs/abstract_for_forms.tex  -- abstract with all \\macroName tokens
                                     replaced by their numeric values, but
                                     other LaTeX syntax preserved.  Paste
                                     this into a form that accepts LaTeX.

  outputs/abstract_plain.txt      -- the same, with LaTeX commands stripped
                                     (\\emph{X} -> X, $...$ kept verbatim
                                     because the math is mostly readable).
                                     Paste this into a plain-text form.

Reads:
  paper/sections/macros.tex
  paper/sections/abstract.tex

Run from repo root:
  uv run python scripts/render_abstract.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MACROS_PATH = ROOT / "paper" / "sections" / "macros.tex"
ABSTRACT_PATH = ROOT / "paper" / "sections" / "abstract.tex"
OUT_TEX = ROOT / "outputs" / "abstract_for_forms.tex"
OUT_TXT = ROOT / "outputs" / "abstract_plain.txt"


# Match \newcommand{\name}{value}.  Allows whitespace and comments above.
_NEWCOMMAND = re.compile(
    r"\\newcommand\{\\(?P<name>[A-Za-z]+)\}\{(?P<value>[^}]*)\}"
)


def load_macros() -> dict[str, str]:
    out = {}
    for m in _NEWCOMMAND.finditer(MACROS_PATH.read_text()):
        out[m.group("name")] = m.group("value")
    return out


def expand_macros(text: str, macros: dict[str, str]) -> str:
    """Replace \\macroName and \\macroName{} with their values.

    Two passes: first \\macroName{} (with empty braces, which TeX uses to
    terminate a control sequence before alphanumerics), then \\macroName.
    Order matters because \\churnMin{} would otherwise see \\churnMin
    followed by a literal {}.
    """
    # Sort by name length descending so longer names match first
    # (avoids \\bagFiveLow being shadowed by a hypothetical \\bag).
    names_by_len = sorted(macros.keys(), key=len, reverse=True)
    for name in names_by_len:
        # Form 1: \name{} -- empty-brace terminator.
        text = text.replace(f"\\{name}{{}}", macros[name])
        # Form 2: \name followed by non-letter (TeX's natural termination).
        # Use a lookahead so we don't eat the following character.
        pattern = re.compile(rf"\\{name}(?![A-Za-z])")
        text = pattern.sub(macros[name], text)
    return text


def strip_latex(text: str) -> str:
    """Crude LaTeX -> plain-text for forms that don't accept LaTeX."""
    text = re.sub(r"\\begin\{abstract\}", "", text)
    text = re.sub(r"\\end\{abstract\}", "", text)
    # \emph{X}, \textbf{X}, \textit{X} -> X
    text = re.sub(r"\\(?:emph|textbf|textit|texttt)\{([^}]*)\}", r"\1", text)
    # Math: $...$ -> the inside, with common LaTeX symbols translated.
    def _math(m: re.Match) -> str:
        s = m.group(1)
        s = s.replace(r"\text{--}", "-")
        s = s.replace(r"\sim", "~")
        s = s.replace(r"\%", "%")
        s = s.replace(r"\times", "x")
        s = s.replace(r"\lambda", "lambda")
        s = re.sub(r"\\([A-Za-z]+)", r"\1", s)  # any leftover \cmd -> cmd
        return s
    text = re.sub(r"\$([^$]*)\$", _math, text)
    # Em-dash placeholder.
    text = text.replace("---", "--")
    # Backslash percent in plain text.
    text = text.replace(r"\%", "%")
    text = text.replace(r"\,", " ")
    # Strip remaining single-letter command escapes.
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    # Collapse multi-blank lines and trim.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def main() -> None:
    macros = load_macros()
    abstract = ABSTRACT_PATH.read_text()
    expanded = expand_macros(abstract, macros)

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(expanded)
    plain = strip_latex(expanded)
    OUT_TXT.write_text(plain + "\n")

    print(f"Wrote {OUT_TEX}  ({len(expanded.splitlines())} lines)")
    print(f"Wrote {OUT_TXT}  ({len(plain.split())} words)")
    print()
    print("=== plain-text preview ===")
    print(plain)


if __name__ == "__main__":
    main()
