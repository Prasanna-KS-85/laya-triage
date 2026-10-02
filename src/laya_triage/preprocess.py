"""Issue text cleaning shared by training, evaluation and runtime (PROJECT_SPEC.md §8.4).

`preprocess(title, body)` applies §8.4 steps 1-6 in order. Interpretations of the spec text:

- Line endings: ``\\r\\n`` and ``\\r`` become ``\\n`` before step 1.
- Step 1: HTML comments ``<!-- ... -->`` are removed, matching across lines (non-greedy). An
  unterminated ``<!--`` is left as is.
- Step 2: "fenced code blocks and logs" means fenced blocks only, opened by a line starting with
  up to 3 spaces and 3+ backticks or tildes, and closed by a line of at least as many of the same
  character. An unterminated fence runs to the end of the body. A block with more than
  ``max_code_lines`` content lines keeps its first 10 and last 5 lines with a
  ``[... N lines omitted ...]`` line in between (N = lines - 15). Fence lines are kept.
- Step 3: image markdown ``![alt](url)`` becomes ``[image: alt]`` (``[image: ]`` for an empty
  alt) before URLs are replaced. A URL that is a markdown link target (right after ``](``) becomes
  ``[url]`` up to the closing ``)``, so ``[text](https://x)`` becomes ``[text]([url])``. Any other
  ``https?://\\S+`` becomes ``[url]``, so trailing punctuation is consumed.
- Step 4: trailing whitespace is stripped from every line, runs of 3+ newlines collapse to 2, and
  leading/trailing blank lines of the body are removed.
- Step 5: the body is cut to ``max_body_chars`` characters.
- Step 6: the title is only ``.strip()``-ed; a ``None`` body becomes ``""``.
"""

import re
from dataclasses import dataclass

CODE_HEAD_LINES = 10
CODE_TAIL_LINES = 5

_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_TARGET_URL = re.compile(r"(?<=\]\()https?://[^\s)]+")
_BARE_URL = re.compile(r"https?://\S+")
_BLANK_RUN = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class PreprocessConfig:
    max_code_lines: int = 20
    max_body_chars: int = 6000

    def __post_init__(self):
        if self.max_code_lines < CODE_HEAD_LINES + CODE_TAIL_LINES:
            raise ValueError(
                f"max_code_lines must be >= {CODE_HEAD_LINES + CODE_TAIL_LINES}, got {self.max_code_lines}"
            )
        if self.max_body_chars < 1:
            raise ValueError(f"max_body_chars must be >= 1, got {self.max_body_chars}")


def normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def strip_html_comments(text: str) -> str:
    return _COMMENT.sub("", text)


def collapse_code_blocks(text: str, max_code_lines: int) -> str:
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        m = _FENCE_OPEN.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        fence = m.group(1)
        close = re.compile(rf"^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$")
        j = i + 1
        while j < len(lines) and not close.match(lines[j]):
            j += 1
        content = lines[i + 1 : j]
        if len(content) > max_code_lines:
            omitted = len(content) - CODE_HEAD_LINES - CODE_TAIL_LINES
            content = [
                *content[:CODE_HEAD_LINES],
                f"[... {omitted} lines omitted ...]",
                *content[-CODE_TAIL_LINES:],
            ]
        out.append(lines[i])
        out.extend(content)
        if j < len(lines):
            out.append(lines[j])
        i = j + 1
    return "\n".join(out)


def replace_images(text: str) -> str:
    return _IMAGE.sub(lambda m: f"[image: {m.group(1)}]", text)


def replace_urls(text: str) -> str:
    return _BARE_URL.sub("[url]", _LINK_TARGET_URL.sub("[url]", text))


def normalise_whitespace(text: str) -> str:
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANK_RUN.sub("\n\n", text).strip("\n")


def preprocess(title: str, body: str | None, cfg: PreprocessConfig | None = None) -> dict:
    cfg = cfg or PreprocessConfig()
    text = normalise_newlines(body or "")
    text = strip_html_comments(text)
    text = collapse_code_blocks(text, cfg.max_code_lines)
    text = replace_urls(replace_images(text))
    text = normalise_whitespace(text)
    text = text[: cfg.max_body_chars]
    return {"title": title.strip(), "body": text}
