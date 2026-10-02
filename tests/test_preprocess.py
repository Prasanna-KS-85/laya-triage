import pytest

from laya_triage.preprocess import PreprocessConfig, preprocess


def body(text, **cfg):
    return preprocess("t", text, PreprocessConfig(**cfg) if cfg else None)["body"]


def fenced(n, fence="```", info="", close=True):
    lines = [fence + info, *(f"line {i}" for i in range(1, n + 1))]
    return "\n".join(lines + ([fence] if close else []))


# Step 1: HTML comments
def test_html_comment_removed_across_lines():
    assert body("a <!-- one\ntwo\nthree --> b") == "a  b"


def test_multiple_html_comments_removed_non_greedy():
    assert body("<!--x-->keep<!--y-->") == "keep"


def test_unterminated_html_comment_left_as_is():
    assert body("a <!-- never closed\nb") == "a <!-- never closed\nb"


# Step 2: fenced code blocks
def test_block_of_exactly_max_code_lines_is_kept():
    text = fenced(20)
    assert body(text) == text


def test_block_of_max_code_lines_plus_one_is_collapsed():
    out = body(fenced(21, info="python")).split("\n")
    assert out[0] == "```python"
    assert out[1:11] == [f"line {i}" for i in range(1, 11)]
    assert out[11] == "[... 6 lines omitted ...]"
    assert out[12:17] == [f"line {i}" for i in range(17, 22)]
    assert out[17] == "```"
    assert len(out) == 18


def test_tilde_fence_is_collapsed():
    out = body(fenced(30, fence="~~~")).split("\n")
    assert out[0] == "~~~" and out[-1] == "~~~"
    assert "[... 15 lines omitted ...]" in out


def test_unterminated_fence_runs_to_end_of_body():
    out = body("before\n" + fenced(25, close=False)).split("\n")
    assert out[:2] == ["before", "```"]
    assert out[12] == "[... 10 lines omitted ...]"
    assert out[-1] == "line 25"


def test_fence_closes_only_on_same_char_and_at_least_same_length():
    text = "````\n" + "\n".join(["```", "~~~"] + [f"x{i}" for i in range(30)]) + "\n````\nafter"
    out = body(text).split("\n")
    assert out[0] == "````" and out[1:3] == ["```", "~~~"]
    assert "[... 17 lines omitted ...]" in out
    assert out[-2:] == ["````", "after"]


def test_max_code_lines_is_configurable():
    out = body(fenced(16), max_code_lines=15).split("\n")
    assert "[... 1 lines omitted ...]" in out


def test_config_rejects_values_that_cannot_collapse():
    with pytest.raises(ValueError):
        PreprocessConfig(max_code_lines=14)
    with pytest.raises(ValueError):
        PreprocessConfig(max_body_chars=0)


# Step 3: images, then URLs
def test_image_markdown_replaced():
    assert body("see ![screen shot](https://x.io/a.png) here") == "see [image: screen shot] here"


def test_empty_image_alt():
    assert body("![](https://x.io/a.png)") == "[image: ]"


def test_bare_url_replaced_and_trailing_punctuation_consumed():
    assert body("go to https://example.com/a?b=1, then (see http://x.org/y).") == "go to [url] then (see [url]"


def test_markdown_link_target_replaced():
    assert body("read [the docs](https://example.com/docs) now") == "read [the docs]([url]) now"


# Step 4: whitespace
def test_trailing_spaces_stripped_and_blank_runs_collapsed():
    assert body("a  \t\n\n\n\n  b   \n\n\nc") == "a\n\n  b\n\nc"


def test_leading_and_trailing_blank_lines_removed():
    assert body("\n\n  \n  indented\n\n \n") == "  indented"


# Step 5: cap
def test_body_capped_at_max_body_chars():
    assert body("x" * 7000) == "x" * 6000
    assert body("y" * 50, max_body_chars=10) == "y" * 10


# Step 6: state
def test_state_keys_and_title_only_stripped():
    state = preprocess("  Title with https://x.io  \n", "body")
    assert list(state) == ["title", "body"]
    assert state["title"] == "Title with https://x.io"


def test_none_and_empty_body():
    assert preprocess("t", None) == {"title": "t", "body": ""}
    assert preprocess("t", "") == {"title": "t", "body": ""}


def test_crlf_and_cr_normalised():
    assert body("a\r\nb\rc\r\n\r\n\r\n\r\nd") == "a\nb\nc\n\nd"


def test_crlf_lines_count_correctly_in_code_blocks():
    out = body(fenced(21).replace("\n", "\r\n")).split("\n")
    assert "[... 6 lines omitted ...]" in out


def test_steps_apply_in_spec_order():
    # The comment hides a URL (step 1 before 3); the collapse marker survives step 4.
    text = "<!-- https://hidden.example -->\n" + fenced(21) + "\n\n\n\nhttps://shown.example"
    out = body(text)
    assert "hidden" not in out
    assert out.endswith("```\n\n[url]")
    assert "[... 6 lines omitted ...]" in out
