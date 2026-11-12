from src.preprocessing import chunk_texts, normalize_text


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  refund\n request  ") == "refund request"


def test_chunk_texts_respects_character_budget():
    assert all(len(chunk) <= 12 for chunk in chunk_texts(["one two three four"], max_chars=12))

