"""Tests for the keyword-based business-aspect detector (src/aspects.py)."""

from src.aspects import ASPECT_KEYWORDS, aspect_names, find_aspect_mentions


def test_detects_app_and_support_in_one_message():
    text = "The mobile app crashes when I try to pay, but support resolved my issue quickly."
    assert aspect_names(text) == ["app / website", "customer support"]


def test_longest_keyword_wins_so_customer_support_is_one_mention():
    mentions = find_aspect_mentions("Customer support was great")
    assert [m["text"].lower() for m in mentions] == ["customer support"]
    assert mentions[0]["aspect"] == "customer support"


def test_whole_words_only_so_happy_does_not_trigger_app():
    assert aspect_names("I am happy and satisfied") == []


def test_offsets_point_at_the_matched_text():
    text = "Delivery was late"
    for mention in find_aspect_mentions(text):
        assert text[mention["start"]:mention["end"]] == mention["text"]


def test_aspect_names_are_unique_and_keep_first_appearance_order():
    assert aspect_names("price, then delivery, then the price again") == ["price", "delivery"]


def test_matching_is_case_insensitive():
    assert aspect_names("BATTERY dies fast") == ["battery"]


def test_every_aspect_has_keywords():
    assert len(ASPECT_KEYWORDS) >= 10
    assert all(words for words in ASPECT_KEYWORDS.values())


def test_non_string_input_does_not_crash():
    assert aspect_names(None) == []
