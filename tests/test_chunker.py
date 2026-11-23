from src.llm.chunker import chunk_text


def test_short_text_single_chunk():
    assert chunk_text("hello") == ["hello"]


def test_long_text_splits():
    chunks = chunk_text("para\n\n" * 20000)
    assert len(chunks) > 1
    assert all(len(c) <= 6000 * 4 for c in chunks)
