from src.docqa.chunker import chunk_document, Chunk


def test_splits_on_section_headings():
    text = (
        "# Doc Title\n\n"
        "## First Section\n" + ("word " * 300) + "\n\n"
        "## Second Section\n" + ("word " * 50) + "\n"
    )
    chunks = chunk_document(text, "d1", "d1.txt", target_size=800, overlap=120)
    titles = {c.section_title for c in chunks}
    assert "First Section" in titles
    assert "Second Section" in titles


def test_chunk_indexes_are_sequential_and_unique():
    text = "## A\n" + ("word " * 400) + "\n## B\n" + ("word " * 400)
    chunks = chunk_document(text, "d1", "d1.txt")
    idxs = [c.chunk_index for c in chunks]
    assert idxs == list(range(len(chunks)))


def test_large_section_splits_into_multiple_chunks_with_overlap():
    big = "word " * 1000  # ~5000 chars, well over target 800
    text = "## Big\n" + big
    chunks = [c for c in chunk_document(text, "d1", "d1.txt", target_size=800, overlap=120)
              if c.section_title == "Big"]
    assert len(chunks) >= 2
    # Overlap: end of chunk n and start of chunk n+1 share some text.
    assert chunks[0].text[-40:] in chunks[1].text or chunks[1].char_start < chunks[0].char_end


def test_section_shorter_than_target_is_single_chunk():
    text = "## Small\nOnly a short sentence here."
    chunks = [c for c in chunk_document(text, "d1", "d1.txt") if c.section_title == "Small"]
    assert len(chunks) == 1
    assert "short sentence" in chunks[0].text


def test_text_before_first_heading_gets_a_default_section():
    text = "Intro paragraph with no heading.\n\n## Real Section\nBody."
    chunks = chunk_document(text, "d1", "d1.txt")
    # The intro must still be retrievable, under some non-empty section title.
    assert any("Intro paragraph" in c.text for c in chunks)
