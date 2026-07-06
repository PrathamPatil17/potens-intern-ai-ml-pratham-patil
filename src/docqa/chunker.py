"""Structure-aware chunking.

Strategy: split the document into sections on Markdown H2 headings (`## Title`)
so each chunk carries a meaningful section title for citations. Within a
section, pack paragraphs up to a target character size, splitting oversized
paragraphs on sentence boundaries, and carry a small character overlap between
consecutive chunks of the same section so meaning isn't lost at a cut point.
"""
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
# Split a paragraph into sentences on ., !, ? followed by whitespace. Good
# enough for policy prose; we are not trying to be a full NLP sentence splitter.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    doc_id: str
    source_file: str
    section_title: str
    chunk_index: int
    char_start: int
    char_end: int
    text: str


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return (section_title, section_body) pairs.

    Text appearing before the first `## heading` is returned under the title
    "Preamble" so it stays retrievable. The document's top-level `# Title` line
    is not treated as a section heading.
    """
    matches = list(_HEADING_RE.finditer(text))
    sections: list[tuple[str, str]] = []
    if not matches:
        return [("Document", text.strip())]
    # Anything before the first ## heading.
    if matches[0].start() > 0:
        pre = text[: matches[0].start()].strip()
        # Drop a lone leading "# Title" line from the preamble body.
        pre = re.sub(r"^#\s+.*$", "", pre, count=1, flags=re.MULTILINE).strip()
        if pre:
            sections.append(("Preamble", pre))
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((title, body))
    return sections


def _pack(paragraphs: list[str], target_size: int) -> list[str]:
    """Greedily pack paragraphs into blocks up to target_size characters.

    A single paragraph longer than target_size is sentence-split first so no
    block wildly exceeds the target.
    """
    units: list[str] = []
    for p in paragraphs:
        if len(p) <= target_size:
            units.append(p)
            continue
        buf = ""
        for sent in _SENTENCE_RE.split(p):
            # A single "sentence" can still exceed target_size (e.g. a very long
            # run of text with no punctuation). Hard-split it on characters so no
            # unit ever wildly overflows the target.
            while len(sent) > target_size:
                if buf:
                    units.append(buf.strip())
                    buf = ""
                units.append(sent[:target_size].strip())
                sent = sent[target_size:]
            if buf and len(buf) + len(sent) + 1 > target_size:
                units.append(buf.strip())
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            units.append(buf.strip())

    blocks: list[str] = []
    buf = ""
    for u in units:
        if buf and len(buf) + len(u) + 2 > target_size:
            blocks.append(buf.strip())
            buf = u
        else:
            buf = f"{buf}\n\n{u}".strip()
    if buf:
        blocks.append(buf.strip())
    return blocks


def chunk_document(text: str, doc_id: str, source_file: str,
                   target_size: int = 800, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for title, body in _split_sections(text):
        if not body:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        blocks = _pack(paragraphs, target_size)
        prev_tail = ""
        for block in blocks:
            # Prepend the overlap tail from the previous block in this section.
            block_text = (prev_tail + " " + block).strip() if prev_tail else block
            # char_start/char_end locate the core block within the full doc so a
            # reviewer can find it; we search from the block's leading text to
            # stay robust to repeated fragments.
            start = text.find(block[:40])
            start = start if start != -1 else 0
            chunks.append(Chunk(
                doc_id=doc_id,
                source_file=source_file,
                section_title=title,
                chunk_index=idx,
                char_start=start,
                char_end=start + len(block),
                text=block_text,
            ))
            idx += 1
            prev_tail = block[-overlap:] if overlap and len(block) > overlap else block
    return chunks
