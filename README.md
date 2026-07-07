# Document Q&A with Citations

A Retrieval-Augmented Generation (RAG) system over six AI-policy and
regulation documents. It answers natural-language questions with verifiable
citations, detects whether two documents conflict on a topic, works across
languages, and — importantly — refuses to answer when the documents don't
cover the question instead of hallucinating.

**Domain:** AI policy & regulation. The six documents deliberately span
binding law and voluntary guidance so they genuinely disagree on real
issues (mandatory vs. voluntary compliance, whether biometric surveillance
is banned or merely "managed"), which makes the contradiction feature
meaningful rather than a toy.

| doc_id | Document |
|---|---|
| `eu_ai_act` | EU AI Act (Regulation 2024/1689) — key articles excerpt (binding) |
| `nist_ai_rmf` | NIST AI Risk Management Framework 1.0 (voluntary, US) |
| `oecd_ai_principles` | OECD Recommendation on AI (non-binding, international) |
| `us_ai_bill_of_rights` | Blueprint for an AI Bill of Rights (non-binding, US) |
| `unesco_ai_ethics` | UNESCO Recommendation on the Ethics of AI (non-binding, global) |
| `g7_hiroshima_principles` | G7 Hiroshima Process Guiding Principles (voluntary, international) |

The documents in `data/documents/` are cleaned, faithful plain-text
excerpts/paraphrases of the real public texts, structured with section
headings so citations can reference a section, not just a byte offset.

---

## How to run it

Requires **Python 3.10+** (the code uses `str | None` union syntax).

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq API key
cp .env.example .env
# then edit .env and set GROQ_API_KEY=... (free tier: https://console.groq.com)

# 4. Ingest the documents (chunk -> embed -> store in ChromaDB)
python -m src.docqa.ingest
# Expected: per-document chunk counts and "Ingested 43 chunks from 6 documents."

# 5. Start the API (terminal 1)
uvicorn src.docqa.api:app --port 8000

# 6. Start the UI (terminal 2, same venv)
streamlit run ui/app.py
```

Then open the Streamlit URL it prints (usually http://localhost:8501). The
**Ask** tab takes a question in any language; the **Contradict** tab takes
two documents and a topic.

### Try it without the UI (curl)

```bash
# Answer with citations
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"Does the EU AI Act ban real-time remote biometric identification in public spaces?"}'

# Out-of-scope question -> covered:false, no invented answer
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What will the weather be in Paris tomorrow?"}'

# Multilingual: Spanish question -> Spanish answer
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"¿La Ley de IA de la UE prohíbe la identificación biométrica remota en tiempo real?"}'

# Contradiction between two documents on a topic
curl -s -X POST localhost:8000/contradict -H 'content-type: application/json' \
  -d '{"doc_id_1":"eu_ai_act","doc_id_2":"nist_ai_rmf","topic":"binding legal force and mandatory compliance"}'

# List document ids/titles
curl -s localhost:8000/documents
```

### Run the tests

```bash
pytest                       # all tests
pytest tests/test_chunker.py tests/test_citations.py   # offline unit tests only
```

The integration tests (`tests/test_rag_integration.py`) make live Groq calls;
they skip themselves automatically if `GROQ_API_KEY` is unset or the store
hasn't been ingested.

---

## API

| Endpoint | Body | Returns |
|---|---|---|
| `POST /ask` | `{"question": str}` | `{answer, covered, citations[], language}` |
| `POST /contradict` | `{"doc_id_1", "doc_id_2", "topic"}` | `{verdict, reasoning, doc_1_evidence[], doc_2_evidence[]}` |
| `GET /documents` | — | list of `{doc_id, source_file, title}` |
| `GET /health` | — | `{"status": "ok"}` |

Each **citation** contains `doc_id`, `source_file`, `section`, `chunk_index`,
and the exact `snippet` text used — so a reviewer can verify the answer
against the source without re-reading the whole document. A `/contradict`
`verdict` is one of `conflict`, `agree`, or `not_addressed_in_one_or_both`.

---

## Chunking strategy

The chunker (`src/docqa/chunker.py`) is **structure-aware and recursive**:

1. **Split on section headings first.** Each document uses Markdown `## Title`
   headings. The chunker splits on these so every chunk carries a meaningful
   `section_title` (e.g. "Prohibited AI Practices"). This is what makes a
   citation read like `eu_ai_act.txt — Prohibited AI Practices (chunk 3)`
   instead of an opaque offset. Any text before the first heading is kept
   under a `Preamble` section so nothing is dropped.
2. **Pack paragraphs up to a target size within a section.** Paragraphs are
   greedily packed into ~800-character blocks. A paragraph larger than the
   target is split on sentence boundaries; a runaway "sentence" with no
   punctuation is hard-split on characters as a last resort, so no chunk
   wildly overflows.
3. **Overlap between consecutive chunks of the same section.** Each chunk
   after the first is prefixed with the last ~120 characters of the previous
   chunk, so meaning isn't lost at a cut point (a definition and the sentence
   that relies on it don't get separated).

**Why ~800/120?** Policy text reasons in multi-sentence units. 800 characters
is large enough to keep a rule and its conditions together, small enough that
a retrieved chunk is a precise, quotable citation rather than a whole page.
120-character overlap (~15%) is enough to bridge a sentence split without
duplicating so much that retrieval returns near-identical neighbours.

Each chunk stores: `doc_id`, `source_file`, `section_title`, `chunk_index`
(sequential per document), and `char_start`/`char_end` offsets.

---

## Design decisions

- **Local embeddings, hosted generation.** Chunks are embedded with
  `sentence-transformers/all-MiniLM-L6-v2` running locally — no API key, no
  rate limits, deterministic, and free. Only the generation/translation/
  reasoning steps use the hosted Groq LLM (`llama-3.3-70b-versatile`). This
  keeps retrieval fast and reproducible and minimises paid API calls.
- **ChromaDB with cosine distance.** Embedded, persistent, zero external
  services to stand up — a reviewer just runs the ingest script. Cosine space
  is set explicitly so distances line up with the coverage threshold.
- **Multilingual via translate-at-the-boundary.** The corpus is English. A
  non-English query is detected (`langdetect`), translated to English (Groq)
  for retrieval and answering, and the final answer is translated back to the
  query language. Retrieval therefore always runs in one language, which is
  simpler and more accurate than a multilingual embedding space for a
  24-hour build. English queries skip the translation round-trip entirely.
- **No silent hallucination — three independent guards:**
  1. **Retrieval gate.** If no retrieved chunk is within the cosine-distance
     threshold (0.75), the LLM is never asked to answer; the API returns
     `covered: false` with an explicit message (translated to the query
     language). This is what makes "what's the weather tomorrow?" refuse.
  2. **Prompt lock.** When chunks *are* found, the model is instructed to use
     only the provided context and to set `answerable: false` if the context
     doesn't actually answer the question — because similarity finds *related*
     text, not necessarily *answering* text.
  3. **Citation validation.** Every `chunk_index` the model cites is checked
     against the set actually retrieved. Fabricated citations are dropped, and
     if nothing valid remains the response is downgraded to `covered: false`.
- **User-supplied topic for `/contradict`.** The caller provides the topic
  plus two doc ids. This retrieves the most relevant chunks *from each
  document* for that topic and asks the LLM to judge the relationship with
  quoted evidence. Auto-discovering the topic was considered but rejected: it
  tends to latch onto weak, incidental overlaps and produce unconvincing
  verdicts.
- **FastAPI + Streamlit split.** The endpoints are real HTTP endpoints
  (testable with curl/Postman), and the UI is a thin client over them — which
  matches the "endpoint" framing of the brief and keeps the UI logic-free.

---

## What's broken / unfinished

Honest list:

- **Retrieval quality on translated queries depends on translation fidelity.**
  The corpus is English-only, so a non-English query is only as good as its
  machine translation before it hits the vector store. A mistranslated key
  term can retrieve the wrong chunks.
- **The distance threshold (0.75) is hand-tuned, not learned.** It was chosen
  so obvious out-of-scope questions retrieve nothing while real questions
  clear the bar. A borderline-relevant question near the threshold could be
  refused (false negative) or, less often, let through thinly (the prompt
  lock and citation check are the backstop). There's no per-document or
  per-query calibration.
- **`char_start`/`char_end` are approximate.** They're located by searching
  for the chunk's leading text, which is robust enough for a citation anchor
  but can be off if a document repeats an identical opening fragment.
- **The contradiction verdict is a single LLM call.** It isn't cross-checked
  or run through a second judge, so a subtle conflict could be mislabelled
  `agree`, or vice-versa. It also only compares the top few chunks per
  document for the given topic, not the whole documents.
- **No PDF upload in the UI.** Documents are pre-ingested from
  `data/documents/*.txt` via the ingest script only. Adding a new document
  means dropping a `.txt` file in and re-running ingest.
- **No auth, no rate limiting, no streaming** on the API — it's a local demo.
- **Language detection can wobble on very short inputs.** `langdetect` is
  probabilistic; a two-word query might be misdetected, which would route the
  answer to the wrong language.

---

## What I'd build next

- **Hybrid retrieval** (BM25 keyword + vector) with a cross-encoder reranker,
  so exact legal terms ("Article 5", "35,000,000 EUR") aren't lost to purely
  semantic matching.
- **An evaluation harness** with a labelled set of questions (covered,
  out-of-scope, and known-contradiction pairs) to tune the distance threshold
  and measure answer/citation precision instead of eyeballing it.
- **Per-document / adaptive thresholds** and a confidence score surfaced in
  the response, rather than a single global cutoff.
- **A second-pass verifier** for `/contradict` (independent judge or
  self-consistency vote) to reduce single-call misclassification.
- **Multilingual embeddings** (e.g. a multilingual MiniLM) so non-English
  queries can retrieve directly without a translation hop, plus caching of
  translations.
- **Streaming answers** and citation highlighting in the UI (jump to the
  exact snippet in the source document).

---

## Project layout

```
data/documents/       6 source .txt files
src/docqa/
  config.py           paths, model ids, thresholds, document registry
  chunker.py          structure-aware chunking
  embeddings.py       local sentence-transformers wrapper
  store.py            ChromaDB add/query/list wrapper
  llm.py              Groq client: complete, complete_json, translate, detect
  rag.py              ask() and contradict() + citation validation
  ingest.py           one-time load -> chunk -> embed -> store CLI
  api.py              FastAPI endpoints
ui/app.py             Streamlit UI
tests/                chunker, citation, and integration tests
```

---

## AI Use Log

Per the assignment's honesty requirement, this is a complete and accurate
account of the AI usage on this project. **One tool was used — Claude Code —
but deliberately, through a structured workflow rather than ad-hoc prompting.**

| Tool | Models | Approx. usage | What it was used for |
|---|---|---|---|
| Claude Code (Anthropic CLI) | Claude Sonnet 5, then Claude Opus 4.8 | ~250k–330k tokens | The full build under my direction: requirements brainstorming, the design spec, the implementation plan, all source code and tests, the six policy-document excerpts, running the pipeline, debugging, prompt tuning, and this README. |

### How Claude Code was used (skills & plugins)

Rather than one-shotting the code, I drove Claude Code through the
**`superpowers`** plugin's workflow skills, in order:

1. **`superpowers:brainstorming`** — turned the brief into concrete decisions
   (domain, document set, vector store, embedding model, multilingual
   strategy, contradiction design) one question at a time, then produced a
   committed design spec in `docs/superpowers/specs/`.
2. **`superpowers:writing-plans`** — expanded the spec into a task-by-task
   implementation plan with exact file paths, interfaces, and TDD steps, saved
   to `docs/superpowers/plans/`.
3. **`superpowers:executing-plans`** — executed the plan task by task:
   writing failing tests first, then implementation, running the suite, and
   pausing at checkpoints for my review (test-driven development throughout).

I switched the underlying model mid-session (Sonnet 5 → **Opus 4.8**, via
Claude Code's `/model`) once the plan was set and heavier implementation
began. The design and every code decision were reviewed and directed by me;
Claude Code executed under that direction.

### Tools NOT used

No other AI assistants were used: no Cursor, Copilot, ChatGPT, Codex, Bolt,
v0, Gemini, or local models (e.g. Ollama). Retrieval embeddings run locally
via `sentence-transformers`; the only hosted model call in the running app is
Groq (for generation, contradiction reasoning, and translation), which is
separate from the AI assistance used to *build* the project and is documented
in the Design Decisions section above.

> Note: the token count is approximate — it reflects the working session
> that produced this repository and is rounded, not an exact meter reading.

### Self-assessment on token usage

For comparison: a senior engineer with a pre-decided spec (no back-and-forth
on chunking strategy, vector store, or multilingual approach) directing
Claude Code through the same scope would likely land closer to **~120k–180k
tokens** — roughly half to two-thirds of what this session used. The gap is
mostly the cost of using Claude Code as a genuine collaborator for
brainstorming trade-offs and iterating on this log, rather than a
code-typing tool against an already-fixed spec. I'd rather be upfront about
that than round the number down.
