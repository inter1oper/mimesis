# Model transcripts

One directory per session. Expected layout:

    data/transcripts/<model>_panel<A|B>/answer.txt      polished answer, verbatim
    data/transcripts/<model>_panel<A|B>/reasoning.txt   raw reasoning trace, verbatim
    data/transcripts/<model>_panel<A|B>/meta.json       optional: model name, date, prompt used

Plain text preferred. `.md` and `.json` are fine — the parser will be written
against whatever format actually arrives, which is why it is not written yet.

Verbatim matters: `claims.schema.json` stores `char_span` offsets into the raw
answer text so the score reveals exactly the span that was labelled, with no
string matching at render time. Reflowing or cleaning the text after parsing
invalidates every span.
