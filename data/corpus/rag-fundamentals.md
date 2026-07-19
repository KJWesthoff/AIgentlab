# Retrieval-Augmented Generation (RAG) fundamentals

Retrieval-augmented generation combines an information retrieval step with
a language model generation step. When a query arrives, the system first
retrieves relevant document chunks — typically using vector similarity
search over embeddings, keyword search such as BM25, or a hybrid of both —
and then passes those chunks to a language model as context so the model
can synthesize an answer grounded in the retrieved material.

A plain database lookup, by contrast, returns stored records that exactly
match a structured query. It performs no synthesis: the result is the data
itself, retrieved deterministically by key, index, or predicate. A database
lookup cannot answer questions whose phrasing does not match the stored
structure, and it cannot combine information from several records into a
new prose answer.

Key differences: RAG handles fuzzy natural-language queries, tolerates
paraphrasing, and produces synthesized prose, but its answers are
probabilistic and can contain errors even with good retrieval. A database
lookup is exact, fast, cheap, and auditable, but only works for structured
queries known in advance.

RAG systems introduce failure modes a database does not have: retrieval can
miss the relevant chunk, the model can ignore retrieved context, and the
model can hallucinate details not present in any retrieved document.
Grounding, citation, and evaluation of retrieval quality are therefore
essential parts of a production RAG system.
