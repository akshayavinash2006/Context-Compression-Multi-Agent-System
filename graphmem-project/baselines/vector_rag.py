"""Vector RAG: top-k messages by cosine to the query, within the same token budget as ours.

Built on LangChain's InMemoryVectorStore with the shared embedding model (fair comparison).
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from baselines.base import MemoryBackend
from graphmem.embed import as_langchain_embeddings


class VectorRAG(MemoryBackend):
    name = "vector"

    def __init__(self, cfg, embedder, counter=None, top_k: int | None = None):
        super().__init__(cfg, counter)
        self.top_k = top_k or cfg.experiment.vector_rag_top_k
        self.store = InMemoryVectorStore(embedding=as_langchain_embeddings(embedder))

    def on_write(self, msg):
        self.store.add_documents([Document(page_content=msg["text"], metadata={"id": msg["id"]})],
                                 ids=[msg["id"]])

    def select(self, agent, query, token_budget):
        if not self.messages:
            return []
        hits = self.store.similarity_search_with_score(query, k=self.top_k)
        msgs = [self.by_id[d.metadata["id"]] for d, _ in hits]
        budget = token_budget or self.cfg.retrieval.token_budget_default
        return self.fit_budget(msgs, budget)
