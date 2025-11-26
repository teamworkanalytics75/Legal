#!/usr/bin/env python3
"""
Interactive conversational retriever over the Harvard corpus using LangChain.

Features:
    * Conversational memory (buffer)
    * Retrieval from Chroma store (legal or general collection)
    * Local Phi-3 mini model via HuggingFace pipeline (GPU if available)
    * Each session logged into system memory
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import Chroma as LCChroma

from langchain_core.retrievers import BaseRetriever

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from system_memory import LawsuitMemoryManager

STORE_PATH = Path("chroma_collections")
MODEL_REGISTRY = {
    "harvard_legal": ("nlpaueb/legal-bert-base-uncased", "harvard_legal"),
    "harvard_general": ("sentence-transformers/all-mpnet-base-v2", "harvard_general"),
    "case_law_legal": ("nlpaueb/legal-bert-base-uncased", "case_law_legal"),
    "ma_motions_legal": ("nlpaueb/legal-bert-base-uncased", "ma_motions_legal"),
    "ma_motions_general": ("sentence-transformers/all-mpnet-base-v2", "ma_motions_general"),
}
DEFAULT_LLM_MODEL = "microsoft/Phi-3-mini-128k-instruct"


def load_llm(model_name: str = DEFAULT_LLM_MODEL) -> HuggingFacePipeline:
    device_available = torch.cuda.is_available()
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="models_cache")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="models_cache",
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    pipeline_kwargs = {
        "task": "text-generation",
        "model": model,
        "tokenizer": tokenizer,
        "max_new_tokens": 512,
        "temperature": 0.2,
        "do_sample": False,
    }
    if not device_available:
        pipeline_kwargs["device"] = -1
    text_gen = pipeline(**pipeline_kwargs)
    return HuggingFacePipeline(pipeline=text_gen)


def load_vectorstore(collection_key: str) -> LCChroma:
    """
    Load a single Chroma collection keyed by embedding model/dataset pair.
    Uses the registry defined above.
    """
    if collection_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown collection key: {collection_key}")

    embedding_model, collection_name = MODEL_REGISTRY[collection_key]
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=embedding_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    client = chromadb.PersistentClient(path=str(STORE_PATH))
    _ = client.get_collection(collection_name)  # ensure it exists
    return LCChroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_fn,
    )


class CombinedRetriever(BaseRetriever):
    """Simple weighted combination of multiple retrievers."""

    retrievers: List[BaseRetriever]
    weights: List[float]
    limit: int = 6

    def __init__(self, retrievers: List[BaseRetriever], weights: List[float], limit: int = 6) -> None:
        if len(retrievers) != len(weights):
            raise ValueError("Retrievers and weights must be the same length.")
        super().__init__(retrievers=retrievers, weights=weights, limit=limit)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[object]:
        scored_docs: List[Tuple[object, float]] = []

        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for idx, doc in enumerate(docs[: self.limit]):
                score = weight / (1 + idx)
                scored_docs.append((doc, score))

        scored_docs.sort(key=lambda item: item[1], reverse=True)

        merged: List[object] = []
        seen = set()
        for doc, score in scored_docs:
            key = (
                doc.metadata.get("dataset"),
                doc.metadata.get("document_id"),
                doc.metadata.get("start_word"),
            )
            if key in seen:
                continue
            doc.metadata["combined_score"] = score
            merged.append(doc)
            seen.add(key)
            if len(merged) >= self.limit:
                break
        return merged


def build_retriever(collection_key: str) -> object:
    """
    Build either a single retriever or an ensemble retriever depending on the key supplied.
    Recognised combos:
        - harvard_legal (default)                      -> single Harvard retriever
        - harvard_general                               -> single Harvard general model
        - case_law_legal                                -> single federal case-law retriever
        - ma_motions_legal / ma_motions_general         -> MA motion embeddings
        - combo_case_harvard                            -> ensemble: case law legal + harvard legal (weights 0.6/0.4)
    """
    if collection_key == "combo_case_harvard":
        case_store = load_vectorstore("case_law_legal")
        harvard_store = load_vectorstore("harvard_legal")
        return CombinedRetriever(
            retrievers=[
                case_store.as_retriever(search_type="mmr", search_kwargs={"k": 4}),
                harvard_store.as_retriever(search_type="mmr", search_kwargs={"k": 4}),
            ],
            weights=[0.6, 0.4],
            limit=8,
        )
    else:
        store = load_vectorstore(collection_key)
        return store.as_retriever(search_type="mmr", search_kwargs={"k": 6})


def initialize_memory(
    collection_key: str,
    *,
    use_persistent: bool,
    session_limit: int,
    turn_limit: int,
) -> tuple[ConversationBufferMemory, int]:
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    if not use_persistent:
        return memory, 0

    memory_manager = LawsuitMemoryManager()
    history_turns = memory_manager.fetch_conversation_history(
        limit_sessions=session_limit,
        max_turns=turn_limit,
        collection=collection_key,
    )
    for turn in history_turns:
        memory.chat_memory.add_user_message(turn["question"])
        memory.chat_memory.add_ai_message(turn["answer"])
    return memory, len(history_turns)


def interactive_session(
    collection_key: str,
    *,
    use_persistent_memory: bool,
    memory_session_limit: int,
    memory_turn_limit: int,
) -> None:
    llm = load_llm()
    retriever = build_retriever(collection_key)

    memory, seeded_turns = initialize_memory(
        collection_key,
        use_persistent=use_persistent_memory,
        session_limit=memory_session_limit,
        turn_limit=memory_turn_limit,
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False,
    )

    print("Conversational retriever ready. Type 'exit' to quit.")
    if seeded_turns:
        print(f"Loaded {seeded_turns} prior turns from persistent memory.")

    transcripts: List[dict] = []
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession terminated.")
            break

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        response = chain({"question": question})
        answer = response["answer"]
        print(f"\nAssistant: {answer}\n")

        transcripts.append(
            {
                "question": question,
                "answer": answer,
                "source_docs": [
                    {
                        "path": doc.metadata.get("relative_path"),
                        "start_word": doc.metadata.get("start_word"),
                        "end_word": doc.metadata.get("end_word"),
                    }
                    for doc in response.get("source_documents", [])
                ],
            }
        )

    if transcripts:
        memory_manager = LawsuitMemoryManager()
        memory_manager.log_event(
            event_type="langchain_conversation",
            summary=f"Conversational retrieval session ({collection_key}) with {len(transcripts)} turns.",
            metadata={"turns": len(transcripts), "collection": collection_key},
            artifacts={"transcripts": transcripts},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Conversational retriever over Harvard corpus.")
    parser.add_argument(
        "--collection",
        choices=list(MODEL_REGISTRY.keys()) + ["combo_case_harvard"],
        default="harvard_legal",
        help="Retrieval target or combo (default: harvard_legal; use combo_case_harvard to merge case law + Harvard).",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Optional single-shot question. When provided, runs one turn and exits.",
    )
    parser.add_argument(
        "--persistent-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Preload the chat buffer with prior logged conversations (default: enabled).",
    )
    parser.add_argument(
        "--memory-sessions",
        type=int,
        default=5,
        help="Maximum number of past sessions to inspect when seeding persistent memory.",
    )
    parser.add_argument(
        "--memory-turns",
        type=int,
        default=50,
        help="Maximum number of total turns to preload from persistent memory.",
    )
    args = parser.parse_args()
    if not STORE_PATH.exists():
        raise FileNotFoundError(f"Chroma store not found at {STORE_PATH}. Run build_vector_index.py first.")
    if args.question:
        llm = load_llm()
        retriever = build_retriever(args.collection)

        memory, _ = initialize_memory(
            args.collection,
            use_persistent=args.persistent_memory,
            session_limit=args.memory_sessions,
            turn_limit=args.memory_turns,
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=False,
        )
        response = chain({"question": args.question})
        answer = response["answer"]
        print(answer)
        memory_manager = LawsuitMemoryManager()
        memory_manager.log_event(
            event_type="langchain_conversation",
            summary="Single-turn retrieval query.",
            metadata={"turns": 1, "collection": args.collection},
            artifacts={
                "question": args.question,
                "answer": answer,
                "sources": [
                    {
                        "path": doc.metadata.get("relative_path"),
                        "start_word": doc.metadata.get("start_word"),
                        "end_word": doc.metadata.get("end_word"),
                    }
                    for doc in response.get("source_documents", [])
                ],
            },
        )
    else:
        interactive_session(
            args.collection,
            use_persistent_memory=args.persistent_memory,
            memory_session_limit=args.memory_sessions,
            memory_turn_limit=args.memory_turns,
        )


if __name__ == "__main__":
    main()
