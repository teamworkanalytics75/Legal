#!/usr/bin/env python3
"""
LangChain SQL Toolkit Prototype
===============================

This script demonstrates how LangChain's SQLDatabaseToolkit can query the
lawsuit SQLite database and reason over the institutional-knowledge question.
It mirrors the evidence retrieval our custom pipeline performs, but delegates
query planning/execution to a LangChain agent (ReAct-style).
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI


def build_agent(db_path: Path) -> AgentExecutor:
    """Create a LangChain SQL agent backed by gpt-4o-mini."""
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY environment variable is required for LangChain.")

    uri = f"sqlite:///{db_path.as_posix()}"
    database = SQLDatabase.from_uri(uri)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    toolkit = SQLDatabaseToolkit(db=database, llm=llm)

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        top_k=10,
    )
    return agent


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    lawsuit_db = Path(r"C:\Users\Owner\Desktop\LawsuitSQL\lawsuit.db")
    if not lawsuit_db.exists():
        raise FileNotFoundError(f"Lawsuit database not found: {lawsuit_db}")

    agent = build_agent(lawsuit_db)

    question = (
        "What is the likelihood that Harvard University as an institution "
        "(through its admissions office, alumni operations, or related channels) "
        "knew about the specific Xi Mingze slide before April 19, 2019? "
        "Use only the lawsuit database to support your reasoning. "
        "Retrieve relevant documents first, then summarise the evidence."
    )

    print("=" * 80)
    print("LANGCHAIN SQL TOOLKIT DEMO")
    print("=" * 80)
    result = agent.invoke({"input": question})

    print("\n=== FINAL ANSWER ===")
    print(result["output"])


if __name__ == "__main__":
    main()
