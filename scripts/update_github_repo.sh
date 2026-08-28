#!/bin/bash
# Update GitHub repo description and topics after pushing.
# Run from the project root after `git push`.

# Update repo description
gh repo edit --description "Multi-agent research system with orchestrator-worker pattern, contextual RAG retrieval, parallel subagents, and LLM-as-judge evaluation. Built on 250 AI research papers."

# Add topics
gh repo edit --add-topic "multi-agent-systems"
gh repo edit --add-topic "rag"
gh repo edit --add-topic "retrieval-augmented-generation"
gh repo edit --add-topic "langgraph"
gh repo edit --add-topic "openai"
gh repo edit --add-topic "qdrant"
gh repo edit --add-topic "llm"
gh repo edit --add-topic "research"
gh repo edit --add-topic "contextual-retrieval"
gh repo edit --add-topic "python"
gh repo edit --add-topic "fastapi"
gh repo edit --add-topic "multi-agent"
gh repo edit --add-topic "agentic-ai"

echo "Done! Repo description and topics updated."
