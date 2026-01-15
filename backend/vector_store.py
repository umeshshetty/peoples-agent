"""
People's Agent - Vector Store Module
Provides semantic search using ChromaDB and sentence transformers.
Enables "find similar notes" and context-aware retrieval.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from pathlib import Path
import os

# Configuration
DATA_DIR = Path(os.getenv("PEOPLES_AGENT_DATA", Path.home() / ".peoples_agent"))
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "thoughts"

# Initialize ChromaDB with persistent storage
_chroma_client = None
_collection = None


def get_chroma_client():
    """Get or create ChromaDB client with persistent storage."""
    global _chroma_client
    if _chroma_client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        print(f"✓ ChromaDB initialized at {CHROMA_DIR}")
    return _chroma_client


def get_collection():
    """Get or create the thoughts collection."""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        # Use built-in embedding function (all-MiniLM-L6-v2)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "User thoughts and notes for semantic search"}
        )
        print(f"✓ Collection '{COLLECTION_NAME}' ready with {_collection.count()} documents")
    return _collection


# ============================================================================
# Core Operations
# ============================================================================

def add_thought(thought_id: str, content: str, metadata: Dict = None) -> None:
    """
    Add a thought to the vector store.
    
    Args:
        thought_id: Unique identifier for the thought
        content: The thought content to embed
        metadata: Optional metadata (timestamp, entities, categories)
    """
    collection = get_collection()
    
    # Prepare metadata (ChromaDB requires flat dict with simple types)
    meta = {
        "timestamp": metadata.get("timestamp", "") if metadata else "",
        "summary": metadata.get("summary", content[:100]) if metadata else content[:100],
    }
    
    # Add entities and categories as comma-separated strings
    if metadata:
        if metadata.get("entities"):
            meta["entities"] = ",".join([e.get("name", "") for e in metadata["entities"]])
        if metadata.get("categories"):
            meta["categories"] = ",".join([c.get("name", "") for c in metadata["categories"]])
    
    try:
        # Upsert to handle updates
        collection.upsert(
            ids=[thought_id],
            documents=[content],
            metadatas=[meta]
        )
    except Exception as e:
        print(f"⚠ Error adding thought to vector store: {e}")


def semantic_search(query: str, limit: int = 5) -> List[Dict]:
    """
    Find thoughts semantically similar to the query.
    
    Args:
        query: Natural language query
        limit: Maximum results to return
        
    Returns:
        List of matching thoughts with similarity scores
    """
    collection = get_collection()
    
    if collection.count() == 0:
        return []
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(limit, collection.count())
        )
        
        # Format results
        formatted = []
        for i, doc_id in enumerate(results["ids"][0]):
            formatted.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results.get("distances") else 0
            })
        
        return formatted
    except Exception as e:
        print(f"⚠ Semantic search error: {e}")
        return []


def find_similar(thought_id: str, limit: int = 5) -> List[Dict]:
    """
    Find thoughts similar to a given thought.
    
    Args:
        thought_id: ID of the thought to find similar notes for
        limit: Maximum results
        
    Returns:
        List of similar thoughts
    """
    collection = get_collection()
    
    try:
        # Get the original thought
        result = collection.get(ids=[thought_id], include=["documents"])
        if not result["documents"]:
            return []
        
        original_content = result["documents"][0]
        
        # Search for similar (exclude self)
        similar = semantic_search(original_content, limit + 1)
        return [s for s in similar if s["id"] != thought_id][:limit]
    except Exception as e:
        print(f"⚠ Find similar error: {e}")
        return []


def get_context_for_query(query: str, limit: int = 3) -> str:
    """
    Get relevant context for answering a query.
    Returns formatted string for LLM context.
    """
    results = semantic_search(query, limit)
    
    if not results:
        return ""
    
    context_parts = ["Relevant notes found:"]
    for r in results:
        content = r["content"][:300] + "..." if len(r["content"]) > 300 else r["content"]
        context_parts.append(f"- {content}")
    
    return "\n".join(context_parts)


def get_stats() -> Dict:
    """Get vector store statistics."""
    collection = get_collection()
    return {
        "total_documents": collection.count(),
        "collection_name": COLLECTION_NAME
    }


# ============================================================================
# Batch Operations
# ============================================================================

def reindex_all(thoughts: List[Dict]) -> int:
    """
    Reindex all thoughts from the knowledge graph.
    Useful for initial setup or rebuilding the index.
    
    Args:
        thoughts: List of thought dictionaries
        
    Returns:
        Number of documents indexed
    """
    collection = get_collection()
    count = 0
    
    for thought in thoughts:
        try:
            add_thought(
                thought_id=thought.get("id", ""),
                content=thought.get("content", ""),
                metadata={
                    "timestamp": thought.get("timestamp", ""),
                    "summary": thought.get("summary", ""),
                    "entities": thought.get("entities", []),
                    "categories": thought.get("categories", [])
                }
            )
            count += 1
        except Exception as e:
            print(f"⚠ Error indexing thought {thought.get('id')}: {e}")
    
    return count


# Initialize on import
collection = get_collection()
