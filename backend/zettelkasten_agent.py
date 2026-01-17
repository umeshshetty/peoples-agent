"""
People's Agent - Zettelkasten Agent
Auto-atomizes long-form content into interconnected atomic notes.
"""

from typing import List, Dict, Tuple
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import json
import re
import uuid


# LLM for atomization
llm = ChatOllama(model="glm4", temperature=0.3, base_url="http://localhost:11434")


ATOMIZATION_PROMPT = """You are a Zettelkasten expert. Your job is to split long-form content into ATOMIC notes.

Rules for atomic notes:
1. Each note should contain ONE concept or idea (50-150 words max)
2. Notes should be self-contained but reference related concepts
3. Preserve the key insights from the original content
4. Create 3-7 atomic notes from the input

Input Content:
{content}

Return a JSON array of atomic notes:
[
    {{"title": "Brief title", "content": "Atomic note content", "related_to": ["title of related note"]}},
    ...
]

Return ONLY the JSON array, no explanation."""


def is_long_form(content: str) -> bool:
    """
    Detect if content is "long-form" and should be atomized.
    Criteria: >500 chars OR contains multiple paragraphs.
    """
    word_count = len(content.split())
    paragraph_count = content.count('\n\n') + 1
    
    return word_count > 100 or paragraph_count >= 3


def atomize_content(content: str) -> List[Dict]:
    """
    Split long-form content into atomic, interconnected notes.
    Returns list of atomic notes with relationships.
    """
    if not is_long_form(content):
        return []
    
    try:
        response = llm.invoke([
            SystemMessage(content=ATOMIZATION_PROMPT.format(content=content)),
            HumanMessage(content="Atomize this content.")
        ])
        
        # Parse JSON response
        response_text = response.content.strip()
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        
        if json_match:
            atoms = json.loads(json_match.group(0))
            
            # Assign unique IDs to each atom
            for atom in atoms:
                atom["id"] = f"atom_{uuid.uuid4().hex[:8]}"
            
            return atoms
        
        return []
        
    except Exception as e:
        print(f"Error atomizing content: {e}")
        return []


def create_atomic_thoughts(knowledge_graph, original_thought_id: str, atoms: List[Dict]) -> List[str]:
    """
    Create ThoughtNodes for each atom and link them together.
    
    Creates:
    - Individual ThoughtNode for each atom
    - ATOMIZED_FROM relationship to original thought
    - RELATED_TO relationships between atoms
    """
    from knowledge_graph import ThoughtNode, Entity
    from datetime import datetime
    
    created_ids = []
    title_to_id = {}
    
    # First pass: Create all atom nodes
    for atom in atoms:
        thought_node = ThoughtNode(
            id=atom["id"],
            content=atom.get("content", ""),
            summary=atom.get("title", atom.get("content", "")[:50]),
            timestamp=datetime.now().isoformat(),
            entities=[],  # Could extract entities from atoms too
            categories=[]
        )
        knowledge_graph.add_thought(thought_node)
        created_ids.append(atom["id"])
        title_to_id[atom.get("title", "")] = atom["id"]
    
    # Second pass: Create relationships
    with knowledge_graph.driver.session() as session:
        # Link atoms to original thought
        for atom_id in created_ids:
            session.run("""
                MATCH (original:Thought {id: $original_id})
                MATCH (atom:Thought {id: $atom_id})
                MERGE (atom)-[:ATOMIZED_FROM]->(original)
            """, {"original_id": original_thought_id, "atom_id": atom_id})
        
        # Link related atoms together
        for atom in atoms:
            atom_id = atom["id"]
            related_titles = atom.get("related_to", [])
            
            for related_title in related_titles:
                related_id = title_to_id.get(related_title)
                if related_id and related_id != atom_id:
                    session.run("""
                        MATCH (a:Thought {id: $atom_id})
                        MATCH (b:Thought {id: $related_id})
                        MERGE (a)-[:RELATED_TO]->(b)
                    """, {"atom_id": atom_id, "related_id": related_id})
    
    return created_ids


def should_atomize(content: str) -> bool:
    """Check if content should be atomized."""
    return is_long_form(content)
