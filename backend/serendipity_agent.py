"""
People's Agent - Serendipity Engine
Detects "Structural Holes" in the knowledge graph and generates serendipitous nudges.

Hybrid Model Approach:
- GLM4 (System 1): Quick nudge generation
- Claude (System 2): Deep remote analogies across domains
"""

from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import json
import re

# Import Claude for System 2 deep serendipity
try:
    from claude_client import claude_find_serendipity, get_claude_llm
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# GLM4 for quick System 1 nudges
llm = ChatOllama(model="glm4", temperature=0.7, base_url="http://localhost:11434")


def find_structural_holes(knowledge_graph, new_thought_entities: List[str], limit: int = 3) -> List[Dict]:
    """
    Find topics that share a common connection with the new thought's entities
    but are NOT directly mentioned in the thought.
    
    This reveals "Structural Holes" - potential connections the user might not see.
    
    Logic:
    1. Get entities mentioned in the new thought (A)
    2. Find thoughts that also mention A (call them T)
    3. Find other entities mentioned in T that are NOT in A (call them B)
    4. B entities are potentially related but disconnected - structural holes!
    """
    if not new_thought_entities:
        return []
    
    with knowledge_graph.driver.session() as session:
        result = session.run("""
            // Start from the new thought's entities
            UNWIND $entities as entity_name
            MATCH (a:Entity)<-[:MENTIONS]-(t:Thought)-[:MENTIONS]->(b:Entity)
            WHERE toLower(a.name) = toLower(entity_name)
              AND NOT toLower(b.name) IN $entities_lower
              AND a <> b
            
            // Group by the "hole" entity
            WITH b, a, count(t) as shared_thoughts, collect(DISTINCT t.summary)[0..2] as context_samples
            WHERE shared_thoughts >= 1
            
            RETURN b.name as disconnected_topic,
                   b.type as topic_type,
                   a.name as connected_via,
                   shared_thoughts,
                   context_samples
            ORDER BY shared_thoughts DESC
            LIMIT $limit
        """, {
            "entities": new_thought_entities,
            "entities_lower": [e.lower() for e in new_thought_entities],
            "limit": limit
        })
        
        holes = []
        for record in result:
            holes.append({
                "disconnected_topic": record["disconnected_topic"],
                "topic_type": record["topic_type"],
                "connected_via": record["connected_via"],
                "shared_thoughts": record["shared_thoughts"],
                "context_samples": record["context_samples"]
            })
        
        return holes


def generate_serendipity_nudge(new_topic: str, disconnected_topic: str, connected_via: str) -> str:
    """
    Generate a human-readable nudge suggesting a potential connection.
    """
    try:
        prompt = f"""You are a creative thinking assistant. The user just mentioned "{new_topic}".
In their knowledge graph, there's another topic "{disconnected_topic}" that is connected via "{connected_via}".
These two topics are NOT directly linked, but might have an interesting relationship.

Generate a SHORT, thought-provoking question (1 sentence) that nudges the user to explore this connection.
Example: "Could your work on 'X' benefit from insights in 'Y'?"

Return ONLY the nudge question, nothing else."""

        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        # Fallback to template if LLM fails
        return f"Is your idea about '{new_topic}' related to '{disconnected_topic}'? They share a connection through '{connected_via}'."


def get_serendipity_nudges(knowledge_graph, new_thought_entities: List[str]) -> List[Dict]:
    """
    Main entry point: Find structural holes and generate nudges.
    Returns list of nudges ready for display.
    """
    holes = find_structural_holes(knowledge_graph, new_thought_entities)
    
    nudges = []
    for hole in holes:
        # Pick the first new entity as the "new topic"
        new_topic = new_thought_entities[0] if new_thought_entities else "your idea"
        
        nudge_text = generate_serendipity_nudge(
            new_topic=new_topic,
            disconnected_topic=hole["disconnected_topic"],
            connected_via=hole["connected_via"]
        )
        
        nudges.append({
            "type": "serendipity",
            "message": nudge_text,
            "disconnected_topic": hole["disconnected_topic"],
            "connected_via": hole["connected_via"],
            "strength": hole["shared_thoughts"]
        })
    
    return nudges


async def get_deep_serendipity_insights(knowledge_graph, focus_topic: str, limit: int = 3) -> List[Dict]:
    """
    Use Claude (System 2) to find deep, remote analogies across domains.
    This is the "aha moment" generator - finding connections that surprise.
    """
    if not CLAUDE_AVAILABLE:
        # Fallback to basic serendipity
        return get_serendipity_nudges(knowledge_graph, [focus_topic])
    
    try:
        # Get distant notes from different clusters
        with knowledge_graph.driver.session() as session:
            result = session.run("""
                MATCH (t:Thought)
                WHERE NOT toLower(t.content) CONTAINS toLower($focus)
                RETURN t.content as content, t.summary as summary
                ORDER BY rand()
                LIMIT 8
            """, {"focus": focus_topic})
            
            distant_notes = [f"{r['summary']}: {r['content'][:200]}" for r in result]
        
        if not distant_notes:
            return []
        
        # Use Claude for deep analysis
        insights = await claude_find_serendipity(focus_topic, distant_notes)
        return insights
        
    except Exception as e:
        print(f"Error in deep serendipity: {e}")
        return get_serendipity_nudges(knowledge_graph, [focus_topic])
