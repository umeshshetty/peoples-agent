"""
People's Agent - Extraction Agents
LLM-powered agents for extracting entities, categories, and relationships.
"""

from typing import List, Tuple
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge_graph import Entity, Category
import os
import json
import re

# Configuration
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm():
    """Get the Ollama LLM instance."""
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.3,  # Lower temperature for more consistent extraction
    )


# ============================================================================
# Entity Extraction
# ============================================================================

ENTITY_EXTRACTION_PROMPT = """You are an entity extraction system. Extract key entities from the user's thought.

For each entity, provide:
- name: The entity name
- type: One of: Person, Place, Topic, Project, Concept, Organization, Tool, Skill
- description: Brief description (1 sentence)

Return ONLY a JSON array of entities. If no entities found, return [].

Example output:
[
  {"name": "Python", "type": "Skill", "description": "Programming language"},
  {"name": "John", "type": "Person", "description": "Colleague mentioned in context"}
]

Extract entities from this thought:"""


async def extract_entities(thought: str) -> List[Entity]:
    """Extract entities from a thought using LLM."""
    llm = get_llm()
    
    messages = [
        SystemMessage(content=ENTITY_EXTRACTION_PROMPT),
        HumanMessage(content=thought)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()
        
        # Try to parse JSON from response
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            entities_data = json.loads(json_match.group())
            return [
                Entity(
                    name=e.get("name", ""),
                    type=e.get("type", "Concept"),
                    description=e.get("description", "")
                )
                for e in entities_data
                if e.get("name")
            ]
    except Exception as e:
        print(f"Entity extraction error: {e}")
    
    return []


# ============================================================================
# Category Classification
# ============================================================================

CATEGORY_PROMPT = """Classify this thought into one or more categories.

Available categories:
- Work: Professional tasks, meetings, work projects
- Personal: Personal life, family, friends, hobbies
- Ideas: Creative ideas, brainstorming, possibilities
- Goals: Objectives, aspirations, plans for the future
- Tasks: Specific to-dos, action items
- Questions: Queries, things to research or learn
- Learning: Educational content, skills to develop
- Reflection: Introspection, feelings, personal insights

Return ONLY a JSON array of categories with confidence scores (0-1).
Example: [{"name": "Work", "confidence": 0.9}, {"name": "Tasks", "confidence": 0.7}]

Classify this thought:"""


async def classify_categories(thought: str) -> List[Category]:
    """Classify a thought into categories."""
    llm = get_llm()
    
    messages = [
        SystemMessage(content=CATEGORY_PROMPT),
        HumanMessage(content=thought)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()
        
        # Try to parse JSON from response
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            categories_data = json.loads(json_match.group())
            return [
                Category(
                    name=c.get("name", "Ideas"),
                    confidence=float(c.get("confidence", 0.5))
                )
                for c in categories_data
                if c.get("name")
            ]
    except Exception as e:
        print(f"Category classification error: {e}")
    
    # Default category
    return [Category(name="Ideas", confidence=0.5)]


# ============================================================================
# Summary Generation
# ============================================================================

SUMMARY_PROMPT = """Create a brief, one-sentence summary of this thought that captures its essence.
The summary should be concise (under 100 characters) and meaningful.
Return ONLY the summary, nothing else.

Thought:"""


async def generate_summary(thought: str) -> str:
    """Generate a brief summary of a thought."""
    llm = get_llm()
    
    messages = [
        SystemMessage(content=SUMMARY_PROMPT),
        HumanMessage(content=thought)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        summary = response.content.strip()
        # Limit summary length
        if len(summary) > 150:
            summary = summary[:147] + "..."
        return summary
    except Exception as e:
        print(f"Summary generation error: {e}")
        # Fallback: use first 100 chars of thought
        return thought[:97] + "..." if len(thought) > 100 else thought


# ============================================================================
# Relationship Detection
# ============================================================================

async def find_relationship_context(
    entities: List[Entity], 
    existing_context: str
) -> str:
    """Generate context about how new entities relate to existing ones."""
    if not entities or not existing_context:
        return ""
    
    llm = get_llm()
    
    entity_names = [e.name for e in entities]
    
    prompt = f"""Given these entities from a new thought: {entity_names}

And this context from previous related thoughts:
{existing_context}

Briefly describe any connections or patterns you notice (1-2 sentences).
If no clear connection, just say "New topic introduced."
"""
    
    messages = [
        SystemMessage(content="You identify connections between ideas."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"Relationship detection error: {e}")
        return ""


# ============================================================================
# Combined Extraction Pipeline
# ============================================================================

async def extract_all(thought: str) -> Tuple[List[Entity], List[Category], str]:
    """
    Run full extraction pipeline on a thought.
    
    Returns:
        Tuple of (entities, categories, summary)
    """
    # Run extractions in parallel would be ideal, but for simplicity run sequentially
    entities = await extract_entities(thought)
    categories = await classify_categories(thought)
    summary = await generate_summary(thought)
    
    return entities, categories, summary
