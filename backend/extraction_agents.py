"""
People's Agent - Extraction Agents
LLM-powered agents for extracting entities, categories, and relationships.
"""

from typing import List, Tuple, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge_graph import Entity, Category
import os
import json
import re

# Configuration
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.3:70b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm(temperature=0.3):
    """Get the Ollama LLM instance."""
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
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
            categories = []
            categories_data = json.loads(json_match.group())
            for c in categories_data:
                if isinstance(c, str):
                    categories.append(Category(name=c, confidence=0.8))
                elif isinstance(c, dict):
                    categories.append(Category(name=c.get("name", "Ideas"), confidence=float(c.get("confidence", 0.8))))
            return categories
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
# Reflection & Critique Agents (Cognitive Loop)
# ============================================================================

CRITIC_PROMPT = """Context from History:
{context}

Original Thought: {thought}

Draft Entities: {entities}

You are a data quality critic. Review the Draft Entities against the Thought and Context.
Did we miss any important entities mentioned in the context (like "he" referring to a specific person)?
Are the entity types correct?

If missing entities are found to exist in the context, explicitly list them.
If everything looks good, just say "Looks good."
"""

REFINER_PROMPT = """You are a Data Refiner.
I have a list of extracted entities and a critique from a reviewer.
Your job is to update the entity list to fix any issues mentioned in the critique.

Critique: {critique}

Current Entities: {entities}

INSTRUCTIONS:
1. If the critique adds new entities (e.g. "Add Susan Storm"), add them to the list.
2. If the critique says "Looks good", return the Current Entities unchanged.
3. Return ONLY the final JSON array of entities. Do not add explanation text.

Example Output:
[
  {{"name": "Susan Storm", "type": "Person", "description": "Flight Director"}}
]
"""

async def critique_extraction(thought: str, entities: List[Entity], context: str) -> str:
    """Review the extraction for quality and missed connections."""
    llm = get_llm(temperature=0.5)
    entities_json = json.dumps([e.to_dict() for e in entities])
    
    prompt_content = CRITIC_PROMPT.format(
        thought=thought,
        context=context,
        entities=entities_json
    )
    
    # Use HumanMessage for better local model adherence
    messages = [
        HumanMessage(content=prompt_content)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()
        return content
    except Exception as e:
        print(f"Critique error: {e}")
        return "Looks good."

async def refine_extraction(thought: str, entities: List[Entity], critique: str) -> List[Entity]:
    """Refine entities based on critique."""
    if critique.lower().startswith("looks good") or len(critique) < 5:
        return entities
        
    llm = get_llm(temperature=0.3)
    entities_json = json.dumps([e.to_dict() for e in entities])
    
    prompt_content = REFINER_PROMPT.format(
        critique=critique,
        entities=entities_json
    )
    
    messages = [
        HumanMessage(content=prompt_content)
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
        print(f"Refinement error: {e}")
    
    return entities


# ============================================================================
# Combined Extraction Pipeline
# ============================================================================

async def extract_all(thought: str, context: str = "") -> Tuple[List[Entity], List[Category], str]:
    """
    Run full extraction pipeline on a thought.
    Now accepts context for better extraction.
    
    Returns:
        Tuple of (entities, categories, summary)
    """
    # Run initial extraction
    entities = await extract_entities(thought)
    
    # Categories and summary don't usually need reflection, keep them fast
    categories = await classify_categories(thought)
    summary = await generate_summary(thought)
    
    return entities, categories, summary
