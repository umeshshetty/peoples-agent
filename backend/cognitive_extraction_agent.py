"""
People's Agent - Cognitive Extraction Agent
Replaces basic flat extraction with cognitive modeling.

Key differences from basic extraction:
1. Accumulates understanding over time (not flat)
2. Tracks entity state changes (not static labels)
3. Creates episodic memories (specific events)
4. Builds semantic facts (distilled knowledge)
5. Detects narratives (ongoing storylines)
"""

from typing import List, Dict, Tuple, Optional
from langchain_core.messages import SystemMessage, HumanMessage
import os
import json
import re
from datetime import datetime
import uuid

# Import cognitive memory structures
from cognitive_memory import (
    CognitiveEntity, EpisodicMemory, SemanticFact, Narrative,
    extract_cognitive_entity, calculate_salience, detect_narrative_continuation,
    infer_narrative_arc_shift, ProjectState, PersonRelationshipState
)

# Import Claude for deep analysis
try:
    from claude_client import get_claude_llm, claude_analyze
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# Import local LLM
from langchain_ollama import ChatOllama

# Configuration
MODEL_NAME = os.getenv("OLLAMA_MODEL", "glm4")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm(temperature=0.3):
    return ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=temperature)


# ============================================================================
# Cognitive Entity Extraction - The Smart Way
# ============================================================================

COGNITIVE_EXTRACTION_PROMPT = """You are a cognitive memory system that extracts knowledge like a human brain.

For each entity mentioned, extract:
1. name: The entity name (resolve pronouns using context)
2. type: Person, Project, Topic, Organization, Tool, Goal, Place
3. new_facts: List of NEW facts learned about this entity from THIS thought
4. emotional_context: How does the user feel about this entity? (positive/negative/neutral)
5. state_signals: Any indicators of the entity's current state

Also extract:
- episodic_event: A brief description of WHAT HAPPENED in this thought (the event)
- emotional_intensity: How emotionally charged is this thought? (0.0-1.0)

Context from previous thoughts:
{context}

Return JSON:
{{
  "entities": [
    {{
      "name": "Alice",
      "type": "Person",
      "new_facts": ["Works in Marketing", "Good at presentations"],
      "emotional_context": "positive",
      "state_signals": ["trusted collaborator", "frequent interaction"]
    }}
  ],
  "episodic_event": "Had a productive meeting with Alice about Q1 budget",
  "emotional_intensity": 0.6
}}

Analyze this thought:
"""


async def extract_cognitive(
    thought: str,
    context: str = "",
    existing_entities: Dict[str, CognitiveEntity] = None
) -> Dict:
    """
    Extract entities with cognitive modeling.
    Returns enriched entities, episodic memory, and narrative signals.
    """
    existing_entities = existing_entities or {}
    
    # Use Claude for deep extraction if available, else GLM4
    if CLAUDE_AVAILABLE:
        llm = get_claude_llm(temperature=0.3)
    else:
        llm = get_llm()
    
    prompt = COGNITIVE_EXTRACTION_PROMPT.format(context=context or "No previous context.")
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=thought)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()
        
        # Parse JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            # Process entities with cognitive modeling
            cognitive_entities = []
            for entity_data in result.get("entities", []):
                name = entity_data.get("name", "")
                entity_type = entity_data.get("type", "Topic")
                
                # Get or create cognitive entity
                existing = existing_entities.get(name.lower())
                cognitive_entity = extract_cognitive_entity(
                    name=name,
                    entity_type=entity_type,
                    thought_content=thought,
                    existing_entity=existing
                )
                
                # Add new semantic facts
                for fact in entity_data.get("new_facts", []):
                    cognitive_entity.add_semantic_fact(fact)
                
                # Update emotional context
                emotional_context = entity_data.get("emotional_context", "neutral")
                if emotional_context == "positive":
                    cognitive_entity.emotional_weight = min(1.0, cognitive_entity.emotional_weight + 0.1)
                elif emotional_context == "negative":
                    cognitive_entity.emotional_weight = min(1.0, cognitive_entity.emotional_weight + 0.2)  # Negative more salient
                
                cognitive_entities.append(cognitive_entity)
            
            # Create episodic memory
            episode_id = str(uuid.uuid4())[:8]
            episodic_memory = EpisodicMemory(
                id=episode_id,
                event_description=result.get("episodic_event", thought[:100]),
                timestamp=datetime.now().isoformat(),
                entities_involved=[e.name for e in cognitive_entities],
                emotional_intensity=result.get("emotional_intensity", 0.5)
            )
            
            # Calculate salience
            salience = calculate_salience(
                thought, cognitive_entities, datetime.now().isoformat()
            )
            
            return {
                "entities": cognitive_entities,
                "episodic_memory": episodic_memory,
                "salience": salience,
                "raw_extraction": result
            }
            
    except Exception as e:
        print(f"Cognitive extraction error: {e}")
    
    # Fallback
    return {
        "entities": [],
        "episodic_memory": None,
        "salience": 0.5,
        "raw_extraction": {}
    }


# ============================================================================
# State Inference - Detect Entity State Changes
# ============================================================================

STATE_INFERENCE_PROMPT = """Analyze this thought for entity state changes.

Known entities and their current states:
{entities_state}

New thought: "{thought}"

For each entity mentioned, determine:
1. Has their state changed? (e.g., project went from "active" to "blocked")
2. What triggered the change?
3. New state name

Project states: idea, planning, active, blocked, stalled, completing, completed, abandoned
Person relationship states: stranger, mentioned, acquaintance, colleague, collaborator, trusted_ally, conflict

Return JSON:
{{
  "state_changes": [
    {{
      "entity_name": "Project X",
      "old_state": "active",
      "new_state": "blocked",
      "trigger": "API dependency not ready"
    }}
  ]
}}
"""


async def infer_state_changes(
    thought: str,
    entities: List[CognitiveEntity]
) -> List[Dict]:
    """
    Infer state changes for entities based on thought content.
    """
    if not entities:
        return []
    
    llm = get_llm()
    
    entities_state = "\n".join([
        f"- {e.name} ({e.entity_type}): {e.current_state}"
        for e in entities
    ])
    
    prompt = STATE_INFERENCE_PROMPT.format(
        entities_state=entities_state,
        thought=thought
    )
    
    messages = [HumanMessage(content=prompt)]
    
    try:
        response = await llm.ainvoke(messages)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            # Apply state changes to entities
            for change in result.get("state_changes", []):
                entity_name = change.get("entity_name", "")
                new_state = change.get("new_state", "")
                trigger = change.get("trigger", "")
                
                for entity in entities:
                    if entity.name.lower() == entity_name.lower():
                        entity.transition_state(new_state, trigger)
            
            return result.get("state_changes", [])
    except Exception as e:
        print(f"State inference error: {e}")
    
    return []


# ============================================================================
# Narrative Extraction - Detect Storylines
# ============================================================================

NARRATIVE_PROMPT = """Analyze if this thought is part of an ongoing narrative/storyline.

Active narratives:
{narratives}

New thought: "{thought}"

Determine:
1. Does this continue an existing narrative? Which one?
2. Is this the START of a new narrative (a new project, goal, or ongoing situation)?
3. Does this RESOLVE a narrative (completion, abandonment, resolution)?
4. What's the emotional arc shift? (beginning, rising_action, climax, falling_action, resolution)

Return JSON:
{{
  "continues_narrative": "narrative_id or null",
  "starts_new_narrative": {{
    "title": "My journey with...",
    "description": "Brief description",
    "entities_involved": ["entity names"]
  }} or null,
  "resolves_narrative": "narrative_id or null",
  "arc_stage": "rising_action"
}}
"""


async def extract_narrative_signals(
    thought: str,
    entities: List[CognitiveEntity],
    existing_narratives: List[Narrative]
) -> Dict:
    """
    Detect narrative signals - continuation, start, or resolution.
    """
    llm = get_llm()
    
    narratives_str = "\n".join([
        f"- {n.id}: {n.title} (stage: {n.arc_stage}, entities: {', '.join(n.entities_involved)})"
        for n in existing_narratives if n.is_active
    ]) or "No active narratives."
    
    prompt = NARRATIVE_PROMPT.format(
        narratives=narratives_str,
        thought=thought
    )
    
    messages = [HumanMessage(content=prompt)]
    
    try:
        response = await llm.ainvoke(messages)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            
            # Create new narrative if detected
            if result.get("starts_new_narrative"):
                new_narrative_data = result["starts_new_narrative"]
                new_narrative = Narrative(
                    id=str(uuid.uuid4())[:8],
                    title=new_narrative_data.get("title", "Untitled"),
                    description=new_narrative_data.get("description", ""),
                    entities_involved=new_narrative_data.get("entities_involved", []),
                    started=datetime.now().isoformat(),
                    last_updated=datetime.now().isoformat()
                )
                result["new_narrative"] = new_narrative
            
            return result
    except Exception as e:
        print(f"Narrative extraction error: {e}")
    
    return {}


# ============================================================================
# Combined Cognitive Pipeline
# ============================================================================

async def run_cognitive_extraction_pipeline(
    thought: str,
    context: str = "",
    existing_entities: Dict[str, CognitiveEntity] = None,
    existing_narratives: List[Narrative] = None
) -> Dict:
    """
    Run the full cognitive extraction pipeline.
    Returns:
    - Cognitive entities with accumulated knowledge
    - Episodic memory of the event
    - State changes detected
    - Narrative signals
    - Salience score
    """
    existing_entities = existing_entities or {}
    existing_narratives = existing_narratives or []
    
    # Step 1: Cognitive extraction
    extraction_result = await extract_cognitive(thought, context, existing_entities)
    entities = extraction_result.get("entities", [])
    episodic_memory = extraction_result.get("episodic_memory")
    salience = extraction_result.get("salience", 0.5)
    
    # Step 2: Infer state changes
    state_changes = await infer_state_changes(thought, entities)
    
    # Step 3: Narrative detection
    narrative_signals = await extract_narrative_signals(
        thought, entities, existing_narratives
    )
    
    return {
        "entities": entities,
        "episodic_memory": episodic_memory,
        "state_changes": state_changes,
        "narrative": narrative_signals,
        "salience": salience,
        "model_used": "claude" if CLAUDE_AVAILABLE else "glm4"
    }
