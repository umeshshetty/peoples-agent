"""
People's Agent - Advanced PKM Agents
Multi-intent decomposition, Serendipity Engine, Daily Briefing, Auto-atomization.
"""

from typing import List, Dict, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import os
import json
import re
from datetime import datetime, timedelta

MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm(temperature: float = 0.3):
    return ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=temperature)


# ============================================================================
# Multi-Intent Dropzone Decomposition
# ============================================================================

async def decompose_intents(content: str) -> List[Dict]:
    """
    Decompose a brain dump into separate intents with PARA routing.
    
    Input: "Check Pinecone pricing for app, buy milk, learn about vector DBs"
    Output: [
        {"intent": "Check Pinecone pricing", "type": "TASK", "para": "PROJECT", "context": "app"},
        {"intent": "Buy milk", "type": "TASK", "para": "AREA", "context": "personal"},
        {"intent": "Learn about vector DBs", "type": "RESOURCE", "para": "RESOURCE", "context": "learning"}
    ]
    """
    llm = get_llm()
    
    prompt = f"""Decompose this brain dump into separate intents. Each intent should be atomic.

Input: "{content}"

For each intent, classify:
- type: TASK (actionable), NOTE (information), RESOURCE (reference), MEETING (calendar)
- para: PROJECT (has deadline), AREA (ongoing), RESOURCE (learning), ARCHIVE (done)
- priority: high/medium/low
- context: relevant project or area name

Return JSON array:
[
    {{
        "intent": "the atomic intent",
        "type": "TASK|NOTE|RESOURCE|MEETING",
        "para": "PROJECT|AREA|RESOURCE|ARCHIVE",
        "priority": "high|medium|low",
        "context": "project or area name"
    }}
]"""

    messages = [
        SystemMessage(content="You decompose brain dumps into atomic intents. Return valid JSON array."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"Intent decomposition error: {e}")
    
    # Fallback: treat as single intent
    return [{"intent": content, "type": "NOTE", "para": "RESOURCE", "priority": "medium", "context": "general"}]


# ============================================================================
# Serendipity Engine
# ============================================================================

async def find_serendipitous_connections(
    current_note: str,
    all_notes: List[Dict],
    vector_results: List[Dict],
    limit: int = 3
) -> List[Dict]:
    """
    Find unexpected but valuable connections.
    Uses semantic similarity + dissimilarity scoring for "structural holes".
    
    Returns notes that are:
    - Semantically related (not random)
    - Topically distant (unexpected)
    - Potentially valuable (has connections)
    """
    llm = get_llm()
    
    # Get notes that have SOME relevance but are from different categories
    candidates = []
    for note in vector_results:
        # Look for notes with medium distance (not too similar, not too different)
        distance = note.get("distance", 0)
        if 0.3 < distance < 0.8:  # Sweet spot for serendipity
            candidates.append(note)
    
    if not candidates:
        candidates = vector_results[:5]
    
    # Use LLM to find unexpected connections
    candidates_text = "\n".join([
        f"- [{c.get('id', '')}] {c.get('content', '')[:100]}..."
        for c in candidates[:10]
    ])
    
    prompt = f"""Find unexpected but valuable connections between these notes.

Current focus: "{current_note[:200]}"

Candidate notes:
{candidates_text}

Find notes that:
1. Are from a DIFFERENT topic/domain
2. Share an underlying pattern or principle
3. Could spark creative insight

Return JSON:
{{
    "connections": [
        {{
            "note_id": "id",
            "reason": "why this is surprisingly relevant",
            "pattern": "the underlying pattern shared"
        }}
    ]
}}"""

    messages = [
        SystemMessage(content="You find creative, unexpected connections between ideas."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result.get("connections", [])[:limit]
    except Exception as e:
        print(f"Serendipity error: {e}")
    
    return []


# ============================================================================
# Daily Briefing Agent
# ============================================================================

async def generate_daily_briefing(
    recent_thoughts: List[Dict],
    open_tasks: List[Dict],
    today_meetings: List[Dict],
    knowledge_graph
) -> Dict:
    """
    Generate a personalized daily briefing.
    
    Returns:
    {
        "greeting": "Good morning!",
        "yesterday_summary": "You worked on...",
        "open_questions": ["..."],
        "today_focus": ["..."],
        "relevant_notes": ["..."]
    }
    """
    llm = get_llm()
    
    # Build context
    recent_text = "\n".join([
        f"- [{t.get('timestamp', '')[:10]}] {t.get('content', '')[:100]}..."
        for t in recent_thoughts[:10]
    ])
    
    tasks_text = "\n".join([
        f"- {t.get('task', '')} (deadline: {t.get('deadline', 'none')})"
        for t in open_tasks[:5]
    ])
    
    meetings_text = "\n".join([
        f"- {m.get('title', '')} at {m.get('when', '')}"
        for m in today_meetings[:3]
    ])
    
    prompt = f"""Generate a helpful daily briefing based on this context.

Recent notes:
{recent_text or "No recent notes"}

Open tasks:
{tasks_text or "No open tasks"}

Today's meetings:
{meetings_text or "No meetings today"}

Current time: {datetime.now().strftime("%A, %B %d, %Y %H:%M")}

Generate a friendly, concise briefing in JSON:
{{
    "greeting": "personalized greeting based on time of day",
    "yesterday_summary": "brief summary of recent work",
    "open_questions": ["questions left in notes"],
    "today_focus": ["suggested focus areas"],
    "relevant_notes": ["notes relevant to today's work"]
}}"""

    messages = [
        SystemMessage(content="You are a helpful personal assistant generating daily briefings."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"Daily briefing error: {e}")
    
    # Fallback
    hour = datetime.now().hour
    greeting = "Good morning!" if hour < 12 else "Good afternoon!" if hour < 17 else "Good evening!"
    return {
        "greeting": greeting,
        "yesterday_summary": f"You have {len(recent_thoughts)} recent notes.",
        "open_questions": [],
        "today_focus": ["Review your recent notes"],
        "relevant_notes": []
    }


# ============================================================================
# Auto-Atomization
# ============================================================================

async def atomize_note(content: str) -> List[Dict]:
    """
    Split a long note into atomic chunks.
    Each chunk contains one distinct idea.
    
    Returns:
    [
        {"title": "atomic title", "content": "atomic content", "tags": []},
        ...
    ]
    """
    # Don't atomize short notes
    if len(content) < 300:
        return [{"title": content[:50], "content": content, "tags": [], "is_atomic": True}]
    
    llm = get_llm()
    
    prompt = f"""Split this long note into atomic chunks. Each chunk should contain ONE distinct idea.

Note:
"{content}"

Rules:
1. Each chunk should be self-contained
2. Preserve important context
3. Generate a short title for each
4. Add relevant tags

Return JSON array:
[
    {{
        "title": "short descriptive title",
        "content": "the atomic content",
        "tags": ["relevant", "tags"]
    }}
]"""

    messages = [
        SystemMessage(content="You split notes into atomic, self-contained chunks."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if json_match:
            chunks = json.loads(json_match.group())
            for chunk in chunks:
                chunk["is_atomic"] = True
            return chunks
    except Exception as e:
        print(f"Atomization error: {e}")
    
    # Fallback: return as single note
    return [{"title": content[:50], "content": content, "tags": [], "is_atomic": False}]


# ============================================================================
# Feynman Agent (Explain Simply)
# ============================================================================

async def feynman_challenge(topic: str, user_notes: List[str]) -> Dict:
    """
    Challenge user to explain a topic simply (Feynman Technique).
    
    Returns a question and evaluation criteria.
    """
    llm = get_llm()
    
    notes_context = "\n".join(user_notes[:3]) if user_notes else "No notes found"
    
    prompt = f"""You are helping someone learn using the Feynman Technique.

Topic: {topic}
Their notes on this topic:
{notes_context}

Generate a simple question that tests their understanding.
Act like a curious beginner who needs it explained simply.

Return JSON:
{{
    "question": "the simple question to ask",
    "key_concepts": ["concepts they should mention"],
    "follow_up": "a follow-up question if they explain well"
}}"""

    messages = [
        SystemMessage(content="You help users learn by asking them to explain things simply."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"Feynman error: {e}")
    
    return {
        "question": f"Can you explain {topic} to me like I'm 5 years old?",
        "key_concepts": [],
        "follow_up": "Why does that matter?"
    }
