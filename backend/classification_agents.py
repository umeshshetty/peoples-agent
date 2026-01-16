"""
People's Agent - Classification and Task Agents
PARA categorization and task/deadline extraction from natural language.
"""

from typing import Dict, List, Optional, Tuple
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import os
import json
import re
from datetime import datetime, timedelta

# Configuration
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.3:70b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm(temperature: float = 0.2):
    """Get LLM for classification - low temp for consistency."""
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )


# ============================================================================
# PARA Classification Agent
# ============================================================================

PARA_DEFINITIONS = """
PARA Classification:
- PROJECT: Has a specific goal and deadline. Action-oriented. "Launch app by Friday", "Plan vacation"
- AREA: Ongoing responsibility, no end date. "Health", "Finances", "Career development" 
- RESOURCE: Reference material, learning. "Python best practices", "AI research notes"
- ARCHIVE: Completed or inactive items. Historical records.
"""


async def classify_para(content: str) -> Dict:
    """
    Classify a thought into PARA categories.
    
    Returns:
        {
            "classification": "PROJECT" | "AREA" | "RESOURCE" | "ARCHIVE",
            "confidence": 0.0-1.0,
            "reasoning": "why this classification"
        }
    """
    llm = get_llm()
    
    prompt = f"""{PARA_DEFINITIONS}

Classify this note:
"{content}"

Return JSON only:
{{
    "classification": "PROJECT" | "AREA" | "RESOURCE" | "ARCHIVE",
    "confidence": 0.0-1.0,
    "reasoning": "brief reason"
}}"""

    messages = [
        SystemMessage(content="You classify notes into PARA categories. Return valid JSON only."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        json_match = re.search(r'\{[^{}]*\}', response.content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
    except Exception as e:
        print(f"PARA classification error: {e}")
    
    # Fallback heuristics
    content_lower = content.lower()
    if any(kw in content_lower for kw in ['deadline', 'by friday', 'tomorrow', 'next week', 'launch', 'complete', 'finish']):
        return {"classification": "PROJECT", "confidence": 0.6, "reasoning": "Contains deadline/action words"}
    elif any(kw in content_lower for kw in ['learning', 'notes on', 'how to', 'tutorial', 'guide', 'research']):
        return {"classification": "RESOURCE", "confidence": 0.6, "reasoning": "Learning/reference content"}
    elif any(kw in content_lower for kw in ['health', 'finance', 'career', 'family', 'relationship']):
        return {"classification": "AREA", "confidence": 0.6, "reasoning": "Ongoing life area"}
    
    return {"classification": "RESOURCE", "confidence": 0.5, "reasoning": "Default classification"}


# ============================================================================
# Task/Deadline Extraction Agent
# ============================================================================

async def extract_tasks(content: str) -> List[Dict]:
    """
    Extract actionable tasks with deadlines from natural language.
    
    Returns:
        List of tasks: [
            {
                "task": "description",
                "deadline": "extracted date/time or null",
                "priority": "high" | "medium" | "low",
                "assignee": "person name or null"
            }
        ]
    """
    llm = get_llm()
    
    prompt = f"""Extract actionable tasks from this text. If no tasks, return empty array.

Text: "{content}"

Today's date: {datetime.now().strftime("%Y-%m-%d")}

Return JSON array only:
[
    {{
        "task": "task description",
        "deadline": "YYYY-MM-DD or relative like 'tomorrow'",
        "priority": "high" | "medium" | "low",
        "assignee": "person name or null"
    }}
]"""

    messages = [
        SystemMessage(content="Extract tasks and deadlines. Return valid JSON array only, empty [] if no tasks."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        # Find JSON array
        json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if json_match:
            tasks = json.loads(json_match.group())
            # Normalize deadlines
            for task in tasks:
                task["deadline"] = normalize_deadline(task.get("deadline"))
            return tasks
    except Exception as e:
        print(f"Task extraction error: {e}")
    
    return []


def normalize_deadline(deadline_str: Optional[str]) -> Optional[str]:
    """Convert relative deadlines to absolute dates."""
    if not deadline_str or deadline_str.lower() in ['null', 'none', '']:
        return None
    
    today = datetime.now()
    deadline_lower = deadline_str.lower()
    
    if 'today' in deadline_lower:
        return today.strftime("%Y-%m-%d")
    elif 'tomorrow' in deadline_lower:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif 'next week' in deadline_lower:
        return (today + timedelta(days=7)).strftime("%Y-%m-%d")
    elif 'friday' in deadline_lower:
        days_ahead = 4 - today.weekday()  # Friday is 4
        if days_ahead <= 0:
            days_ahead += 7
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    # Return as-is if already formatted or couldn't parse
    return deadline_str


# ============================================================================
# Combined Classification Pipeline
# ============================================================================

async def run_classification_pipeline(content: str) -> Dict:
    """
    Run PARA classification and task extraction.
    
    Returns:
        {
            "para": { classification details },
            "tasks": [ extracted tasks ]
        }
    """
    para_result = await classify_para(content)
    tasks = await extract_tasks(content)
    
    return {
        "para": para_result,
        "tasks": tasks
    }
