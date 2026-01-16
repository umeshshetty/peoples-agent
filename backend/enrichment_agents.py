from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
import json
import re
import traceback
from typing import List, Dict, Any

# Initialize Ollama
llm = ChatOllama(model="llama3.2:latest", temperature=0.0, base_url="http://localhost:11434")

# ============================================================================
# 1. Latent Intent & Blocker Agent
# ============================================================================

LATENT_INTENT_PROMPT = """You are an expert Project Manager AI. Your goal is to identify "Deep Intent" and project risks from a user's thought.
Analyze the user's thought for:
1. BLOCKERS: Anything stopping progress (e.g., "slow API", "waiting for approval").
2. RISKS: Potential future issues (e.g., "might miss deadline", "complicated logic").
3. AFFECTED PROJECTS: Try to map the thought to one of the active projects provided.

Active Projects:
{projects}

Return a JSON object with:
- "is_blocker": boolean
- "risk_level": "none", "low", "medium", "high", "critical"
- "affected_project_name": string or null
- "reason": brief explanation
"""

def analyze_intent_and_risk(thought_content: str, active_projects: List[str]) -> Dict[str, Any]:
    """
    Agent 1: Detects blockers and risks to projects.
    """
    try:
        projects_str = ", ".join(active_projects) if active_projects else "No active projects known."
        
        response = llm.invoke([
            SystemMessage(content=LATENT_INTENT_PROMPT.format(projects=projects_str)),
            HumanMessage(content=thought_content)
        ])
        
        # Parse JSON
        content = response.content.strip()
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"is_blocker": False, "risk_level": "none", "affected_project_name": None, "reason": ""}
        
    except Exception as e:
        print(f"Error in intent analysis: {e}")
        return {"is_blocker": False, "risk_level": "none", "affected_project_name": None, "reason": str(e)}

# ============================================================================
# 2. Social Graph Architect
# ============================================================================

SOCIAL_GRAPH_PROMPT = """You are a Super-Connector AI. Your goal is to map thoughts to the user's social network.
Given a list of TOPICS extracted from the thought, and a list of PEOPLE and their roles/interests, identify meaningful connections.

Topics: {topics}
People:
{people}

Logic:
- If a Topic matches a Person's role, interest, or past context -> Suggest a connection.
- Nudge the user to mention this topic to that person.

Return ONLY a valid JSON object with a "nudges" key. Do not add markdown formatting.
Example:
{{
    "nudges": [
        {{"person_name": "Alice", "reason": "Expert in X", "suggestion": "Talk to Alice"}}
    ]
}}
"""

def find_social_connections(topics: List[str], people_profiles: List[Dict]) -> List[Dict]:
    """
    Agent 2: Finds social connections based on topics.
    people_profiles should be list of dicts: {'name': '...', 'role': '...', 'topics': [...]}
    """
    try:
        if not topics or not people_profiles:
            return []
            
        people_str = "\n".join([f"- {p['name']} ({p.get('role', 'Unknown')}): {', '.join(p.get('topics', []))}" for p in people_profiles])
        
        response = llm.invoke([
            SystemMessage(content=SOCIAL_GRAPH_PROMPT.format(topics=", ".join(topics), people=people_str)),
            HumanMessage(content="Analyze connections.")
        ])
        
        # Try to find JSON object
        content = response.content.strip()
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            if isinstance(data, dict):
                return data.get("nudges", [])
            elif isinstance(data, list):
                return data
            return []
        return []
        
    except Exception as e:
        print(f"Error in social graph agent: {type(e).__name__}: {e}")
        return []

# ============================================================================
# 3. Actionability Auditor
# ============================================================================

ACTION_AUDITOR_PROMPT = """You are a GTD (Getting Things Done) expert. Your goal is to convert passive notes into atomic Action Items.
Analyze the thought for IMPLIED COMMITMENTS.
- "I need to fix X" -> Action "Fix X"
- "We should call Y" -> Action "Call Y"
- "Bug only happens on Safari" -> Action "Investigate bug on Safari"

Return a JSON object with a list of "actions":
{
    "actions": [
        {"description": "Fix the Safari bug", "urgency": 4, "status": "pending"}
    ]
}
Urgency is 1 (low) to 5 (critical).
If no action implied, return empty list.
"""

def audit_actionability(thought_content: str) -> List[Dict]:
    """
    Agent 3: Extracts action items from thought.
    """
    try:
        response = llm.invoke([
            SystemMessage(content=ACTION_AUDITOR_PROMPT),
            HumanMessage(content=thought_content)
        ])
        
        content = response.content.strip()
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            return data.get("actions", [])
        return []
        
    except Exception as e:
        print(f"Error in action auditor: {e}")
        return []
