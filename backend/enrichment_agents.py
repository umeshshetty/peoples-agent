"""
People's Agent - Enrichment Agents
Active enrichment of thoughts with risk analysis, social connections, and actions.

Hybrid Model Approach:
- GLM4 (System 1): Fast extraction and initial analysis
- Claude (System 2): Consistency auditing and emotional analysis
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
import json
import re
import traceback
from typing import List, Dict, Any

# Import Claude for System 2 deep analysis
try:
    from claude_client import claude_check_consistency, claude_extract_latent_anxiety
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# GLM4 for fast System 1 tasks
llm = ChatOllama(model="glm4", temperature=0.0, base_url="http://localhost:11434")

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


# ============================================================================
# 4. Consistency Agent (System 2 - Claude)
# ============================================================================

async def check_consistency(new_thought: str, previous_context: str) -> Dict[str, Any]:
    """
    Agent 4: Check if new thought contradicts previous commitments or stated goals.
    Uses Claude for deep analysis of logical consistency.
    """
    if not CLAUDE_AVAILABLE:
        return {"has_contradiction": False, "analysis": "Claude not available for consistency check"}
    
    try:
        result = await claude_check_consistency(new_thought, previous_context)
        return result
    except Exception as e:
        print(f"Error in consistency check: {e}")
        return {"has_contradiction": False, "analysis": str(e)}


# ============================================================================
# 5. Latent Anxiety Extractor (System 2 - Claude)
# ============================================================================

async def extract_latent_anxiety(thought_content: str) -> Dict[str, Any]:
    """
    Agent 5: Analyze the emotional undertones of a thought.
    Detect stress, procrastination, or hidden concerns.
    Uses Claude for psychological insight.
    """
    if not CLAUDE_AVAILABLE:
        return {"emotional_analysis": "Claude not available for emotional analysis"}
    
    try:
        result = await claude_extract_latent_anxiety(thought_content)
        return result
    except Exception as e:
        print(f"Error in latent anxiety extraction: {e}")
        return {"emotional_analysis": str(e)}


# ============================================================================
# Combined Enrichment Pipeline
# ============================================================================

async def run_full_enrichment(
    thought_content: str,
    extracted_entities: List[Dict],
    active_projects: List[str],
    people_profiles: List[Dict],
    previous_context: str = ""
) -> Dict[str, Any]:
    """
    Run all enrichment agents on a thought.
    - GLM4 for fast initial analysis (System 1)
    - Claude for deep analysis when available (System 2)
    """
    results = {
        "intent_and_risk": {},
        "social_nudges": [],
        "actions": [],
        "consistency": {},
        "emotional_analysis": {}
    }
    
    # System 1 - Fast (GLM4)
    extracted_topics = [e["name"] for e in extracted_entities 
                       if e.get("type") in ["Topic", "Concept", "Technology", "Interest"]]
    
    results["intent_and_risk"] = analyze_intent_and_risk(thought_content, active_projects)
    results["social_nudges"] = find_social_connections(extracted_topics, people_profiles)
    results["actions"] = audit_actionability(thought_content)
    
    # System 2 - Deep (Claude) - only if significant thought
    if len(thought_content) > 50 and CLAUDE_AVAILABLE:
        results["consistency"] = await check_consistency(thought_content, previous_context)
        results["emotional_analysis"] = await extract_latent_anxiety(thought_content)
    
    return results

