"""
People's Agent - Synthesis Agents
Background agents that process raw thoughts into organized profiles.
These run after each thought to synthesize intelligence.
"""

from typing import List, Dict, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
import os
import json
import re

# Configuration
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_llm(temperature: float = 0.3):
    """Get LLM for synthesis - lower temp for consistency."""
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )


# ============================================================================
# Person Profile Synthesizer
# ============================================================================

async def synthesize_person_profile(
    person_name: str,
    related_thoughts: List[Dict],
    existing_profile: Optional[Dict] = None
) -> Dict:
    """
    Synthesize a person profile from related thoughts.
    Returns structured info: role, relationship, topics, last_context
    """
    llm = get_llm()
    
    # Build context from thoughts
    thoughts_text = "\n".join([
        f"- [{t.get('timestamp', '')[:10]}] {t.get('content', '')}"
        for t in related_thoughts[:10]
    ])
    
    existing_info = ""
    if existing_profile:
        existing_info = f"""
Current profile:
- Role: {existing_profile.get('role', 'Unknown')}
- Relationship: {existing_profile.get('relationship', 'Unknown')}
- Topics: {', '.join(existing_profile.get('topics', []))}
"""
    
    prompt = f"""Analyze these thoughts about "{person_name}" and create a profile.

Thoughts mentioning {person_name}:
{thoughts_text}
{existing_info}
Return JSON only:
{{
    "name": "{person_name}",
    "role": "their job title or role (e.g., Boss, Manager, Colleague, Friend)",
    "relationship": "your relationship to them (e.g., Reports to, Works with, Friend)",
    "topics": ["key topics discussed with them"],
    "summary": "one sentence about who they are and your interactions"
}}"""

    messages = [
        SystemMessage(content="You are a profile synthesizer. Extract structured info about people. Return valid JSON only."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        # Parse JSON from response
        json_match = re.search(r'\{[^{}]*\}', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"Error synthesizing person profile: {e}")
    
    # Fallback
    return {
        "name": person_name,
        "role": "Unknown",
        "relationship": "Mentioned",
        "topics": [],
        "summary": f"Mentioned in {len(related_thoughts)} thoughts"
    }


# ============================================================================
# Project Synthesizer
# ============================================================================

async def synthesize_project(
    project_name: str,
    related_thoughts: List[Dict]
) -> Dict:
    """
    Synthesize project info from related thoughts.
    Returns: status, people involved, deadlines, summary
    """
    llm = get_llm()
    
    thoughts_text = "\n".join([
        f"- [{t.get('timestamp', '')[:10]}] {t.get('content', '')}"
        for t in related_thoughts[:10]
    ])
    
    prompt = f"""Analyze these thoughts about the project "{project_name}".

Thoughts:
{thoughts_text}

Return JSON only:
{{
    "name": "{project_name}",
    "status": "Active/Completed/Blocked/Planning",
    "people": ["people involved"],
    "deadline": "mentioned deadline or null",
    "summary": "one sentence project summary"
}}"""

    messages = [
        SystemMessage(content="You are a project analyzer. Extract structured project info. Return valid JSON only."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        json_match = re.search(r'\{[^{}]*\}', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"Error synthesizing project: {e}")
    
    return {
        "name": project_name,
        "status": "Active",
        "people": [],
        "deadline": None,
        "summary": f"Mentioned in {len(related_thoughts)} thoughts"
    }


# ============================================================================
# Meeting Synthesizer
# ============================================================================

async def synthesize_meeting(thought_content: str) -> Optional[Dict]:
    """
    Extract meeting info from a thought.
    Returns: title, when, participants, agenda
    """
    llm = get_llm()
    
    prompt = f"""Extract meeting details from this text. If no meeting is mentioned, return null.

Text: {thought_content}

Return JSON only (or null):
{{
    "title": "meeting title or subject",
    "when": "date/time mentioned",
    "participants": ["people attending"],
    "agenda": "what will be discussed"
}}"""

    messages = [
        SystemMessage(content="Extract meeting details. Return valid JSON or null."),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = await llm.ainvoke(messages)
        if "null" in response.content.lower():
            return None
        json_match = re.search(r'\{[^{}]*\}', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"Error synthesizing meeting: {e}")
    
    return None


# ============================================================================
# Main Synthesis Pipeline
# ============================================================================

class SynthesisStatus:
    """Track synthesis status for UI feedback."""
    def __init__(self):
        self.steps = []
        self.current_step = ""
        self.complete = False
    
    def add_step(self, step: str):
        self.steps.append({"step": step, "status": "done"})
    
    def set_current(self, step: str):
        self.current_step = step
        self.steps.append({"step": step, "status": "running"})
    
    def to_dict(self):
        return {
            "steps": self.steps,
            "current": self.current_step,
            "complete": self.complete
        }


async def run_synthesis_pipeline(
    thought_id: str,
    thought_content: str,
    entities: List[Dict],
    knowledge_graph
) -> Dict:
    """
    Run all synthesis agents on a new thought.
    Updates Neo4j with synthesized profiles.
    Returns status for UI.
    """
    status = SynthesisStatus()
    results = {"people": [], "projects": [], "meetings": None}
    
    # 1. Synthesize person profiles
    people_entities = [e for e in entities if e.get('type', '').lower() == 'person']
    if people_entities:
        status.set_current("Updating person profiles...")
        for entity in people_entities:
            name = entity.get('name', '')
            if name:
                # Get related thoughts from graph
                related = knowledge_graph.find_by_entity(name)
                related_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in related]
                
                # Synthesize profile
                profile = await synthesize_person_profile(name, related_dicts)
                results["people"].append(profile)
                
                # Store in Neo4j
                await store_person_profile(knowledge_graph, name, profile)
        
        status.add_step(f"Updated {len(people_entities)} person profiles")
    
    # 2. Synthesize project info
    project_entities = [e for e in entities if e.get('type', '').lower() in ['project', 'tool']]
    if project_entities:
        status.set_current("Synthesizing projects...")
        for entity in project_entities:
            name = entity.get('name', '')
            if name:
                related = knowledge_graph.find_by_entity(name)
                related_dicts = [t.to_dict() if hasattr(t, 'to_dict') else t for t in related]
                
                project = await synthesize_project(name, related_dicts)
                results["projects"].append(project)
                
                await store_project_profile(knowledge_graph, name, project)
        
        status.add_step(f"Updated {len(project_entities)} projects")
    
    # 3. Check for meeting in thought
    status.set_current("Checking for meetings...")
    meeting = await synthesize_meeting(thought_content)
    if meeting:
        results["meetings"] = meeting
        await store_meeting(knowledge_graph, thought_id, meeting)
        status.add_step("Extracted meeting info")
    else:
        status.add_step("No meeting detected")
    
    status.complete = True
    return {"status": status.to_dict(), "results": results}


# ============================================================================
# Neo4j Storage Functions
# ============================================================================

async def store_person_profile(knowledge_graph, person_name: str, profile: Dict):
    """Store synthesized person profile in Neo4j."""
    try:
        with knowledge_graph.driver.session() as session:
            session.run("""
                MERGE (e:Entity {name: $name, type: 'Person'})
                MERGE (p:PersonProfile {name: $name})
                SET p.role = $role,
                    p.relationship = $relationship,
                    p.topics = $topics,
                    p.summary = $summary,
                    p.updated_at = datetime()
                MERGE (e)-[:HAS_PROFILE]->(p)
            """, 
                name=person_name,
                role=profile.get('role', 'Unknown'),
                relationship=profile.get('relationship', 'Mentioned'),
                topics=profile.get('topics', []),
                summary=profile.get('summary', '')
            )
    except Exception as e:
        print(f"Error storing person profile: {e}")


async def store_project_profile(knowledge_graph, project_name: str, profile: Dict):
    """Store synthesized project in Neo4j."""
    try:
        with knowledge_graph.driver.session() as session:
            session.run("""
                MERGE (e:Entity {name: $name})
                MERGE (p:ProjectProfile {name: $name})
                SET p.status = $status,
                    p.people = $people,
                    p.deadline = $deadline,
                    p.summary = $summary,
                    p.updated_at = datetime()
                MERGE (e)-[:HAS_PROFILE]->(p)
            """,
                name=project_name,
                status=profile.get('status', 'Active'),
                people=profile.get('people', []),
                deadline=profile.get('deadline'),
                summary=profile.get('summary', '')
            )
    except Exception as e:
        print(f"Error storing project profile: {e}")


async def store_meeting(knowledge_graph, thought_id: str, meeting: Dict):
    """Store extracted meeting in Neo4j."""
    try:
        with knowledge_graph.driver.session() as session:
            session.run("""
                MERGE (m:Meeting {thought_id: $thought_id})
                SET m.title = $title,
                    m.when = $when,
                    m.participants = $participants,
                    m.agenda = $agenda
                WITH m
                MATCH (t:Thought {id: $thought_id})
                MERGE (t)-[:HAS_MEETING]->(m)
            """,
                thought_id=thought_id,
                title=meeting.get('title', ''),
                when=meeting.get('when', ''),
                participants=meeting.get('participants', []),
                agenda=meeting.get('agenda', '')
            )
    except Exception as e:
        print(f"Error storing meeting: {e}")
