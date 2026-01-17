"""
People's Agent - User Context Module
Loads user profile and provides context injection for all agent prompts.

The "conscious identity" of the assistant - knowing who it serves.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime


# Profile location - defaults to backend/user_profile.yaml
PROFILE_PATH = os.getenv(
    "USER_PROFILE_PATH", 
    Path(__file__).parent / "user_profile.yaml"
)

# Cached profile
_cached_profile: Optional[Dict] = None
_cache_time: Optional[datetime] = None
CACHE_DURATION_SECONDS = 300  # Reload every 5 minutes


def load_user_profile(force_reload: bool = False) -> Dict:
    """
    Load user profile from YAML file.
    Caches the result to avoid repeated disk reads.
    """
    global _cached_profile, _cache_time
    
    # Check cache
    if not force_reload and _cached_profile is not None:
        if _cache_time and (datetime.now() - _cache_time).seconds < CACHE_DURATION_SECONDS:
            return _cached_profile
    
    try:
        with open(PROFILE_PATH, 'r') as f:
            _cached_profile = yaml.safe_load(f)
            _cache_time = datetime.now()
            return _cached_profile
    except FileNotFoundError:
        print(f"⚠ User profile not found at {PROFILE_PATH}")
        return {}
    except yaml.YAMLError as e:
        print(f"⚠ Error parsing user profile: {e}")
        return {}


def get_user_identity() -> Dict:
    """Get the basic identity section."""
    profile = load_user_profile()
    return profile.get("identity", {})


def get_user_name() -> str:
    """Get user's name."""
    identity = get_user_identity()
    return identity.get("name", "User")


def get_active_projects() -> List[Dict]:
    """Get list of active projects."""
    profile = load_user_profile()
    projects = profile.get("projects", {})
    return [
        {"key": k, **v} for k, v in projects.items()
    ]


def get_project_names() -> List[str]:
    """Get just the project names for entity matching."""
    projects = get_active_projects()
    return [p.get("name", "") for p in projects]


def get_key_people() -> List[Dict]:
    """Get key people from personal context."""
    profile = load_user_profile()
    personal = profile.get("personal", {})
    return personal.get("family", [])


def get_interaction_preferences() -> Dict:
    """Get how the AI should interact with this user."""
    profile = load_user_profile()
    return profile.get("interaction_preferences", {})


def generate_user_context_prompt() -> str:
    """
    Generate the user context section to inject into agent prompts.
    This is the "conscious knowledge" about who the assistant is serving.
    """
    profile = load_user_profile()
    
    if not profile:
        return ""
    
    identity = profile.get("identity", {})
    prefs = profile.get("interaction_preferences", {})
    projects = profile.get("projects", {})
    strengths = profile.get("strengths", [])
    personal = profile.get("personal", {})
    
    # Build context string
    context_parts = []
    
    # Identity
    name = identity.get("name", "User")
    role = identity.get("role", "")
    company = identity.get("company", "")
    background = identity.get("background", "")
    
    context_parts.append(f"""## Who You're Talking To
- **Name**: {name}
- **Role**: {role} at {company}
- **Background**: {background}""")
    
    # Active Projects
    if projects:
        project_lines = []
        for key, proj in projects.items():
            proj_name = proj.get("name", key)
            proj_status = proj.get("status", proj.get("focus", "active"))
            project_lines.append(f"  - **{proj_name}**: {proj_status}")
        context_parts.append(f"""## Active Projects
{chr(10).join(project_lines)}""")
    
    # Strengths (abbreviated)
    if strengths:
        context_parts.append(f"""## Core Expertise
{', '.join(strengths[:3])}""")
    
    # Personal context (sensitive - used respectfully)
    family = personal.get("family", [])
    health = personal.get("health", {})
    if family or health:
        personal_lines = []
        for f in family:
            personal_lines.append(f"- {f.get('relationship', '')}: {f.get('name', '')} ({f.get('context', '')})")
        if health.get("goal"):
            personal_lines.append(f"- Fitness goal: {health.get('goal')} by {health.get('target_date', 'TBD')}")
        context_parts.append(f"""## Personal Context (respect this)
{chr(10).join(personal_lines)}""")
    
    # Interaction style
    if prefs:
        context_parts.append(f"""## How to Interact
- Treat as: {prefs.get('persona', 'professional')}
- Language: {prefs.get('language', 'clear and direct')}
- Proposals should align with: {prefs.get('proposals', 'their active projects')}""")
    
    return "\n\n".join(context_parts)


def generate_compact_context() -> str:
    """
    Generate a compact one-line context for token-constrained prompts.
    """
    profile = load_user_profile()
    identity = profile.get("identity", {})
    projects = profile.get("projects", {})
    
    name = identity.get("name", "User")
    role = identity.get("role", "")
    project_names = [p.get("name", "") for p in projects.values()]
    
    return f"User: {name}, {role}. Active projects: {', '.join(project_names[:3])}"


def add_learned_fact(fact: str) -> bool:
    """
    Add an auto-learned fact to the user profile.
    These are accumulated from conversations.
    """
    try:
        profile = load_user_profile(force_reload=True)
        
        if "auto_learned" not in profile:
            profile["auto_learned"] = {"facts": [], "last_updated": None}
        
        # Avoid duplicates (simple check)
        existing_facts = profile["auto_learned"].get("facts", [])
        for existing in existing_facts:
            if existing.get("fact", "").lower() == fact.lower():
                return False  # Already exists
        
        # Add new fact
        profile["auto_learned"]["facts"].append({
            "fact": fact,
            "learned_at": datetime.now().isoformat(),
            "source": "conversation"
        })
        profile["auto_learned"]["last_updated"] = datetime.now().isoformat()
        
        # Save back
        with open(PROFILE_PATH, 'w') as f:
            yaml.dump(profile, f, default_flow_style=False, allow_unicode=True)
        
        # Invalidate cache
        global _cached_profile
        _cached_profile = None
        
        return True
    except Exception as e:
        print(f"Error saving learned fact: {e}")
        return False


# Load profile on module import
_profile = load_user_profile()
if _profile:
    print(f"✓ User profile loaded for: {_profile.get('identity', {}).get('name', 'Unknown')}")
else:
    print("⚠ No user profile found - assistant will operate in generic mode")
