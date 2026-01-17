"""
Intent Classifier for Context-Aware Responses

Detects whether a user's thought requires:
- UTILITY: Direct action (reminders, scheduling, status checks)
- STRATEGIC: Deep thinking (brainstorming, analysis, decisions)
- SIMPLE: Basic interactions (greetings, acknowledgments)
"""

from typing import Literal

IntentType = Literal["simple", "utility", "strategic"]


# Keyword patterns for quick classification
UTILITY_KEYWORDS = [
    # Reminders & Scheduling
    "remind", "reminder", "schedule", "calendar", "meeting",
    "set alarm", "when is", "what time", "deadline",
    
    # Status & Lookups
    "status of", "check on", "what's the", "show me",
    "find", "where is", "who is", "list",
    
    # Quick Actions
    "create task", "add to", "note that", "save this",
    "update", "change", "delete", "remove"
]

STRATEGIC_KEYWORDS = [
    # Decision Making
    "should i", "should we", "would it be", "is it worth",
    "what if", "how should", "why would", "which approach",
    
    # Analysis
    "analyze", "evaluate", "compare", "assess",
    "pros and cons", "trade-offs", "implications",
    
    # Brainstorming
    "ideas for", "help me think", "brainstorm", "explore",
    "alternatives", "options", "possibilities",
    
    # Architecture/Design
    "design", "architecture", "refactor", "optimize",
    "scalability", "performance", "infrastructure"
]

SIMPLE_PATTERNS = [
    "hi", "hello", "hey", "thanks", "thank you",
    "ok", "okay", "got it", "bye", "goodbye"
]


def classify_intent(thought: str, salience_score: float = 0.5) -> IntentType:
    """
    Classify thought intent to determine response mode.
    
    Args:
        thought: User's input
        salience_score: Cognitive salience (0-1), higher = more complex
        
    Returns:
        "simple" | "utility" | "strategic"
    """
    thought_lower = thought.lower().strip()
    
    # 1. Check for simple patterns first
    if len(thought) < 15 or any(thought_lower.startswith(p) for p in SIMPLE_PATTERNS):
        return "simple"
    
    # 2. Check for utility keywords
    utility_matches = sum(1 for kw in UTILITY_KEYWORDS if kw in thought_lower)
    
    # 3. Check for strategic keywords
    strategic_matches = sum(1 for kw in STRATEGIC_KEYWORDS if kw in thought_lower)
    
    # 4. Use salience as a tie-breaker
    # Low salience + utility words = definitely utility
    if utility_matches > 0 and salience_score < 0.4:
        return "utility"
    
    # High salience + strategic words = definitely strategic
    if strategic_matches > 0 and salience_score > 0.6:
        return "strategic"
    
    # 5. Keyword dominance decides
    if utility_matches > strategic_matches:
        return "utility"
    elif strategic_matches > utility_matches:
        return "strategic"
    
    # 6. Salience as final arbiter
    if salience_score < 0.3:
        return "utility"  # Low complexity, likely administrative
    elif salience_score > 0.7:
        return "strategic"  # High complexity, needs deep thought
    
    # 7. Default to strategic for ambiguous cases
    # (Better to over-think than under-serve)
    return "strategic"


def get_intent_description(intent: IntentType) -> str:
    """Get human-readable description of intent."""
    descriptions = {
        "simple": "Basic interaction",
        "utility": "Administrative task or quick action",
        "strategic": "Complex thinking or decision-making"
    }
    return descriptions.get(intent, "Unknown")
