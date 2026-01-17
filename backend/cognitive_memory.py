"""
People's Agent - Cognitive Memory Architecture
Implements human-like memory systems:
- Episodic Memory: Specific events with context
- Semantic Memory: Accumulated facts and knowledge
- Entity State Machine: Track evolution over time
- Narrative Detection: Recognize ongoing storylines
- Salience Scoring: Weight by importance and recency
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Literal
from datetime import datetime, timedelta
from enum import Enum
import math


# ============================================================================
# Entity States - How entities evolve over time
# ============================================================================

class ProjectState(Enum):
    """States a project can be in"""
    IDEA = "idea"
    PLANNING = "planning" 
    ACTIVE = "active"
    BLOCKED = "blocked"
    STALLED = "stalled"
    COMPLETING = "completing"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class PersonRelationshipState(Enum):
    """States of relationship with a person"""
    STRANGER = "stranger"
    MENTIONED = "mentioned"
    ACQUAINTANCE = "acquaintance"
    COLLEAGUE = "colleague"
    COLLABORATOR = "collaborator"
    TRUSTED_ALLY = "trusted_ally"
    CONFLICT = "conflict"
    FORMER = "former"


class GoalState(Enum):
    """States a goal can be in"""
    ASPIRATIONAL = "aspirational"
    COMMITTED = "committed"
    IN_PROGRESS = "in_progress"
    STRUGGLING = "struggling"
    NEAR_COMPLETION = "near_completion"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"


# State transition rules
STATE_TRANSITIONS = {
    "Project": {
        "idea": ["planning", "abandoned"],
        "planning": ["active", "idea", "abandoned"],
        "active": ["blocked", "stalled", "completing", "abandoned"],
        "blocked": ["active", "stalled", "abandoned"],
        "stalled": ["active", "abandoned"],
        "completing": ["completed", "active"],
        "completed": [],  # Terminal state
        "abandoned": []   # Terminal state
    },
    "Person": {
        "stranger": ["mentioned"],
        "mentioned": ["acquaintance", "colleague"],
        "acquaintance": ["colleague", "collaborator", "conflict"],
        "colleague": ["collaborator", "trusted_ally", "conflict", "former"],
        "collaborator": ["trusted_ally", "colleague", "conflict"],
        "trusted_ally": ["collaborator", "conflict", "former"],
        "conflict": ["colleague", "former"],
        "former": ["colleague", "acquaintance"]
    }
}


# ============================================================================
# Episodic Memory - Specific Events
# ============================================================================

@dataclass
class EpisodicMemory:
    """A specific event/experience - the 'what happened'"""
    id: str
    event_description: str
    timestamp: str
    entities_involved: List[str]  # Entity names
    location: Optional[str] = None
    emotional_valence: str = "neutral"  # positive, negative, neutral
    emotional_intensity: float = 0.5  # 0-1
    source_thought_id: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Semantic Memory - Accumulated Facts
# ============================================================================

@dataclass
class SemanticFact:
    """A distilled fact/belief about an entity"""
    fact: str
    confidence: float = 0.8  # 0-1, increases with more evidence
    source_count: int = 1    # How many episodes support this
    first_learned: str = ""
    last_confirmed: str = ""
    contradicted_by: List[str] = field(default_factory=list)  # IDs of contradicting facts
    
    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Cognitive Entity - Rich Entity with Memory
# ============================================================================

@dataclass
class CognitiveEntity:
    """
    An entity with human-like memory representation.
    Tracks both episodic experiences and semantic facts.
    """
    name: str
    entity_type: str  # Person, Project, Topic, etc.
    
    # Current state in state machine
    current_state: str = "mentioned"
    state_history: List[Dict] = field(default_factory=list)
    
    # Semantic facts (accumulated knowledge)
    semantic_facts: List[SemanticFact] = field(default_factory=list)
    
    # Episodic memories (specific events)
    episodic_memories: List[str] = field(default_factory=list)  # IDs of EpisodicMemory
    
    # Relationship dynamics (for Person type)
    relationship_trajectory: str = "stable"  # improving, stable, declining
    trust_level: float = 0.5
    interaction_frequency: str = "occasional"  # daily, weekly, occasional, rare
    last_interaction: Optional[str] = None
    
    # Salience metrics
    mention_count: int = 1
    emotional_weight: float = 0.5  # Average emotional intensity
    last_accessed: str = ""
    
    # Narrative connections
    active_narratives: List[str] = field(default_factory=list)  # IDs of narratives this entity is part of
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "current_state": self.current_state,
            "state_history": self.state_history,
            "semantic_facts": [f.to_dict() if hasattr(f, 'to_dict') else f for f in self.semantic_facts],
            "episodic_memories": self.episodic_memories,
            "relationship_trajectory": self.relationship_trajectory,
            "trust_level": self.trust_level,
            "interaction_frequency": self.interaction_frequency,
            "last_interaction": self.last_interaction,
            "mention_count": self.mention_count,
            "emotional_weight": self.emotional_weight,
            "active_narratives": self.active_narratives
        }
    
    def add_semantic_fact(self, fact: str, confidence: float = 0.8):
        """Add or strengthen a semantic fact"""
        now = datetime.now().isoformat()
        
        # Check if fact already exists
        for existing in self.semantic_facts:
            if self._facts_similar(existing.fact, fact):
                # Strengthen existing fact
                existing.source_count += 1
                existing.confidence = min(1.0, existing.confidence + 0.1)
                existing.last_confirmed = now
                return
        
        # Add new fact
        self.semantic_facts.append(SemanticFact(
            fact=fact,
            confidence=confidence,
            source_count=1,
            first_learned=now,
            last_confirmed=now
        ))
    
    def _facts_similar(self, fact1: str, fact2: str) -> bool:
        """Check if two facts are semantically similar (simple version)"""
        # Basic similarity - could be enhanced with embeddings
        words1 = set(fact1.lower().split())
        words2 = set(fact2.lower().split())
        overlap = len(words1 & words2) / max(len(words1 | words2), 1)
        return overlap > 0.6
    
    def transition_state(self, new_state: str, reason: str = ""):
        """Transition entity to new state if valid"""
        valid_transitions = STATE_TRANSITIONS.get(self.entity_type, {})
        current_valid = valid_transitions.get(self.current_state, [])
        
        if new_state in current_valid or not current_valid:
            self.state_history.append({
                "from": self.current_state,
                "to": new_state,
                "timestamp": datetime.now().isoformat(),
                "reason": reason
            })
            self.current_state = new_state
            return True
        return False


# ============================================================================
# Narrative - Ongoing Storylines
# ============================================================================

@dataclass
class Narrative:
    """
    An ongoing storyline that connects multiple thoughts.
    Like "The Project X Journey" or "My health improvement"
    """
    id: str
    title: str
    description: str
    entities_involved: List[str]
    
    # Narrative arc tracking
    arc_stage: str = "beginning"  # beginning, rising_action, climax, falling_action, resolution
    emotional_arc: str = "neutral"  # hopeful, tense, triumphant, defeated
    
    # Timeline
    started: str = ""
    last_updated: str = ""
    episode_ids: List[str] = field(default_factory=list)
    
    # Status
    is_active: bool = True
    resolution: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Salience Calculator
# ============================================================================

def calculate_salience(
    thought_content: str,
    entities: List[CognitiveEntity],
    timestamp: str,
    emotional_markers: List[str] = None
) -> float:
    """
    Calculate how 'salient' (important/memorable) a thought is.
    Higher salience = more likely to be retrieved.
    """
    salience = 0.0
    
    # 1. Emotional markers (+0.3 max)
    emotional_words = [
        "frustrated", "excited", "worried", "happy", "angry", "anxious",
        "proud", "disappointed", "thrilled", "stressed", "relieved"
    ]
    content_lower = thought_content.lower()
    emotion_count = sum(1 for word in emotional_words if word in content_lower)
    salience += min(emotion_count * 0.1, 0.3)
    
    # 2. Commitment/Action words (+0.2 max)
    commitment_words = ["i will", "i need to", "i must", "have to", "going to", "decided to"]
    if any(word in content_lower for word in commitment_words):
        salience += 0.2
    
    # 3. Entity connection density (+0.3 max)
    total_connections = sum(e.mention_count for e in entities)
    salience += min(total_connections * 0.05, 0.3)
    
    # 4. Question/uncertainty markers (+0.1)
    if "?" in thought_content or any(w in content_lower for w in ["how", "why", "what if", "should i"]):
        salience += 0.1
    
    # 5. Novelty (fewer entity connections = more novel = more salient)
    if entities and all(e.mention_count == 1 for e in entities):
        salience += 0.1
    
    # Cap at 1.0
    return min(salience, 1.0)


def recency_decay(timestamp_str: str, half_life_days: int = 7) -> float:
    """
    Calculate decay factor based on recency.
    Uses exponential decay with configurable half-life.
    """
    try:
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        now = datetime.now(timestamp.tzinfo) if timestamp.tzinfo else datetime.now()
        age_days = (now - timestamp).days
        
        # Exponential decay: 0.5^(age/half_life)
        decay = math.pow(0.5, age_days / half_life_days)
        return max(decay, 0.1)  # Minimum 10% weight
    except:
        return 1.0  # Default to full weight on error


# ============================================================================
# Cognitive Extraction - The New Way
# ============================================================================

def extract_cognitive_entity(
    name: str,
    entity_type: str,
    thought_content: str,
    existing_entity: Optional[CognitiveEntity] = None
) -> CognitiveEntity:
    """
    Create or update a cognitive entity from a thought.
    This is the 'accumulation' model vs flat extraction.
    """
    now = datetime.now().isoformat()
    
    if existing_entity:
        entity = existing_entity
        entity.mention_count += 1
        entity.last_accessed = now
    else:
        entity = CognitiveEntity(
            name=name,
            entity_type=entity_type,
            current_state="mentioned",
            last_accessed=now
        )
    
    # Infer state from content
    content_lower = thought_content.lower()
    
    if entity_type == "Project":
        if any(w in content_lower for w in ["blocked", "stuck", "can't proceed"]):
            entity.transition_state("blocked", "Blocking language detected")
        elif any(w in content_lower for w in ["finished", "completed", "done", "shipped"]):
            entity.transition_state("completed", "Completion language detected")
        elif any(w in content_lower for w in ["started", "beginning", "kicked off"]):
            entity.transition_state("active", "Active language detected")
    
    elif entity_type == "Person":
        if any(w in content_lower for w in ["great", "helpful", "trust", "appreciate"]):
            entity.relationship_trajectory = "improving"
            entity.trust_level = min(1.0, entity.trust_level + 0.1)
        elif any(w in content_lower for w in ["frustrated", "difficult", "conflict", "issue with"]):
            entity.relationship_trajectory = "declining"
            entity.trust_level = max(0.0, entity.trust_level - 0.1)
    
    return entity


# ============================================================================
# Narrative Detection
# ============================================================================

def detect_narrative_continuation(
    thought_content: str,
    entities: List[str],
    existing_narratives: List[Narrative]
) -> Optional[Narrative]:
    """
    Check if a thought continues an existing narrative.
    Returns the narrative if found, None otherwise.
    """
    if not existing_narratives:
        return None
    
    content_lower = thought_content.lower()
    
    for narrative in existing_narratives:
        if not narrative.is_active:
            continue
        
        # Check entity overlap
        entity_overlap = len(set(entities) & set(narrative.entities_involved))
        if entity_overlap == 0:
            continue
        
        # Check thematic continuity (simple keyword matching)
        narrative_words = set(narrative.title.lower().split() + narrative.description.lower().split())
        thought_words = set(content_lower.split())
        word_overlap = len(narrative_words & thought_words)
        
        if entity_overlap >= 1 and word_overlap >= 2:
            return narrative
    
    return None


def infer_narrative_arc_shift(thought_content: str, current_arc: str) -> str:
    """
    Infer if the narrative arc has shifted based on thought content.
    """
    content_lower = thought_content.lower()
    
    # Resolution signals
    if any(w in content_lower for w in ["finally", "solved", "fixed", "done", "achieved", "success"]):
        return "resolution"
    
    # Climax signals
    if any(w in content_lower for w in ["deadline", "critical", "make or break", "crucial", "big day"]):
        return "climax"
    
    # Rising action signals
    if any(w in content_lower for w in ["challenge", "problem", "issue", "complication", "obstacle"]):
        return "rising_action"
    
    # Falling action signals
    if any(w in content_lower for w in ["wrapping up", "almost done", "finishing", "final steps"]):
        return "falling_action"
    
    return current_arc  # No change detected
