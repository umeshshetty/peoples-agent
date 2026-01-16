"""
People's Agent - Entity Resolver
Disambiguates entities with similar names (e.g., "John Smith" vs "John Doe").
"""

from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher


def similarity_score(a: str, b: str) -> float:
    """Calculate string similarity (0-1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def resolve_entity(
    entity_name: str, 
    entity_type: str,
    existing_entities: List[Dict],
    context: str = "",
    threshold: float = 0.85
) -> Tuple[str, bool]:
    """
    Resolve an entity against existing entities in the graph.
    
    Returns:
        (resolved_name, is_new) - The canonical name and whether it's a new entity
    
    Logic:
    1. Exact match: Return existing entity
    2. High similarity (>0.85): Likely same entity, merge
    3. Partial match with same type: Flag for disambiguation
    4. No match: New entity
    """
    entity_name_lower = entity_name.lower().strip()
    
    # Check for exact matches first
    for existing in existing_entities:
        if existing.get("name", "").lower() == entity_name_lower:
            return existing.get("name"), False
    
    # Check for high-similarity matches
    candidates = []
    for existing in existing_entities:
        existing_name = existing.get("name", "")
        existing_type = existing.get("type", "")
        
        # Same type preferred
        type_match = existing_type.lower() == entity_type.lower()
        
        sim = similarity_score(entity_name, existing_name)
        
        if sim >= threshold:
            candidates.append({
                "name": existing_name,
                "score": sim,
                "type_match": type_match
            })
    
    if candidates:
        # Sort by score (and type match as tiebreaker)
        candidates.sort(key=lambda x: (x["score"], x["type_match"]), reverse=True)
        best = candidates[0]
        
        # If very high similarity (>0.9) and same type, auto-merge
        if best["score"] > 0.9 and best["type_match"]:
            print(f"   ► Entity resolved: '{entity_name}' → '{best['name']}' (score: {best['score']:.2f})")
            return best["name"], False
        
        # Otherwise, use context to disambiguate
        if best["score"] > threshold:
            # Check if context mentions distinguishing details
            if has_distinguishing_context(entity_name, best["name"], context):
                # Keep as separate entities
                return entity_name, True
            else:
                # Merge with existing
                print(f"   ► Entity merged: '{entity_name}' → '{best['name']}'")
                return best["name"], False
    
    # No match found, new entity
    return entity_name, True


def has_distinguishing_context(new_name: str, existing_name: str, context: str) -> bool:
    """
    Check if context contains info that distinguishes two similar entities.
    E.g., "John Smith from Marketing" vs "John Smith from Engineering"
    """
    context_lower = context.lower()
    
    # Look for role/department indicators
    distinguishing_patterns = [
        "from ", "at ", "in ", "the ", "our ", "their ",
        "manager", "director", "lead", "head", "chief",
        "department", "team", "company", "org"
    ]
    
    # If new name appears with distinguishing context, keep separate
    new_name_lower = new_name.lower()
    for pattern in distinguishing_patterns:
        # Check if pattern appears near the entity name
        if pattern in context_lower:
            idx = context_lower.find(new_name_lower)
            if idx != -1:
                # Check 50 chars before and after
                window_start = max(0, idx - 50)
                window_end = min(len(context), idx + len(new_name) + 50)
                window = context_lower[window_start:window_end]
                if pattern in window:
                    return True
    
    return False


def batch_resolve_entities(
    entities: List[Dict],
    existing_entities: List[Dict],
    context: str = ""
) -> List[Dict]:
    """
    Resolve a batch of entities, returning deduplicated list.
    """
    resolved = []
    seen_names = set()
    
    for entity in entities:
        name = entity.get("name", "")
        entity_type = entity.get("type", "Unknown")
        
        resolved_name, is_new = resolve_entity(
            name, entity_type, existing_entities, context
        )
        
        # Deduplicate within batch
        if resolved_name.lower() not in seen_names:
            seen_names.add(resolved_name.lower())
            resolved.append({
                **entity,
                "name": resolved_name,
                "is_new": is_new
            })
    
    return resolved
