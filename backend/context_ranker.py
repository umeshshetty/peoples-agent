"""
People's Agent - Context Ranker
Ranks and filters retrieved context to prevent performance degradation
as the knowledge graph grows.
"""

from typing import List, Dict, Tuple
import re


def rank_context(query: str, context_items: List[Dict], max_items: int = 5) -> List[Dict]:
    """
    Rank context items by relevance to the query.
    Uses multiple signals for scoring:
    1. Entity overlap (entities mentioned in both query and context)
    2. Recency (newer items scored higher)
    3. Length penalty (very long contexts scored lower to save tokens)
    
    Returns top N most relevant items.
    """
    if not context_items:
        return []
    
    query_tokens = set(query.lower().split())
    scored_items = []
    
    for item in context_items:
        score = 0.0
        content = item.get("content", "").lower()
        
        # 1. Token overlap score (0-1)
        content_tokens = set(content.split())
        overlap = len(query_tokens & content_tokens)
        if content_tokens:
            score += (overlap / len(query_tokens)) * 0.4  # Weight: 40%
        
        # 2. Entity match bonus
        # Check if any capitalized words (likely entities) match
        query_entities = set(re.findall(r'\b[A-Z][a-z]+\b', query))
        for entity in query_entities:
            if entity.lower() in content:
                score += 0.2  # Bonus for each entity match
        
        # 3. Recency score (based on position, assuming newer items first)
        position = context_items.index(item)
        recency_score = max(0, (len(context_items) - position) / len(context_items)) * 0.2
        score += recency_score
        
        # 4. Length penalty (penalize very long contexts)
        word_count = len(content.split())
        if word_count > 200:
            score -= 0.1
        elif word_count < 50:
            score += 0.1  # Bonus for concise context
        
        scored_items.append((score, item))
    
    # Sort by score descending
    scored_items.sort(key=lambda x: x[0], reverse=True)
    
    # Return top N
    return [item for _, item in scored_items[:max_items]]


def compress_context(context_text: str, max_chars: int = 2000) -> str:
    """
    Compress context to fit within token budget.
    Keeps most relevant parts (beginning and end of each section).
    """
    if len(context_text) <= max_chars:
        return context_text
    
    # Split by newlines to preserve structure
    lines = context_text.split('\n')
    
    # Keep first and last portions
    half_budget = max_chars // 2
    
    result_start = []
    result_end = []
    chars_start = 0
    chars_end = 0
    
    for line in lines:
        if chars_start + len(line) < half_budget:
            result_start.append(line)
            chars_start += len(line) + 1
        else:
            break
    
    for line in reversed(lines):
        if chars_end + len(line) < half_budget:
            result_end.insert(0, line)
            chars_end += len(line) + 1
        else:
            break
    
    if result_start and result_end:
        return '\n'.join(result_start) + '\n... [context compressed] ...\n' + '\n'.join(result_end)
    
    # Fallback: just truncate
    return context_text[:max_chars] + "..."


def estimate_token_count(text: str) -> int:
    """
    Rough estimate of token count (for Llama models).
    Approximately 1 token per 4 characters.
    """
    return len(text) // 4


def should_fetch_more_context(current_context: str, max_tokens: int = 1500) -> bool:
    """
    Check if we should fetch more context or if we're at capacity.
    """
    current_tokens = estimate_token_count(current_context)
    return current_tokens < max_tokens
