"""
People's Agent - Claude Client for System 2 (Deliberative) Tasks
Uses Claude 3.5 Sonnet for multi-hop reasoning, synthesis, and deep analysis.
"""

import os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

# Configuration - API key should be set via ANTHROPIC_API_KEY environment variable
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def get_claude_llm(temperature: float = 0.3):
    """
    Get Claude 3.5 Sonnet for System 2 (deliberative) tasks.
    Use for: Synthesis, Serendipity, Consistency Auditing, Deep Analysis
    """
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=ANTHROPIC_API_KEY,
        temperature=temperature,
        max_tokens=2048
    )


async def claude_analyze(prompt: str, context: str = "", system_prompt: str = None) -> str:
    """
    Use Claude for deep analysis tasks.
    
    Args:
        prompt: The main question/task
        context: Background context to include
        system_prompt: Optional system prompt override
    
    Returns:
        Claude's response as string
    """
    llm = get_claude_llm()
    
    default_system = """You are a Socratic thinking partner with deep reasoning capabilities.
Your role is NOT to be a passive clerk. You are an active co-cognitive partner.

Key behaviors:
1. Challenge assumptions rather than just agreeing
2. Identify potential risks and contradictions
3. Find non-obvious connections between disparate ideas
4. Surface the "why" behind the "what"
5. Ask clarifying questions that force precision of thought"""
    
    messages = [
        SystemMessage(content=system_prompt or default_system),
        HumanMessage(content=f"{context}\n\n{prompt}" if context else prompt)
    ]
    
    response = await llm.ainvoke(messages)
    return response.content


# ============================================================================
# Specialized Claude Functions for System 2 Tasks
# ============================================================================

async def claude_synthesize_person(name: str, mentions: list, context: str = "") -> dict:
    """
    Deep synthesis of a person's profile using Claude's reasoning.
    """
    prompt = f"""Analyze these mentions of "{name}" and synthesize a deep profile.

Mentions:
{chr(10).join(f'- {m}' for m in mentions[:10])}

Extract:
1. role: Their actual role/title (not a placeholder)
2. relationship: Your relationship to them
3. topics: Key topics you discuss with them
4. summary: A narrative summary of who they are to you
5. working_style: How they prefer to work/communicate (if evident)
6. potential_value: What value this relationship could provide

Be specific and insightful, not generic."""

    response = await claude_analyze(prompt, context)
    
    # Parse response into structured format
    return {
        "name": name,
        "analysis": response,
        "model": "claude-sonnet-4-20250514"
    }


async def claude_find_serendipity(current_topic: str, distant_notes: list) -> list:
    """
    Find unexpected connections between current topic and distant notes.
    Uses Claude's ability to find remote analogies across domains.
    """
    prompt = f"""You are finding "Structural Holes" - unexpected but valuable connections.

Current Focus: {current_topic}

Distant Notes (from different clusters):
{chr(10).join(f'- {n}' for n in distant_notes[:8])}

Find 2-3 non-obvious connections. For each:
1. The connection (what links these ideas?)
2. Why it matters (what insight does this reveal?)
3. Action suggestion (how could user leverage this?)

Look for analogies across domains (e.g., "Your note about ant colonies is structurally similar to your distributed systems problem")."""

    response = await claude_analyze(prompt, system_prompt="""You are a Serendipity Engine.
Your job is to find remote analogies and unexpected connections.
Think like Steve Jobs: "Creativity is just connecting things."
Find connections that would make the user say "I never thought of it that way!"
""")
    
    return [{
        "insight": response,
        "source": "claude-serendipity",
        "topic": current_topic
    }]


async def claude_check_consistency(new_thought: str, previous_context: str) -> dict:
    """
    Check if new thought contradicts previous commitments or stated goals.
    """
    prompt = f"""Review this new thought for logical consistency with previous context.

NEW THOUGHT:
{new_thought}

PREVIOUS CONTEXT:
{previous_context}

Check for:
1. Contradictions to previous commitments
2. Conflicts with stated goals
3. Shifting priorities without acknowledgment
4. Potential blind spots

If you find dissonance, explain it helpfully and non-confrontationally.
If consistent, briefly note what aligns well."""

    response = await claude_analyze(prompt, system_prompt="""You are a Consistency Auditor.
Your job is to catch when someone's current thinking contradicts their past statements.
Be helpful, not judgmental. Frame contradictions as opportunities for reflection.
Example: "I notice you mentioned X last week, but this seems to suggest Y. Worth exploring?"
""")
    
    return {
        "has_contradiction": "contradict" in response.lower() or "conflict" in response.lower(),
        "analysis": response
    }


async def claude_extract_latent_anxiety(thought: str) -> dict:
    """
    Analyze the emotional undertones of a thought.
    Detect stress, procrastination, or hidden concerns.
    """
    prompt = f"""Analyze the emotional tone and hidden signals in this note:

"{thought}"

Extract:
1. latent_anxiety: What might the user be worried about? (null if nothing)
2. procrastination_signals: Are they avoiding something? (null if not)
3. stress_level: low/medium/high
4. underlying_need: What might they really need right now?

Be psychologically insightful but not presumptuous."""

    response = await claude_analyze(prompt, system_prompt="""You are an emotionally intelligent AI.
Your job is to read between the lines - what is the user feeling, not just saying?
Surface the human behind the note. Be empathetic, not clinical.""")
    
    return {
        "emotional_analysis": response,
        "model": "claude-sonnet-4-20250514"
    }
