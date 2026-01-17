"""
People's Agent - LangGraph Agentic Orchestration
A personal assistant that stores notes, remembers conversations, and provides
context-aware responses using your stored knowledge.
"""

from typing import TypedDict, Annotated, Literal, List, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
import uuid
from datetime import datetime

# Local imports
from knowledge_graph import (
    knowledge_graph, ThoughtNode, Entity, Category, ActionItem, SocialNudge
)
from extraction_agents import extract_all, find_relationship_context, critique_extraction, refine_extraction
from enrichment_agents import (
    analyze_intent_and_risk, find_social_connections, audit_actionability
)
from serendipity_agent import get_serendipity_nudges
from zettelkasten_agent import should_atomize, atomize_content, create_atomic_thoughts
from task_decomposition_agent import decompose_task, create_task_hierarchy
from context_ranker import compress_context, estimate_token_count
from background_worker import run_in_background
from entity_resolver import batch_resolve_entities
from synthesis_agents import run_synthesis_pipeline
from classification_agents import run_classification_pipeline
import vector_store

# Configuration
MODEL_NAME = os.getenv("OLLAMA_MODEL", "glm4")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ============================================================================
# State Definition
# ============================================================================

class AgentState(TypedDict):
    """State that flows through the agent graph."""
    # Original user input
    thought: str
    # Generated thought ID
    thought_id: str
    # Is this a question that needs retrieval?
    is_question: bool
    # Retrieved notes relevant to question
    retrieved_notes: str
    # Analysis of the thought
    analysis: str
    # Generated insights
    insights: str
    # Final response
    response: str
    # Conversation context
    conversation_context: str
    # Conversation messages
    messages: Annotated[list, add_messages]
    # Processing stage
    stage: str
    # Extracted entities
    entities: List[dict]
    # Categories
    categories: List[dict]
    # Summary
    summary: str
    # Related context from knowledge graph
    related_context: str
    # Synthesis status for UI
    synthesis_status: dict
    # PARA classification (Project/Area/Resource/Archive)
    para_classification: str
    # Extracted tasks with deadlines
    tasks: List[dict]
    # Reflection loop state
    reflection_iterations: int
    critique: str
    # Enrichment State
    is_blocker: bool
    affected_project: str
    actions: List[dict]
    nudges: List[dict]
    serendipity_nudges: List[dict]
    atomic_notes_created: List[str]
    task_hierarchy: dict


# ============================================================================
# LLM Setup
# ============================================================================

def get_llm():
    """Get the Ollama LLM instance."""
    return ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
    )


# ============================================================================
# Agent Nodes
# ============================================================================

async def context_loader(state: AgentState) -> AgentState:
    """
    Loads conversation context and CHECKS for relevant notes for ALL thoughts.
    This enables 'Proactive Context' - giving the AI short-term memory of related concepts.
    """
    thought = state['thought']
    
    # Get conversation context
    conversation_context = knowledge_graph.get_conversation_context(limit=6)
    
    # Detect if this is a question (simple heuristic)
    is_question = (
        '?' in thought or
        thought.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who', 'can', 'could', 'would', 'should', 'do', 'does', 'is', 'are', 'tell me', 'explain', 'help me'))
    )
    
    # Proactive Retrieval: ALWAYS fetch top 3 relevant notes from Vector Store
    # This helps disambiguate entities like "him" or "the project"
    retrieved_notes = ""
    semantic_results = vector_store.get_context_for_query(thought, limit=3)
    if semantic_results:
        retrieved_notes = semantic_results + "\n"
    
    # Also search knowledge graph for questions or if keywords found
    if is_question:
        relevant_thoughts = knowledge_graph.search_notes(thought, limit=3)
        if relevant_thoughts:
            retrieved_notes += "\nFrom your knowledge graph:\n"
            for t in relevant_thoughts:
                retrieved_notes += f"- [{t.timestamp[:10]}] {t.content[:200]}...\n"
    
    # Compress context if too long (prevents performance degradation at scale)
    if len(retrieved_notes) > 2500:
        retrieved_notes = compress_context(retrieved_notes, max_chars=2000)
        print(f"   ► Context compressed to ~{estimate_token_count(retrieved_notes)} tokens")
    
    return {
        **state,
        "conversation_context": conversation_context,
        "is_question": is_question,
        "retrieved_notes": retrieved_notes,
        "reflection_iterations": 0,  # Initialize loop counter
        "critique": "",
        "stage": "context_loaded"
    }


async def knowledge_extractor(state: AgentState) -> AgentState:
    """
    Extracts entities, categories, and summary from the thought.
    Uses 'retrieved_notes' and 'conversation_context' to improve extraction accuracy.
    """
    # Combine contexts for extraction
    full_context = f"{state.get('conversation_context', '')}\n{state.get('retrieved_notes', '')}"
    
    entities, categories, summary = await extract_all(state['thought'], context=full_context)
    
    # Entity Resolution: Disambiguate similar names against existing graph entities
    if entities:
        existing_entities = knowledge_graph.get_all_entities(limit=100)  # Get existing for comparison
        resolved_entities = batch_resolve_entities(
            [e.to_dict() for e in entities],
            existing_entities,
            context=full_context
        )
        entities = [Entity(name=e["name"], type=e["type"], description=e.get("description", "")) for e in resolved_entities]
    
    # Get deeper context from knowledge graph based on extracted entities
    # This is "2nd hop" retrieval
    entity_names = [e.name for e in entities]
    related_context = knowledge_graph.get_context_for_thought(entity_names)
    
    return {
        **state,
        "entities": [e.to_dict() for e in entities],
        "categories": [{"name": c.name, "confidence": c.confidence} for c in categories],
        "summary": summary,
        "related_context": related_context,
        "stage": "extracted"
    }


async def reflection_node(state: AgentState) -> AgentState:
    """
    The Critic Agent.
    Reviews the draft entities against the context to find missing links.
    """
    entities = [Entity(**e) for e in state['entities']]
    context = f"{state.get('conversation_context', '')}\n{state.get('retrieved_notes', '')}"
    
    critique = await critique_extraction(state['thought'], entities, context)
    
    return {
        **state,
        "critique": critique,
        "reflection_iterations": state['reflection_iterations'] + 1,
        "stage": "reflected"
    }


async def refinement_node(state: AgentState) -> AgentState:
    """
    The Refiner Agent.
    Updates the entities based on the critique.
    """
    entities = [Entity(**e) for e in state['entities']]
    critique = state['critique']
    
    new_entities = await refine_extraction(state['thought'], entities, critique)
    
    # If refining entities changes them, we might want to update related context
    # But for now, we just update the entity list
    return {
        **state,
        "entities": [e.to_dict() for e in new_entities],
        "stage": "refined"
    }


# ============================================================================
# NEW: Enrichment Node
# ============================================================================

async def enrich(state: AgentState):
    """
    Active Enrichment Node.
    Runs 3 specialized agents in parallel:
    1. Latent Intent (Blocker detection)
    2. Social Graph (Connection nudges)
    3. Action Auditor (Task extraction)
    """
    print(f"--- ENRICHMENT AGENTS ---")
    thought_content = state["thought"]
    extracted_entities = state.get("entities", [])
    extracted_topics = [e["name"] for e in extracted_entities if e.get("type") in ["Topic", "Concept", "Technology", "Interest"]]
    
    # 1. Latent Intent
    # Get active projects from graph for context
    active_projects_data = knowledge_graph.get_synthesized_projects()
    active_project_names = [p["name"] for p in active_projects_data]
    intent_data = analyze_intent_and_risk(thought_content, active_project_names)
    
    # 2. Social Graph
    # Get people profiles from graph
    people_profiles = knowledge_graph.get_people()
    nudges = find_social_connections(extracted_topics, people_profiles)
    
    # 3. Action Auditor
    actions = audit_actionability(thought_content)
    
    print(f"   ► Blocker: {intent_data.get('is_blocker')} ({intent_data.get('risk_level')})")
    print(f"   ► Nudges: {len(nudges)}")
    print(f"   ► Actions: {len(actions)}")
    
    return {
        **state,
        "is_blocker": intent_data.get("is_blocker", False),
        "affected_project": intent_data.get("affected_project_name"),
        "nudges": nudges,
        "actions": actions,
        "stage": "enriched"
    }


async def assistant_responder(state: AgentState) -> AgentState:
    """
    Main assistant response node - handles both questions and notes.
    Uses conversation context, retrieved notes, and knowledge graph.
    """
    llm = get_llm()
    
    # Build context sections
    context_parts = []
    
    if state.get('conversation_context'):
        context_parts.append(state['conversation_context'])
    
    if state.get('retrieved_notes'):
        context_parts.append(state['retrieved_notes'])
    
    if state.get('related_context'):
        context_parts.append(state['related_context'])
    
    full_context = "\n\n".join(context_parts) if context_parts else "No previous context available."
    
    # Different prompts for questions vs notes - NOW SOCRATIC
    if state.get('is_question'):
        system_prompt = f"""You are a Socratic thinking partner with access to the user's notes and history.
The user is asking a question. Your job is to:
1. Answer using their stored notes if relevant (cite them specifically)
2. Challenge assumptions in their question - ask "what makes you think X?"
3. Identify potential blind spots or risks they might not see
4. Connect their question to insights from unrelated notes if applicable
5. If you see contradictions with past statements, surface them helpfully

Context available:
{full_context}

Be helpful but NOT passive. Don't just answer - make them think deeper.
If something seems risky or problematic, say so clearly but diplomatically."""
    else:
        system_prompt = f"""You are a Socratic co-cognitive partner, NOT a passive note-taker.
The user is sharing a thought. Your job is to:
1. CHALLENGE: If this is a plan, identify ONE potential risk from their history
2. CONNECT: Link to surprising connections from their past notes
3. CLARIFY: Ask ONE follow-up question that forces them to define terms more clearly
4. SURFACE: If this contradicts something they said before, note it helpfully

Context available:
{full_context}

Do NOT just agree or acknowledge. Add value by making them think.
Keep response to 2-4 sentences but make them count. Be a thinking partner, not a stenographer."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['thought'])
    ]
    
    response = await llm.ainvoke(messages)
    
    return {
        **state,
        "response": response.content,
        "analysis": "Question" if state.get('is_question') else "Note/Thought",
        "insights": state.get('retrieved_notes', '') or state.get('related_context', ''),
        "stage": "responded"
    }


async def knowledge_saver(state: AgentState) -> AgentState:
    """
    Saves the thought to knowledge graph and updates conversation history.
    """
    # Create thought node
    # Create thought node
    thought_node = ThoughtNode(
        id=state['thought_id'],
        content=state['thought'],
        summary=state['summary'] or state['thought'][:100],
        timestamp=datetime.now().isoformat(),
        entities=[Entity(**e) for e in state.get('entities', [])],
        categories=[Category(**c) for c in state.get('categories', [])],
        related_thought_ids=[],
        actions=[ActionItem(**a) for a in state.get("actions", [])],
        nudges=[SocialNudge(**n) for n in state.get("nudges", [])],
        is_blocker=state.get("is_blocker", False),
        affected_project=state.get("affected_project")
    )
    
    # Save to knowledge graph
    knowledge_graph.add_thought(thought_node)
    
    # Index in vector store for semantic search
    vector_store.add_thought(
        thought_id=state['thought_id'],
        content=state['thought'],
        metadata={
            "timestamp": thought_node.timestamp,
            "summary": thought_node.summary,
            "entities": state.get('entities', []),
            "categories": state.get('categories', [])
        }
    )
    
    # Save conversation history
    knowledge_graph.add_conversation_message("user", state['thought'], state['thought_id'])
    knowledge_graph.add_conversation_message("assistant", state['response'], state['thought_id'])
    
    # Generate Serendipity Nudges (Structural Hole Detection)
    entity_names = [e.get("name", "") for e in state.get('entities', [])]
    serendipity_nudges = get_serendipity_nudges(knowledge_graph, entity_names)
    
    # Phase 4.1: Hierarchical Task Decomposition
    task_hierarchy = {}
    if state.get('actions'):  # If actions were detected, check for complex tasks
        decomposition = decompose_task(state['thought'])
        if decomposition.get('is_complex'):
            task_hierarchy = create_task_hierarchy(
                knowledge_graph, 
                state['thought_id'], 
                decomposition
            )
            print(f"   ► Task Decomposition: {task_hierarchy.get('subtask_count', 0)} subtasks created")
    
    # Phase 4.2: Zettelkasten Auto-Atomization (for long-form content)
    atomic_notes_created = []
    if should_atomize(state['thought']):
        atoms = atomize_content(state['thought'])
        if atoms:
            atomic_notes_created = create_atomic_thoughts(
                knowledge_graph,
                state['thought_id'],
                atoms
            )
            print(f"   ► Zettelkasten: {len(atomic_notes_created)} atomic notes created")
    
    return {
        **state,
        "serendipity_nudges": serendipity_nudges,
        "atomic_notes_created": atomic_notes_created,
        "task_hierarchy": task_hierarchy,
        "stage": "saved"
    }


async def synthesis_node(state: AgentState) -> AgentState:
    """
    Run background synthesis agents to organize data.
    Creates profiles for people, projects, and meetings.
    
    NOTE: This is now FIRE-AND-FORGET to reduce user-perceived latency.
    The user gets their response immediately; synthesis happens in background.
    """
    # Fire and forget - don't block on synthesis
    synthesis_coro = run_synthesis_pipeline(
        thought_id=state['thought_id'],
        thought_content=state['thought'],
        entities=state.get('entities', []),
        knowledge_graph=knowledge_graph
    )
    
    # Run in background thread, don't wait
    run_in_background(synthesis_coro, task_id=f"synth_{state['thought_id']}")
    
    return {
        **state,
        "synthesis_status": {"status": "queued", "message": "Running in background"},
        "stage": "complete"
    }


async def simple_responder(state: AgentState) -> AgentState:
    """Handles very simple inputs like greetings."""
    llm = get_llm()
    
    # Get conversation context for continuity
    conv_context = state.get('conversation_context', '')
    
    system_prompt = f"""You are a friendly personal AI assistant. Respond warmly and briefly.
If there's conversation history, acknowledge the ongoing relationship.

{conv_context if conv_context else ''}"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['thought'])
    ]
    
    response = await llm.ainvoke(messages)
    
    return {
        **state,
        "analysis": "Greeting",
        "insights": "",
        "response": response.content,
        "summary": state['thought'][:50],
        "entities": [],
        "categories": [{"name": "Personal", "confidence": 0.5}],
        "stage": "responded"
    }


# ============================================================================
# Routing
# ============================================================================

def should_use_full_pipeline(state: AgentState) -> Literal["full_pipeline", "simple"]:
    """Decide whether to use full processing or simple response."""
    thought = state['thought'].lower().strip()
    simple_patterns = ['hi', 'hello', 'hey', 'thanks', 'thank you', 'ok', 'okay', 'bye', 'goodbye']
    
    if len(thought) < 15 or thought in simple_patterns:
        return "simple"
    return "full_pipeline"


def should_continue_reflection(state: AgentState) -> Literal["refine", "conclude"]:
    """
    Decide if we need to refine the extraction based on the critique.
    Limit recursion to avoid infinite loops.
    
    Enhanced logic:
    1. If critique explicitly suggests improvements, refine
    2. If entity count is suspiciously low, refine
    3. If critique mentions "missing" or "should include", refine
    """
    critique = state['critique'].lower()
    iterations = state['reflection_iterations']
    entity_count = len(state.get('entities', []))
    
    # Hard stop after 2 iterations to prevent infinite loops
    if iterations >= 2:
        return "conclude"
    
    # If critique is generic "looks good", conclude
    if critique.startswith("looks good") and len(critique) < 30:
        return "conclude"
    
    # Quality signals that suggest we should refine
    refinement_signals = [
        "missing", "should include", "could add", "forgot",
        "overlooked", "didn't catch", "also mention", "important",
        "person", "project", "relationship"
    ]
    
    for signal in refinement_signals:
        if signal in critique:
            print(f"   ► Reflection trigger: Found '{signal}' in critique, refining...")
            return "refine"
    
    # If very few entities extracted from a substantive thought, force refinement
    thought_length = len(state['thought'].split())
    if thought_length > 20 and entity_count < 2 and iterations == 1:
        print(f"   ► Reflection trigger: Low entity count ({entity_count}) for thought length ({thought_length})")
        return "refine"
    
    # Default: conclude if nothing obvious to fix
    return "conclude"


# ============================================================================
# Graph Construction
# ============================================================================

def build_graph() -> StateGraph:
    """Builds the LangGraph workflow for the personal assistant."""
    
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("load_context", context_loader)
    graph.add_node("extract", knowledge_extractor)
    graph.add_node("reflect", reflection_node)      # NEW: Critic
    graph.add_node("refine", refinement_node)        # NEW: Refiner
    graph.add_node("enrich", enrich)                 # NEW: Enrichment
    graph.add_node("respond", assistant_responder)
    graph.add_node("save", knowledge_saver)
    graph.add_node("synthesize", synthesis_node)
    graph.add_node("simple", simple_responder)
    
    # Define flow
    graph.add_edge(START, "load_context")
    
    # Route based on complexity
    graph.add_conditional_edges(
        "load_context",
        should_use_full_pipeline,
        {
            "full_pipeline": "extract",
            "simple": "simple"
        }
    )
    
    # Extraction -> Reflection
    graph.add_edge("extract", "reflect")
    
    # Reflection loop
    graph.add_conditional_edges(
        "reflect",
        should_continue_reflection,
        {
            "refine": "refine",
            "conclude": "enrich"  # Redirect to Enrichment instead of Respond
        }
    )
    
    # Refinement goes back to reflection
    graph.add_edge("refine", "reflect")
    
    # Enrichment -> Respond
    graph.add_edge("enrich", "respond")
    
    # Main path continues
    graph.add_edge("respond", "save")
    graph.add_edge("save", "synthesize")
    graph.add_edge("synthesize", END)
    
    # Simple path
    graph.add_edge("simple", "save")
    
    return graph.compile()


# ============================================================================
# Main Interface
# ============================================================================

agent = build_graph()


async def process_thought(thought: str) -> dict:
    """
    Process a user thought through the assistant.
    
    Returns:
        dict with response, analysis, entities, categories, etc.
    """
    thought_id = str(uuid.uuid4())[:8]
    
    initial_state: AgentState = {
        "thought": thought,
        "thought_id": thought_id,
        "is_question": False,
        "retrieved_notes": "",
        "analysis": "",
        "insights": "",
        "response": "",
        "conversation_context": "",
        "messages": [],
        "stage": "start",
        "entities": [],
        "categories": [],
        "summary": "",
        "related_context": "",
        "reflection_iterations": 0,
        "critique": ""
    }
    
    result = await agent.ainvoke(initial_state)
    
    return {
        "thought_id": thought_id,
        "response": result["response"],
        "analysis": result["analysis"],
        "insights": result["insights"],
        "entities": result.get("entities", []),
        "categories": result.get("categories", []),
        "summary": result.get("summary", ""),
        "has_connections": bool(result.get("related_context") or result.get("retrieved_notes")),
        "serendipity_nudges": result.get("serendipity_nudges", []),
        "actions": result.get("actions", []),
        "is_blocker": result.get("is_blocker", False)
    }


# Testing
if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Test 1: Store a note
        print("Storing a note...")
        r1 = await process_thought("My meeting with John is scheduled for Friday at 3pm to discuss the Python project.")
        print(f"Response: {r1['response']}\n")
        
        # Test 2: Ask a question
        print("Asking a question...")
        r2 = await process_thought("When is my meeting with John?")
        print(f"Response: {r2['response']}\n")
    
    asyncio.run(test())
