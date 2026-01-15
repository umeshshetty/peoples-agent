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
    knowledge_graph, ThoughtNode, Entity, Category
)
from extraction_agents import extract_all, find_relationship_context
from synthesis_agents import run_synthesis_pipeline
from classification_agents import run_classification_pipeline
import vector_store

# Configuration
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
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
    Loads conversation context and checks if this is a question.
    Also retrieves relevant notes if the user is asking a question.
    """
    thought = state['thought']
    
    # Get conversation context
    conversation_context = knowledge_graph.get_conversation_context(limit=6)
    
    # Detect if this is a question (simple heuristic)
    is_question = (
        '?' in thought or
        thought.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who', 'can', 'could', 'would', 'should', 'do', 'does', 'is', 'are', 'tell me', 'explain', 'help me'))
    )
    
    # If it's a question, search stored notes (vector + graph)
    retrieved_notes = ""
    if is_question:
        # Try semantic search first
        semantic_results = vector_store.get_context_for_query(thought, limit=3)
        if semantic_results:
            retrieved_notes = semantic_results + "\n"
        
        # Also search knowledge graph
        relevant_thoughts = knowledge_graph.search_notes(thought, limit=3)
        if relevant_thoughts:
            retrieved_notes += "\nFrom your knowledge graph:\n"
            for t in relevant_thoughts:
                retrieved_notes += f"- [{t.timestamp[:10]}] {t.content[:200]}...\n"
    
    return {
        **state,
        "conversation_context": conversation_context,
        "is_question": is_question,
        "retrieved_notes": retrieved_notes,
        "stage": "context_loaded"
    }


async def knowledge_extractor(state: AgentState) -> AgentState:
    """
    Extracts entities, categories, and summary from the thought.
    Also retrieves context from related past thoughts.
    """
    entities, categories, summary = await extract_all(state['thought'])
    
    # Get context from knowledge graph based on extracted entities
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
    
    # Different prompts for questions vs notes
    if state.get('is_question'):
        system_prompt = f"""You are a personal AI assistant with access to the user's notes and conversation history.
The user is asking a question. Your job is to:
1. FIRST check if there's relevant information in their stored notes
2. Answer based on what you find in their notes if available
3. If no relevant notes, provide helpful guidance
4. Reference specific notes when answering (e.g., "In your note from X, you mentioned...")
5. Be conversational and remember the ongoing conversation

Context available:
{full_context}

Important: If the user's notes contain relevant information, ALWAYS reference it in your answer.
If you don't have relevant information from their notes, say so honestly and offer to help them think through it."""
    else:
        system_prompt = f"""You are a personal AI assistant and second brain. The user is sharing a thought/note with you.
Your job is to:
1. Acknowledge what they've shared
2. Make connections to their previous notes if relevant
3. Offer insights or perspectives that add value
4. Remember this conversation context for future reference
5. Be warm, helpful, and conversational

Context available:
{full_context}

Keep your response natural and conversational (2-4 sentences). Don't be overly formal."""

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
    thought_node = ThoughtNode(
        id=state['thought_id'],
        content=state['thought'],
        summary=state['summary'] or state['thought'][:100],
        timestamp=datetime.now().isoformat(),
        entities=[Entity(**e) for e in state.get('entities', [])],
        categories=[Category(**c) for c in state.get('categories', [])]
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
    
    return {
        **state,
        "stage": "saved"
    }


async def synthesis_node(state: AgentState) -> AgentState:
    """
    Run background synthesis agents to organize data.
    Creates profiles for people, projects, and meetings.
    """
    try:
        result = await run_synthesis_pipeline(
            thought_id=state['thought_id'],
            thought_content=state['thought'],
            entities=state.get('entities', []),
            knowledge_graph=knowledge_graph
        )
        return {
            **state,
            "synthesis_status": result.get('status', {}),
            "stage": "complete"
        }
    except Exception as e:
        print(f"Synthesis error: {e}")
        return {
            **state,
            "synthesis_status": {"error": str(e)},
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


# ============================================================================
# Graph Construction
# ============================================================================

def build_graph() -> StateGraph:
    """Builds the LangGraph workflow for the personal assistant."""
    
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("load_context", context_loader)
    graph.add_node("extract", knowledge_extractor)
    graph.add_node("respond", assistant_responder)
    graph.add_node("save", knowledge_saver)
    graph.add_node("synthesize", synthesis_node)  # Background synthesis
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
    
    # Full pipeline: extract -> respond -> save -> synthesize
    graph.add_edge("extract", "respond")
    graph.add_edge("respond", "save")
    graph.add_edge("save", "synthesize")
    graph.add_edge("synthesize", END)
    
    # Simple path also saves and synthesizes
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
        "related_context": ""
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
        "has_connections": bool(result.get("related_context") or result.get("retrieved_notes"))
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
        
        # Test 3: Follow-up
        print("Follow-up...")
        r3 = await process_thought("What was the meeting about again?")
        print(f"Response: {r3['response']}\n")
    
    asyncio.run(test())
