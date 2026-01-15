"""
People's Agent - FastAPI Backend Server
Serves the LangGraph agent via REST API with knowledge graph capabilities.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json
import uuid
from datetime import datetime

from graph import process_thought, agent, AgentState
from knowledge_graph import knowledge_graph, ThoughtNode
import vector_store
from advanced_agents import (
    decompose_intents, find_serendipitous_connections,
    generate_daily_briefing, atomize_note, feynman_challenge
)
from langchain_core.messages import HumanMessage

# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="People's Agent API",
    description="AI-powered second brain using LangGraph with knowledge graph",
    version="2.0.0"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class ThoughtRequest(BaseModel):
    """Request model for processing a thought."""
    thought: str


class EntityModel(BaseModel):
    """Entity in a thought."""
    name: str
    type: str
    description: str = ""


class CategoryModel(BaseModel):
    """Category of a thought."""
    name: str
    confidence: float = 1.0


class ThoughtResponse(BaseModel):
    """Response model with AI analysis, response, and knowledge graph data."""
    thought_id: str
    response: str
    analysis: str
    insights: str
    entities: List[EntityModel] = []
    categories: List[CategoryModel] = []
    summary: str = ""
    has_connections: bool = False


class GraphNode(BaseModel):
    """Node in the knowledge graph visualization."""
    id: str
    type: str
    label: str
    data: dict = {}


class GraphEdge(BaseModel):
    """Edge in the knowledge graph visualization."""
    source: str
    target: str
    type: str


class GraphResponse(BaseModel):
    """Knowledge graph data for visualization."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    knowledge_graph: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    kg_type = type(knowledge_graph).__name__
    return HealthResponse(
        status="healthy", 
        version="2.0.0",
        knowledge_graph=kg_type
    )


@app.post("/api/think", response_model=ThoughtResponse)
async def think(request: ThoughtRequest):
    """
    Process a user thought through the LangGraph agent.
    Extracts entities, categories, and stores in knowledge graph.
    
    Returns analysis, insights, synthesized response, and extracted knowledge.
    """
    if not request.thought or not request.thought.strip():
        raise HTTPException(status_code=400, detail="Thought cannot be empty")
    
    try:
        result = await process_thought(request.thought.strip())
        return ThoughtResponse(
            thought_id=result.get("thought_id", ""),
            response=result.get("response", ""),
            analysis=result.get("analysis", ""),
            insights=result.get("insights", ""),
            entities=[EntityModel(**e) for e in result.get("entities", [])],
            categories=[CategoryModel(**c) for c in result.get("categories", [])],
            summary=result.get("summary", ""),
            has_connections=result.get("has_connections", False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing thought: {str(e)}")


@app.get("/api/graph", response_model=GraphResponse)
async def get_knowledge_graph():
    """
    Get the full knowledge graph for visualization.
    Returns all thoughts, entities, and their relationships.
    """
    try:
        graph_data = knowledge_graph.get_graph_data()
        return GraphResponse(
            nodes=[GraphNode(**n) for n in graph_data.get("nodes", [])],
            edges=[GraphEdge(**e) for e in graph_data.get("edges", [])]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching graph: {str(e)}")


@app.get("/api/entities")
async def get_entities():
    """Get all entities with their connection counts."""
    try:
        entities = knowledge_graph.get_all_entities()
        return {"entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching entities: {str(e)}")


@app.get("/api/search")
async def search_thoughts(
    entity: Optional[str] = Query(None, description="Search by entity name"),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Search thoughts by entity or category."""
    try:
        if entity:
            thoughts = knowledge_graph.find_by_entity(entity)
        elif category:
            thoughts = knowledge_graph.find_by_category(category)
        else:
            raise HTTPException(status_code=400, detail="Provide 'entity' or 'category' parameter")
        
        # Handle both dict and ThoughtNode results
        results = []
        for t in thoughts:
            if hasattr(t, 'to_dict'):
                results.append(t.to_dict())
            else:
                results.append(t)
        
        return {"thoughts": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")


@app.get("/api/related/{thought_id}")
async def get_related_thoughts(thought_id: str, limit: int = 5):
    """Get thoughts related to a specific thought via shared entities."""
    try:
        related = knowledge_graph.find_related_thoughts(thought_id, limit)
        
        # Handle both dict and ThoughtNode results
        results = []
        for t in related:
            if hasattr(t, 'to_dict'):
                results.append(t.to_dict())
            else:
                results.append(t)
        
        return {"related": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding related: {str(e)}")


# ============================================================================
# Brain World API Endpoints
# ============================================================================

@app.get("/api/brain/insights")
async def get_brain_insights():
    """
    Get organized insights for the Brain World dashboard.
    Returns all smart categories with counts and previews.
    """
    try:
        insights = knowledge_graph.get_brain_insights()
        return {"insights": insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching insights: {str(e)}")


@app.get("/api/brain/category/{category_name}")
async def get_category_items(category_name: str, limit: int = 50):
    """Get all items in a specific brain category."""
    try:
        items = knowledge_graph.get_category_items(category_name, limit)
        return {"category": category_name, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching category: {str(e)}")


@app.get("/api/brain/people")
async def get_people():
    """Get synthesized person profiles with role and relationship."""
    try:
        people = knowledge_graph.get_people()
        return {"people": people}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching people: {str(e)}")


@app.get("/api/brain/projects")
async def get_projects():
    """Get synthesized project profiles."""
    try:
        projects = knowledge_graph.get_synthesized_projects()
        return {"projects": projects}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching projects: {str(e)}")


@app.get("/api/brain/meetings")
async def get_meetings():
    """Get extracted meetings."""
    try:
        meetings = knowledge_graph.get_meetings()
        return {"meetings": meetings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching meetings: {str(e)}")


@app.get("/api/brain/stats")
async def get_brain_stats():
    """Get overall brain statistics."""
    try:
        neo4j_stats = knowledge_graph.get_stats()
        vector_stats = vector_store.get_stats()
        return {
            "stats": {
                **neo4j_stats,
                "vector_documents": vector_stats.get("total_documents", 0)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")


@app.get("/api/brain/search")
async def semantic_search(q: str = Query(..., description="Search query"), limit: int = 10):
    """
    Semantic search across all notes.
    Uses vector embeddings to find semantically similar content.
    """
    try:
        results = vector_store.semantic_search(q, limit=limit)
        return {"query": q, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.get("/api/brain/similar/{thought_id}")
async def find_similar_notes(thought_id: str, limit: int = 5):
    """Find notes similar to a given thought."""
    try:
        similar = vector_store.find_similar(thought_id, limit=limit)
        return {"thought_id": thought_id, "similar": similar}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar: {str(e)}")


# ============================================================================
# Advanced PKM Endpoints
# ============================================================================

class DecomposeRequest(BaseModel):
    content: str


@app.post("/api/brain/decompose")
async def decompose_brain_dump(request: DecomposeRequest):
    """
    Decompose a brain dump into separate intents with PARA routing.
    Zero-taxonomy dropzone.
    """
    try:
        intents = await decompose_intents(request.content)
        return {"content": request.content, "intents": intents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decomposition error: {str(e)}")


@app.get("/api/brain/serendipity")
async def get_serendipitous_connections(focus: str = Query(...), limit: int = 3):
    """
    Find unexpected but valuable connections (Serendipity Engine).
    Discovers structural holes between knowledge clusters.
    """
    try:
        # Get vector search results first
        vector_results = vector_store.semantic_search(focus, limit=10)
        connections = await find_serendipitous_connections(
            current_note=focus,
            all_notes=[],  # Could load from graph
            vector_results=vector_results,
            limit=limit
        )
        return {"focus": focus, "serendipitous_connections": connections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Serendipity error: {str(e)}")


@app.get("/api/brain/briefing")
async def get_daily_briefing():
    """
    Generate personalized daily briefing.
    Summarizes recent work, open tasks, and today's focus.
    """
    try:
        # Get recent context
        recent = knowledge_graph.get_recent_conversation(limit=20)
        recent_thoughts = [{"content": m.content, "timestamp": m.timestamp} for m in recent if m.role == "user"]
        
        # Get meetings
        meetings = knowledge_graph.get_meetings()
        
        briefing = await generate_daily_briefing(
            recent_thoughts=recent_thoughts,
            open_tasks=[],  # Could track tasks
            today_meetings=meetings,
            knowledge_graph=knowledge_graph
        )
        return {"briefing": briefing, "generated_at": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Briefing error: {str(e)}")


class AtomizeRequest(BaseModel):
    content: str


@app.post("/api/brain/atomize")
async def atomize_long_note(request: AtomizeRequest):
    """
    Split a long note into atomic chunks (Auto-Zettelkasten).
    Each chunk contains one distinct idea.
    """
    try:
        chunks = await atomize_note(request.content)
        return {"original_length": len(request.content), "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Atomization error: {str(e)}")


@app.get("/api/brain/feynman")
async def feynman_teaching_mode(topic: str = Query(...)):
    """
    Feynman Technique - challenge user to explain a topic simply.
    Generates questions to test understanding.
    """
    try:
        # Get user's notes on this topic
        search_results = vector_store.semantic_search(topic, limit=3)
        user_notes = [r.get("content", "") for r in search_results]
        
        challenge = await feynman_challenge(topic, user_notes)
        return {"topic": topic, "challenge": challenge}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feynman error: {str(e)}")


@app.post("/api/think/stream")
async def think_stream(request: ThoughtRequest):
    """
    Stream the agent's response in real-time.
    Sends Server-Sent Events (SSE) with processing updates.
    """
    if not request.thought or not request.thought.strip():
        raise HTTPException(status_code=400, detail="Thought cannot be empty")
    
    async def generate():
        try:
            thought_id = str(uuid.uuid4())[:8]
            
            initial_state: AgentState = {
                "thought": request.thought.strip(),
                "thought_id": thought_id,
                "analysis": "",
                "insights": "",
                "response": "",
                "messages": [],
                "stage": "start",
                "entities": [],
                "categories": [],
                "summary": "",
                "related_context": ""
            }
            
            # Stream through graph nodes
            async for event in agent.astream(initial_state):
                for node_name, node_output in event.items():
                    if node_name == "extract":
                        entities = node_output.get("entities", [])
                        categories = node_output.get("categories", [])
                        yield f"data: {json.dumps({'type': 'extraction', 'entities': entities, 'categories': categories})}\n\n"
                    elif node_name == "analyze" and node_output.get("analysis"):
                        yield f"data: {json.dumps({'type': 'analysis', 'content': node_output['analysis']})}\n\n"
                    elif node_name == "generate_insights" and node_output.get("insights"):
                        yield f"data: {json.dumps({'type': 'insights', 'content': node_output['insights']})}\n\n"
                    elif node_name in ["synthesize", "simple_response"] and node_output.get("response"):
                        yield f"data: {json.dumps({'type': 'response', 'content': node_output['response']})}\n\n"
                    elif node_name == "save":
                        yield f"data: {json.dumps({'type': 'saved', 'thought_id': thought_id})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
