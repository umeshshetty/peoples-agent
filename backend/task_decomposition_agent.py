"""
People's Agent - Task Decomposition Agent
Handles hierarchical task decomposition (Parent → Child tasks).
"""

from typing import List, Dict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import json
import re
import uuid


llm = ChatOllama(model="llama3.2:latest", temperature=0.3, base_url="http://localhost:11434")


DECOMPOSITION_PROMPT = """You are a project manager expert. Analyze this task/idea and break it down into subtasks.

Task/Idea:
{content}

If this is a COMPLEX task that needs to be broken down:
- Create 2-5 specific, actionable subtasks
- Each subtask should be clear and achievable

Return a JSON object:
{{
    "is_complex": true/false,
    "parent_task": "Main task title",
    "subtasks": [
        {{"title": "Subtask 1", "description": "Details", "urgency": 1-5}},
        ...
    ]
}}

If the task is simple (doesn't need breakdown), set "is_complex": false and "subtasks": [].
Return ONLY the JSON, no explanation."""


def decompose_task(content: str) -> Dict:
    """
    Analyze content for complex tasks and decompose into hierarchy.
    """
    try:
        response = llm.invoke([
            HumanMessage(content=DECOMPOSITION_PROMPT.format(content=content))
        ])
        
        response_text = response.content.strip()
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        if json_match:
            result = json.loads(json_match.group(0))
            return result
        
        return {"is_complex": False, "parent_task": "", "subtasks": []}
        
    except Exception as e:
        print(f"Error decomposing task: {e}")
        return {"is_complex": False, "parent_task": "", "subtasks": []}


def create_task_hierarchy(knowledge_graph, thought_id: str, decomposition: Dict) -> Dict:
    """
    Create task nodes in Neo4j with parent-child relationships.
    
    Creates:
    - ParentTask node linked to original Thought
    - ChildTask nodes linked to ParentTask via HAS_SUBTASK
    """
    if not decomposition.get("is_complex") or not decomposition.get("subtasks"):
        return {"created": False}
    
    parent_id = f"task_{uuid.uuid4().hex[:8]}"
    child_ids = []
    
    with knowledge_graph.driver.session() as session:
        # Create Parent Task
        session.run("""
            CREATE (p:Task {
                id: $id,
                title: $title,
                type: 'parent',
                status: 'pending'
            })
            WITH p
            MATCH (t:Thought {id: $thought_id})
            MERGE (t)-[:HAS_TASK]->(p)
        """, {
            "id": parent_id,
            "title": decomposition.get("parent_task", "Untitled Task"),
            "thought_id": thought_id
        })
        
        # Create Child Tasks
        for subtask in decomposition.get("subtasks", []):
            child_id = f"task_{uuid.uuid4().hex[:8]}"
            child_ids.append(child_id)
            
            session.run("""
                CREATE (c:Task {
                    id: $id,
                    title: $title,
                    description: $description,
                    urgency: $urgency,
                    type: 'child',
                    status: 'pending'
                })
                WITH c
                MATCH (p:Task {id: $parent_id})
                MERGE (p)-[:HAS_SUBTASK]->(c)
            """, {
                "id": child_id,
                "title": subtask.get("title", ""),
                "description": subtask.get("description", ""),
                "urgency": subtask.get("urgency", 3),
                "parent_id": parent_id
            })
    
    return {
        "created": True,
        "parent_id": parent_id,
        "child_ids": child_ids,
        "subtask_count": len(child_ids)
    }
