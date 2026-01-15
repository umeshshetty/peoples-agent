# Technical Architecture 🏗️

People's Agent uses a modern, retrieval-augmented agent architecture derived from the "Second Brain" cognitive framework.

## System Diagram

```mermaid
graph TD
    User[User] -->|Input| Frontend[Frontend (Vanilla JS)]
    Frontend -->|REST API| Server[FastAPI Server]
    
    subgraph "Agentic Core (LangGraph)"
        Server -->|Thought| Graph[State Graph]
        Graph -->|Context| ContextLoader
        Graph -->|Extract| Extractor
        Graph -->|Synthesize| SynthesisAgents
    end
    
    subgraph "Memory Systems"
        ContextLoader <-->|Semantic Search| ChromaDB[(ChromaDB Vector Store)]
        ContextLoader <-->|Structural Query| Neo4j[(Neo4j Knowledge Graph)]
        SynthesisAgents -->|Write Profiles| Neo4j
        Extractor -->|Index Notes| ChromaDB
    end
    
    subgraph "Intelligence"
        Graph <-->|Inference| Ollama[Ollama (Llama 3.2)]
    end
```

## Key Components

### 1. Hybrid Memory System
We typically see two types of RAG (Retrieval Augmented Generation):
- **Vector RAG**: Good for finding "similar" text. (Tool: **ChromaDB**)
- **Graph RAG**: Good for traversing relationships (A->B->C). (Tool: **Neo4j**)

People's Agent uses **both**. 
- When you ask "Who is John?", we query the Graph for the `Person` node.
- When you ask "What do I know about leadership?", we query the Vector DB for semantic matches.

### 2. LangGraph Workflow
The brain is modeled as a state machine using **LangGraph**:
1. **Load Context**: Retrieve relevant memories.
2. **Classify**: Is this a task? A question? A project update?
3. **Decompose**: Split multi-intent inputs.
4. **Respond**: Generate a helpful answer.
5. **Synthesize**: (Background) Update profiles, extracting tasks, finding connections.

### 3. Synthesis Pipeline
Running in the background (asynchronously), specialized agents organize data:
- `PersonProfiler`: Updates `PersonProfile` nodes in Neo4j.
- `ProjectSynthesizer`: Updates `ProjectProfile` nodes.
- `SerendipityEngine`: Scans for structural holes and unexpected connections.

## Data Model

### Neo4j Schema
- **Nodes**: `Thought`, `Entity` (Person, Project, Tool, etc.), `Category`, `Meeting`
- **Edges**: `MENTIONS`, `BELONGS_TO`, `HAS_PROFILE`, `HAS_MEETING`

### Vector Schema
- **Document**: The raw note content.
- **Metadata**: Timestamp, extracted entities, tags.
