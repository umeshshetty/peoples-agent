# Development Guide 💻

This guide covers the end-to-end development lifecycle of People's Agent, from initial setup to extending the AI capabilities.

---

## 1. System Architecture Overview

The system is composed of three main layers:
1.  **Frontend Layer**: Vanilla JS (`index.html`, `app.js`). Lightweight, communicating via REST.
2.  **API Layer**: FastAPI (`server.py`). Handles routing, SSE (streaming), and request validation.
3.  **Intelligence Layer**:
    *   **Orchestrator**: LangGraph (`graph.py`). Manages state and decision tools.
    *   **Memory**: Neo4j (Graph) + ChromaDB (Vector).
    *   **Inference**: Ollama (Llama 3.3 70B).

---

## 2. Environment Setup

### 2.1 Prerequisites
*   **Docker Desktop**: Required for running Neo4j.
*   **Python 3.9+**: The core backend language.
*   **Ollama**: For local LLM inference.
*   **Node.js**: (Optional) Only if you plan to introduce a JS build pipeline.

### 2.2 Database Initialization (Neo4j)
We use a Dockerized Neo4j instance.

1.  **Start the container**:
    ```bash
    docker-compose up -d
    ```
    *   This starts Neo4j on `localhost:7474` (Browser) and `localhost:7687` (Bolt).
    *   It mounts a volume `./neo4j_data` for persistence.

2.  **Verify Access**:
    *   Go to `http://localhost:7474`
    *   Login with `neo4j` / `peoplesagent123` (configured in `docker-compose.yml`).

### 2.3 Vector Store Initialization (ChromaDB)
ChromaDB runs in-process (embedded). No separate server is needed.
*   Data is stored in `~/.peoples_agent/chroma_db`.
*   Embedding model (`all-MiniLM-L6-v2`) is downloaded automatically on first run.

### 2.4 LLM Setup
1.  **Install Ollama**: [ollama.com](https://ollama.com)
2.  **Pull Model**:
    ```bash
    ollama pull llama3.3:70b
    ```
3.  **Verify**:
    ```bash
    curl http://localhost:11434/api/generate -d '{"model": "llama3.3:70b", "prompt": "Hi"}'
    ```

---

## 3. Backend Development

### 3.1 Installation
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3.2 Project Structure
*   `server.py`: **Entry Point**. FastAPI app definition.
*   `graph.py`: **The Brain**. LangGraph workflow definition (Nodes & Edges).
*   `knowledge_graph.py`: **Data Access Object (DAO)**. Wraps Neo4j queries.
*   `vector_store.py`: **Vector Logic**. Embeddings and similarity search.
*   `synthesis_agents.py`: **Background Workers**. Profile generation logic.
*   `advanced_agents.py`: **Specialists**. Decomposition, Serendipity, Feynman logic.

### 3.3 Running the Server
```bash
uvicorn server:app --reload --port 8000
```
*   `--reload`: Auto-restarts on code changes.
*   The server automatically initializes connections (Neo4j, Chroma) on startup.

---

## 4. Extending the System

### Case Study: Adding a "Mood Tracker" Agent

**Step 1: Define the Agent Logic**
Create a new function in `backend/synthesis_agents.py`:
```python
async def analyze_mood(thought_text: str):
    # Call LLM to extract emotion
    response = await llm.ainvoke(f"Extract mood from: {thought_text}")
    return response
```

**Step 2: Update the Graph Schema**
In `backend/knowledge_graph.py`, add a method to store mood:
```python
def add_mood_entry(self, mood: str, timestamp: str):
    query = "CREATE (m:Mood {emotion: $mood, time: $time})"
    self.query(query, {"mood": mood, "time": timestamp})
```

**Step 3: Hook into Workflow**
In `backend/graph.py`, inside the `run_synthesis_pipeline` function, add:
```python
# existing synthesis calls...
mood = await analyze_mood(thought.content)
kg.add_mood_entry(mood, thought.timestamp)
```

**Step 4: Expose to UI**
Add an endpoint in `backend/server.py`:
```python
@app.get("/api/brain/mood")
def get_mood_history():
    return kg.get_moods()
```

---

## 5. Troubleshooting

**Common Issues:**
*   **Neo4j Connection Refused**: Ensure Docker container is running (`docker ps`).
*   **Ollama Connection Error**: Ensure Ollama is serving (`ollama serve`).
*   **Import Errors**: Ensure you activated the venv (`source venv/bin/activate`).

## 6. Deployment
To deploy to production:
1.  **Database**: Use a managed Neo4j AuraDB instance or a persistent cloud VM.
2.  **LLM**: Switch `OLLAMA_BASE_URL` to a remote Ollama instance or use OpenAI API (via LangChain drop-in replacement).
3.  **Server**: Use a production server like Gunicorn behind Nginx.
    ```bash
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app
    ```
