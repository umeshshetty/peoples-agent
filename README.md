# People's Agent 🧠

**AI-Powered Second Brain** - A personal knowledge management system that stores, organizes, and synthesizes your thoughts using LLMs, knowledge graphs, and vector search.

## Features

### Core
- 💬 **Conversational Interface** - Chat with your second brain
- 🔗 **Knowledge Graph** - Neo4j stores entities and relationships
- 🔍 **Semantic Search** - ChromaDB vector embeddings
- 🏷️ **Auto-categorization** - PARA method classification

### Synthesis Agents
- 👤 **Person Profiler** - Extracts role, relationship, topics
- 📁 **Project Synthesizer** - Tracks status, deadlines, team
- 📅 **Meeting Organizer** - Extracts meeting details
- ✂️ **Auto-Atomization** - Splits long notes into atomic chunks

### Advanced Features
- 🔮 **Serendipity Engine** - Finds unexpected connections
- 📋 **Daily Briefing** - Personalized morning summary
- 🎓 **Feynman Mode** - Teaching/learning challenges
- 🎯 **Multi-Intent Decomposition** - Routes brain dumps to categories

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + LangGraph |
| LLM | Ollama (llama3.2) |
| Graph DB | Neo4j |
| Vector DB | ChromaDB |
| Frontend | Vanilla JS |

## Quick Start

```bash
# 1. Start Neo4j
docker start peoples-agent-neo4j
# Or: docker run -d --name peoples-agent-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/peoplesagent123 neo4j:5-community

# 2. Start Ollama
ollama serve

# 3. Install dependencies
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Run server
uvicorn server:app --reload --port 8000

# 5. Open frontend
open index.html
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/think` | Process a thought |
| `GET /api/brain/insights` | Brain World categories |
| `GET /api/brain/people` | Synthesized person profiles |
| `GET /api/brain/search?q=...` | Semantic search |
| `GET /api/brain/briefing` | Daily briefing |
| `GET /api/brain/serendipity?focus=...` | Unexpected connections |
| `POST /api/brain/decompose` | Multi-intent decomposition |
| `GET /api/brain/feynman?topic=...` | Feynman teaching mode |

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Frontend                       │
│         (index.html, app.js, styles.css)        │
└─────────────────────────────────────────────────┘
                       │
┌─────────────────────────────────────────────────┐
│              FastAPI Backend                     │
│                 server.py                        │
└─────────────────────────────────────────────────┘
                       │
┌─────────────────────────────────────────────────┐
│              LangGraph Workflow                  │
│                  graph.py                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │Context  │→│Extract  │→│Respond  │→│Synthesize│
│  │Loader   │ │Entities │ │         │ │        │ │
│  └─────────┘ └─────────┘ └─────────┘ └────────┘ │
└─────────────────────────────────────────────────┘
                       │
     ┌─────────────────┼─────────────────┐
     ▼                 ▼                 ▼
┌─────────┐       ┌─────────┐       ┌─────────┐
│  Neo4j  │       │ChromaDB │       │  Ollama │
│ (Graph) │       │(Vectors)│       │  (LLM)  │
└─────────┘       └─────────┘       └─────────┘
```

## License

MIT
