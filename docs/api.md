# API Reference 🔌

People's Agent exposes a RESTful API for all its capabilities.

## Base URL
`http://localhost:8000`

## Core Endpoints

### Process Thought
**POST** `/api/think`
- **Body**: `{"thought": "string"}`
- **Description**: Main entry point. Processes a thought through the entire agentic pipeline (Extract -> Respond -> Save -> Synthesize).

### Semantic Search
**GET** `/api/brain/search`
- **Query**: `q` (string), `limit` (int)
- **Description**: Search for notes using vector embeddings.

### Find Similar
**GET** `/api/brain/similar/{thought_id}`
- **Description**: Find notes semantically similar to a specific thought ID.

### Brain Insights
**GET** `/api/brain/insights`
- **Description**: Get aggregated counts and top items for all categories (People, Projects, etc.).

## Advanced Agent Endpoints

### Daily Briefing
**GET** `/api/brain/briefing`
- **Description**: Generate a personalized daily briefing based on recent context and meetings.

### Decompose Intent
**POST** `/api/brain/decompose`
- **Body**: `{"content": "string"}`
- **Description**: Decompose a multi-intent brain dump into atomic intents with PARA classification.

### Serendipity
**GET** `/api/brain/serendipity`
- **Query**: `focus` (string)
- **Description**: Find unexpected connections related to the focus topic.

### Feynman Challenge
**GET** `/api/brain/feynman`
- **Query**: `topic` (string)
- **Description**: Start a Feynman Technique learning session on a topic.

### Auto-Atomize
**POST** `/api/brain/atomize`
- **Body**: `{"content": "string"}`
- **Description**: Split a long text into atomic notes.
