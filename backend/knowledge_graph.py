"""
People's Agent - Neo4j Knowledge Graph Module
Stores all thoughts, entities, and relationships in Neo4j graph database.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import os
import json
from pathlib import Path

# Neo4j imports
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠ neo4j package not installed. Run: pip install neo4j")

# Configuration - Neo4j Docker container
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "peoplesagent123")

# Fallback file storage
DATA_DIR = Path(os.getenv("PEOPLES_AGENT_DATA", Path.home() / ".peoples_agent"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATION_FILE = DATA_DIR / "conversations.json"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    type: str
    description: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Category:
    """Represents a thought category."""
    name: str
    confidence: float = 1.0


@dataclass
class ActionItem:
    """Represents an actionable task extracted from a thought."""
    description: str
    urgency: int = 1
    status: str = "pending"
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SocialNudge:
    """Represents a suggested social connection."""
    person_name: str
    reason: str
    suggestion: str
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class ThoughtNode:
    """Represents a thought in the knowledge graph."""
    id: str
    content: str
    summary: str
    timestamp: str
    entities: List[Entity] = field(default_factory=list)
    categories: List[Category] = field(default_factory=list)
    related_thought_ids: List[str] = field(default_factory=list)
    # Enrichment fields
    is_blocker: bool = False
    affected_project: Optional[str] = None
    actions: List[ActionItem] = field(default_factory=list)
    nudges: List[SocialNudge] = field(default_factory=list)
    # Spaced Repetition fields
    review_count: int = 0
    last_reviewed: Optional[str] = None
    ease_factor: float = 2.5  # SM-2 default
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "entities": [e.to_dict() for e in self.entities],
            "categories": [{"name": c.name, "confidence": c.confidence} for c in self.categories],
            "related_thought_ids": self.related_thought_ids,
            "actions": [a.to_dict() for a in self.actions],
            "nudges": [n.to_dict() for n in self.nudges],
            "is_blocker": self.is_blocker,
            "affected_project": self.affected_project,
            "review_count": self.review_count,
            "last_reviewed": self.last_reviewed,
            "ease_factor": self.ease_factor
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ThoughtNode":
        return cls(
            id=data["id"],
            content=data["content"],
            summary=data["summary"],
            timestamp=data["timestamp"],
            entities=[Entity(**e) for e in data.get("entities", [])],
            categories=[Category(**c) for c in data.get("categories", [])],
            related_thought_ids=data.get("related_thought_ids", []),
            actions=[ActionItem(**a) for a in data.get("actions", [])],
            nudges=[SocialNudge(**n) for n in data.get("nudges", [])],
            is_blocker=data.get("is_blocker", False),
            affected_project=data.get("affected_project"),
            review_count=data.get("review_count", 0),
            last_reviewed=data.get("last_reviewed"),
            ease_factor=data.get("ease_factor", 2.5)
        )


@dataclass
class ConversationMessage:
    """A message in conversation history."""
    role: str
    content: str
    timestamp: str
    thought_id: Optional[str] = None


# ============================================================================
# Neo4j Graph Implementation
# ============================================================================

class Neo4jKnowledgeGraph:
    """
    Full Neo4j implementation for the knowledge graph.
    Stores all thoughts, entities, and relationships in Neo4j.
    """
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.conversation_history: List[ConversationMessage] = []
        self._init_schema()
        self._load_conversations()
        print(f"✓ Connected to Neo4j at {uri}")
    
    def close(self):
        self.driver.close()
    
    def _init_schema(self):
        """Initialize Neo4j schema with constraints and indexes."""
        with self.driver.session() as session:
            # Create constraints for uniqueness
            session.run("CREATE CONSTRAINT thought_id IF NOT EXISTS FOR (t:Thought) REQUIRE t.id IS UNIQUE")
            session.run("CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE")
            session.run("CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")
            # Create indexes for faster lookups
            session.run("CREATE INDEX thought_timestamp IF NOT EXISTS FOR (t:Thought) ON (t.timestamp)")
    
    def _load_conversations(self):
        """Load conversation history from file (for efficiency)."""
        if CONVERSATION_FILE.exists():
            try:
                with open(CONVERSATION_FILE, 'r') as f:
                    data = json.load(f)
                self.conversation_history = [
                    ConversationMessage(**msg) for msg in data.get("messages", [])
                ]
            except Exception as e:
                print(f"⚠ Error loading conversations: {e}")
    
    def _save_conversations(self):
        """Save conversation history to file."""
        try:
            data = {
                "messages": [
                    {"role": m.role, "content": m.content, "timestamp": m.timestamp, "thought_id": m.thought_id}
                    for m in self.conversation_history[-100:]
                ],
                "saved_at": datetime.now().isoformat()
            }
            with open(CONVERSATION_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠ Error saving conversations: {e}")
    
    # ========================================================================
    # Core CRUD Operations
    # ========================================================================
    
    def add_thought(self, thought: ThoughtNode) -> None:
        """Add a thought with its entities and categories to Neo4j."""
        with self.driver.session() as session:
            # Create thought node
            session.run("""
                MERGE (t:Thought {id: $id})
                SET t.content = $content,
                    t.summary = $summary,
                    t.timestamp = $timestamp,
                    t.review_count = coalesce(t.review_count, $review_count),
                    t.last_reviewed = coalesce(t.last_reviewed, $last_reviewed),
                    t.ease_factor = coalesce(t.ease_factor, $ease_factor),
                    t.is_blocker = $is_blocker,
                    t.affected_project = $affected_project
            """, {"id": thought.id, "content": thought.content, 
                  "summary": thought.summary, "timestamp": thought.timestamp,
                  "review_count": thought.review_count,
                  "last_reviewed": thought.last_reviewed,
                  "ease_factor": thought.ease_factor,
                  "is_blocker": thought.is_blocker,
                  "affected_project": thought.affected_project})
            
            # Create entities and relationships
            for entity in thought.entities:
                entity_key = f"{entity.type}:{entity.name}".lower()
                session.run("""
                    MERGE (e:Entity {key: $key})
                    SET e.name = $name, e.type = $type, e.description = $description
                    WITH e
                    MATCH (t:Thought {id: $thought_id})
                    MERGE (t)-[:MENTIONS]->(e)
                """, {"key": entity_key, "name": entity.name, "type": entity.type,
                      "description": entity.description, "thought_id": thought.id})
            
            # Create categories and relationships
            for category in thought.categories:
                session.run("""
                    MERGE (c:Category {name: $name})
                    WITH c
                    MATCH (t:Thought {id: $thought_id})
                    MERGE (t)-[:BELONGS_TO {confidence: $confidence}]->(c)
                """, {"name": category.name, "thought_id": thought.id,
                      "confidence": category.confidence})
            
            # Create Action Items
            for action in thought.actions:
                session.run("""
                    MERGE (a:ActionItem {description: $desc, thought_id: $tid})
                    SET a.urgency = $urgency, a.status = $status
                    WITH a
                    MATCH (t:Thought {id: $tid})
                    MERGE (t)-[:IMPLIES]->(a)
                """, {"desc": action.description, "tid": thought.id, 
                      "urgency": action.urgency, "status": action.status})
                      
            # Handle Project Risks
            if thought.is_blocker and thought.affected_project:
                session.run("""
                    MATCH (p:Entity {name: $proj_name})
                    WHERE p.type IN ['Project', 'Product', 'Tool']
                    SET p.status = 'at_risk', p.risk_level = 'high'
                    MERGE (t:Thought {id: $tid})
                    MERGE (t)-[:BLOCKS]->(p)
                """, {"proj_name": thought.affected_project, "tid": thought.id})
    
    def add_conversation_message(self, role: str, content: str, thought_id: str = None):
        """Add a message to conversation history."""
        msg = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            thought_id=thought_id
        )
        self.conversation_history.append(msg)
        self._save_conversations()
    
    def get_recent_conversation(self, limit: int = 10) -> List[ConversationMessage]:
        return self.conversation_history[-limit:]
    
    def get_conversation_context(self, limit: int = 5) -> str:
        recent = self.get_recent_conversation(limit)
        if not recent:
            return ""
        context_parts = ["Recent conversation:"]
        for msg in recent:
            role = "You" if msg.role == "user" else "Assistant"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            context_parts.append(f"{role}: {content}")
        return "\n".join(context_parts)
    
    # ========================================================================
    # Search and Retrieval
    # ========================================================================
    
    def search_notes(self, query: str, limit: int = 5) -> List[ThoughtNode]:
        """Search thoughts by content."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Thought)
                WHERE toLower(t.content) CONTAINS toLower($query)
                   OR toLower(t.summary) CONTAINS toLower($query)
                OPTIONAL MATCH (t)-[:MENTIONS]->(e:Entity)
                OPTIONAL MATCH (t)-[:BELONGS_TO]->(c:Category)
                WITH t, collect(DISTINCT {name: e.name, type: e.type, description: e.description}) as entities,
                     collect(DISTINCT {name: c.name, confidence: 1.0}) as categories
                ORDER BY t.timestamp DESC
                LIMIT $limit
                RETURN t, entities, categories
            """, {"query": query, "limit": limit})
            
            thoughts = []
            for record in result:
                t = record["t"]
                entities = [Entity(**e) for e in record["entities"] if e["name"]]
                categories = [Category(**c) for c in record["categories"] if c["name"]]
                thoughts.append(ThoughtNode(
                    id=t["id"], content=t["content"], summary=t["summary"],
                    timestamp=t["timestamp"], entities=entities, categories=categories
                ))
            return thoughts
    
    def find_by_entity(self, entity_name: str) -> List[ThoughtNode]:
        """Find thoughts mentioning an entity."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Thought)-[:MENTIONS]->(e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($name)
                RETURN t {.id, .content, .summary, .timestamp}
            """, {"name": entity_name})  # Pass as dict
            return [ThoughtNode(entities=[], categories=[], **record["t"]) for record in result]
    
    def find_by_category(self, category: str) -> List[ThoughtNode]:
        """Find thoughts in a category."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Thought)-[:BELONGS_TO]->(c:Category {name: $category})
                RETURN t {.id, .content, .summary, .timestamp}
            """, {"category": category})  # Pass as dict
            return [ThoughtNode(entities=[], categories=[], **record["t"]) for record in result]
    
    def find_related_thoughts(self, thought_id: str, limit: int = 5) -> List[ThoughtNode]:
        """Find thoughts connected via shared entities."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t1:Thought {id: $id})-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(t2:Thought)
                WHERE t1 <> t2
                WITH t2, count(e) as shared
                ORDER BY shared DESC
                LIMIT $limit
                RETURN t2 {.id, .content, .summary, .timestamp}
            """, {"id": thought_id, "limit": limit})  # Pass as dict
            return [ThoughtNode(entities=[], categories=[], **record["t2"]) for record in result]
    
    def get_context_for_thought(self, entity_names: List[str], limit: int = 5) -> str:
        """Get context from related thoughts."""
        if not entity_names:
            return ""
        with self.driver.session() as session:
            result = session.run("""
                UNWIND $entities as entity_name
                MATCH (t:Thought)-[:MENTIONS]->(e:Entity)
                WHERE toLower(e.name) CONTAINS toLower(entity_name)
                RETURN DISTINCT t.content as content
                LIMIT $limit
            """, {"entities": entity_names, "limit": limit})  # Pass as dict
            contents = [record["content"] for record in result]
            if not contents:
                return ""
            return "Your related notes:\n" + "\n".join(f"- {c[:200]}..." if len(c) > 200 else f"- {c}" for c in contents)
    
    # ========================================================================
    # Spaced Repetition / Smart Re-surface
    # ========================================================================
    
    def get_resurface_queue(self, limit: int = 5) -> List[ThoughtNode]:
        """
        Get thoughts due for review based on Spaced Repetition (SM-2 simplified).
        Logic: 
        - New thoughts (review_count=0) that are > 1 day old
        - Reviewed thoughts where interval has passed
        """
        with self.driver.session() as session:
            # Simple heuristic for now: 
            # 1. Keep it simple: Find items not reviewed recently matching rough intervals
            # 2. Sort by review priority
            result = session.run("""
                MATCH (t:Thought)
                WHERE t.review_count IS NULL OR t.review_count = 0
                   OR (
                       t.review_count > 0 
                       AND datetime(t.last_reviewed) + duration({days: toInteger(t.ease_factor ^ t.review_count)}) < datetime()
                   )
                RETURN t
                ORDER BY t.timestamp ASC
                LIMIT $limit
            """, {"limit": limit})
            
            thoughts = []
            for record in result:
                t = record["t"]
                # Fetch related info lazily or update query to fetch all
                # For queue, we just need basic info usually
                thoughts.append(ThoughtNode(
                    id=t["id"], content=t["content"], summary=t["summary"],
                    timestamp=t["timestamp"],
                    review_count=t.get("review_count", 0),
                    last_reviewed=t.get("last_reviewed"),
                    ease_factor=t.get("ease_factor", 2.5)
                ))
            return thoughts

    def mark_as_reviewed(self, thought_id: str, difficulty: str) -> None:
        """
        Update review stats using SM-2 algorithm principles.
        Difficulty: 'easy', 'medium', 'hard'
        """
        # Map difficulty to quality score (0-5)
        quality_map = {'easy': 5, 'medium': 4, 'hard': 3}
        quality = quality_map.get(difficulty, 4)
        
        with self.driver.session() as session:
            # Fetch current stats
            result = session.run("""
                MATCH (t:Thought {id: $id})
                RETURN t.review_count as count, t.ease_factor as ef
            """, {"id": thought_id})
            record = result.single()
            
            if not record:
                return
                
            current_ef = record["ef"] if record["ef"] else 2.5
            current_count = record["count"] if record["count"] else 0
            
            # SM-2 Algorithm update for Ease Factor
            # EF' = EF + (0.1 - (5-q)*(0.08 + (5-q)*0.02))
            # Review Count increments
            
            new_ef = current_ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
            if new_ef < 1.3: new_ef = 1.3
            
            session.run("""
                MATCH (t:Thought {id: $id})
                SET t.review_count = $count,
                    t.ease_factor = $ef,
                    t.last_reviewed = $now
            """, {
                "id": thought_id,
                "count": current_count + 1,
                "ef": new_ef,
                "now": datetime.now().isoformat()
            })
    
    # ========================================================================
    # Brain World - Insights
    # ========================================================================
    
    def get_brain_insights(self) -> Dict:
        """Get organized insights for Brain World dashboard."""
        insights = {
            "people": {"icon": "👤", "label": "People", "items": [], "count": 0},
            "projects": {"icon": "📁", "label": "Projects", "items": [], "count": 0},
            "meetings": {"icon": "📅", "label": "Meetings", "items": [], "count": 0},
            "urgent": {"icon": "🔴", "label": "Urgent", "items": [], "count": 0},
            "ideas": {"icon": "💡", "label": "Ideas", "items": [], "count": 0},
            "tasks": {"icon": "✅", "label": "Tasks", "items": [], "count": 0},
            "learning": {"icon": "📚", "label": "Learning", "items": [], "count": 0},
            "reflections": {"icon": "💭", "label": "Reflections", "items": [], "count": 0},
        }
        
        urgent_kw = ['urgent', 'asap', 'deadline', 'immediately', 'critical', 'priority', 'today', 'tomorrow']
        meeting_kw = ['meeting', 'call', 'sync', 'standup', 'discussion', 'appointment']
        
        with self.driver.session() as session:
            # Get all thoughts with entities and categories
            result = session.run("""
                MATCH (t:Thought)
                OPTIONAL MATCH (t)-[:MENTIONS]->(e:Entity)
                OPTIONAL MATCH (t)-[:BELONGS_TO]->(c:Category)
                WITH t, collect(DISTINCT {name: e.name, type: e.type, description: e.description}) as entities,
                     collect(DISTINCT c.name) as categories
                ORDER BY t.timestamp DESC
                RETURN t, entities, categories
            """)
            
            for record in result:
                t = record["t"]
                entities = [e for e in record["entities"] if e["name"]]
                categories = record["categories"]
                content_lower = t["content"].lower()
                
                thought_data = {
                    "id": t["id"],
                    "summary": t["summary"],
                    "content": t["content"][:150] + "..." if len(t["content"]) > 150 else t["content"],
                    "timestamp": t["timestamp"],
                    "entities": entities
                }
                
                # Categorize by urgency
                if any(kw in content_lower for kw in urgent_kw):
                    insights["urgent"]["items"].append(thought_data)
                
                # Categorize by meeting keywords
                if any(kw in content_lower for kw in meeting_kw):
                    insights["meetings"]["items"].append(thought_data)
                
                # Categorize by entities
                for e in entities:
                    if e["type"] and e["type"].lower() == "person":
                        if thought_data not in insights["people"]["items"]:
                            insights["people"]["items"].append(thought_data)
                    elif e["type"] and e["type"].lower() in ["project", "tool"]:
                        if thought_data not in insights["projects"]["items"]:
                            insights["projects"]["items"].append(thought_data)
                
                # Categorize by assigned categories
                for cat in categories:
                    if cat:
                        cat_lower = cat.lower()
                        if cat_lower in ["work", "tasks"]:
                            insights["tasks"]["items"].append(thought_data)
                        elif cat_lower in ["ideas", "brainstorm"]:
                            insights["ideas"]["items"].append(thought_data)
                        elif cat_lower in ["learning"]:
                            insights["learning"]["items"].append(thought_data)
                        elif cat_lower in ["personal", "reflection"]:
                            insights["reflections"]["items"].append(thought_data)
        
        # Update counts
        for key in insights:
            insights[key]["count"] = len(insights[key]["items"])
            insights[key]["items"] = insights[key]["items"][:10]
        
        return insights
    
    def get_category_items(self, category: str, limit: int = 50) -> List[Dict]:
        insights = self.get_brain_insights()
        cat_key = category.lower()
        if cat_key in insights:
            return insights[cat_key]["items"][:limit]
        return []
    
    def get_people(self) -> List[Dict]:
        """Get synthesized person profiles from Neo4j."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {type: 'Person'})
                OPTIONAL MATCH (e)-[:HAS_PROFILE]->(p:PersonProfile)
                OPTIONAL MATCH (e)<-[:MENTIONS]-(t:Thought)
                WITH e, p, count(t) as mention_count, collect(t.content)[0] as last_context
                RETURN e.name as name,
                       coalesce(p.role, 'Unknown') as role,
                       coalesce(p.relationship, 'Mentioned') as relationship,
                       coalesce(p.topics, []) as topics,
                       coalesce(p.summary, 'No profile yet') as summary,
                       mention_count
                ORDER BY mention_count DESC
            """)
            return [dict(record) for record in result]
    
    def get_synthesized_projects(self) -> List[Dict]:
        """Get synthesized project profiles from Neo4j."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.type IN ['Project', 'Tool']
                OPTIONAL MATCH (e)-[:HAS_PROFILE]->(p:ProjectProfile)
                OPTIONAL MATCH (e)<-[:MENTIONS]-(t:Thought)
                WITH e, p, count(t) as mention_count
                RETURN e.name as name,
                       coalesce(p.status, 'Active') as status,
                       coalesce(p.people, []) as people,
                       coalesce(p.deadline, null) as deadline,
                       coalesce(p.summary, 'No summary yet') as summary,
                       mention_count
                ORDER BY mention_count DESC
            """)
            return [dict(record) for record in result]
    
    def get_project_radar_data(self) -> List[Dict]:
        """
        Get data for Project Radar visualization.
        Metrics:
        - Velocity: Recent activity (last 7 days)
        - Maturity: Total knowledge accumulation (total mentions)
        - Impact: Number of connected entities (people, tools)
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Entity)
                WHERE p.type IN ['Project', 'Product', 'Tool']
                
                // Calculate Total Mentions (Maturity)
                OPTIONAL MATCH (p)<-[:MENTIONS]-(t:Thought)
                WITH p, count(t) as total_mentions
                
                // Calculate Recent Mentions (Velocity)
                OPTIONAL MATCH (p)<-[:MENTIONS]-(t2:Thought)
                WHERE datetime(t2.timestamp) > datetime() - duration({days: 7})
                WITH p, total_mentions, count(t2) as recent_mentions
                
                // Calculate Connections (Impact)
                OPTIONAL MATCH (p)<-[:MENTIONS]-(t3:Thought)-[:MENTIONS]->(e:Entity)
                WHERE p <> e
                WITH p, total_mentions, recent_mentions, count(distinct e) as connections
                
                RETURN p.name as name,
                       total_mentions as maturity,
                       recent_mentions as velocity,
                       connections as impact
                ORDER BY velocity DESC, maturity DESC
                LIMIT 8
            """)
            
            projects = []
            max_maturity = 1
            max_velocity = 1
            max_impact = 1
            
            data = [dict(record) for record in result]
            
            # Normalize scores
            if data:
                max_maturity = max(d["maturity"] for d in data) or 1
                max_velocity = max(d["velocity"] for d in data) or 1
                max_impact = max(d["impact"] for d in data) or 1
            
            for d in data:
                projects.append({
                    "name": d["name"],
                    "velocity": min(int((d["velocity"] / max_velocity) * 100), 100) if max_velocity > 0 else 0,
                    "maturity": min(int((d["maturity"] / max_maturity) * 100), 100) if max_maturity > 0 else 0,
                    "impact": min(int((d["impact"] / max_impact) * 100), 100) if max_impact > 0 else 0,
                    "status": "Active" if d["velocity"] > 0 else "Idle"
                })
                
            return projects
    
    def get_meetings(self) -> List[Dict]:
        """Get extracted meetings from Neo4j."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (m:Meeting)<-[:HAS_MEETING]-(t:Thought)
                RETURN m.title as title,
                       m.when as when,
                       m.participants as participants,
                       m.agenda as agenda,
                       t.timestamp as created_at
                ORDER BY t.timestamp DESC
            """)
            return [dict(record) for record in result]
    
    def get_all_entities(self, limit: int = 100) -> List[Dict]:
        """Get all entities with counts."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)<-[:MENTIONS]-(t:Thought)
                RETURN e.name as name, e.type as type, e.description as description, count(t) as thought_count
                ORDER BY thought_count DESC
                LIMIT $limit
            """, {"limit": limit})
            return [dict(record) for record in result]
    
    def get_graph_data(self) -> Dict:
        """Get full graph for visualization."""
        nodes = []
        edges = []
        with self.driver.session() as session:
            # Get thoughts
            thoughts = session.run("MATCH (t:Thought) RETURN t")
            for record in thoughts:
                t = record["t"]
                nodes.append({
                    "id": t["id"], "type": "thought",
                    "label": t["summary"][:50] if t["summary"] else t["content"][:50],
                    "data": dict(t)
                })
            
            # Get entities
            entities = session.run("MATCH (e:Entity) RETURN e")
            for record in entities:
                e = record["e"]
                nodes.append({
                    "id": e["key"], "type": "entity",
                    "entityType": e["type"], "label": e["name"],
                    "data": dict(e)
                })
            
            # Get relationships
            rels = session.run("MATCH (t:Thought)-[r:MENTIONS]->(e:Entity) RETURN t.id as source, e.key as target")
            for record in rels:
                edges.append({"source": record["source"], "target": record["target"], "type": "MENTIONS"})
        
        return {"nodes": nodes, "edges": edges}
    
    def get_stats(self) -> Dict:
        """Get brain statistics."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (t:Thought) WITH count(t) as thoughts
                MATCH (e:Entity) WITH thoughts, count(e) as entities
                RETURN thoughts, entities
            """)
            record = result.single()
            return {
                "total_thoughts": record["thoughts"] if record else 0,
                "total_entities": record["entities"] if record else 0,
                "total_conversations": len(self.conversation_history)
            }


# ============================================================================
# Graph Factory
# ============================================================================

_graph_instance = None


def get_knowledge_graph():
    """Get or create the knowledge graph instance."""
    global _graph_instance
    
    if _graph_instance is not None:
        return _graph_instance
    
    # Try Neo4j first
    if NEO4J_AVAILABLE:
        try:
            _graph_instance = Neo4jKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            return _graph_instance
        except Exception as e:
            print(f"⚠ Neo4j connection failed: {e}")
            print("→ Make sure Docker container is running: docker start peoples-agent-neo4j")
    
    # Fallback error - Neo4j is required now
    raise RuntimeError("Neo4j database is required. Start with: docker start peoples-agent-neo4j")


# Initialize on import
knowledge_graph = get_knowledge_graph()
