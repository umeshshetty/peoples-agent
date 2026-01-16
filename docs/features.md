# Features & Use Cases Guide 🚀

People's Agent is not just a note-taking app; it is a **Co-Cognition System**. This guide details every feature from a user's perspective, explaining how it fits into your daily workflow and the specific problems it solves.

---

## 1. The Intelligent Dropzone (Capture)

**Problem:** You hesitate to write things down because you don't know where to put them. "Is this a task? A project note? A random thought?"
**Solution:** A "Zero-Taxonomy" input. You dump; the AI sorts.

### Feature: Multi-Intent Decomposition
The AI analyzes your "brain dump" and splits it into actionable atoms.

**User Scenario:**
You are in a hurry and type:
> "Need to call Sarah about the Q3 budget. also, read up on vector databases for the new search feature. oh, and buy milk."

**What Happens:**
The AI detects three distinct intents and routes them:
1.  **Task (High Priority)**: "Call Sarah about Q3 budget" → Linked to **Project: Q3 Finance** (if exists) or **Person: Sarah**.
2.  **Learning Goal**: "Read about Vector Databases" → Filed under **Area: Technical Skills** or **Project: Search Feature**.
3.  **Personal Task**: "Buy milk" → Filed under **Area: Personal**.

**Benefit:** You capture freely without the cognitive load of organizing.

---

## 2. Brain World Dashboard (Reflect)

**Problem:** Traditional notes are static lists. You can't see the "shape" of your life or work.
**Solution:** A live, synthesized dashboard that visualizes your reality.

### Feature: Dynamic Person Profiles
**Scenario:** You've had 10 different meetings with "John" over the last year. You forget exactly what his role is.
**What You See:** You click on "John" in the Brain World.
**AI Output:**
> **John Smith** (Synthesized Profile)
> *   **Role**: CTO / Technical Lead
> *   **Relationship**: You report to him on the Search Project.
> *   **Key Topics**: Discussed API Latency, Q3 Budget, Vector DBs.
> *   **Recent Context**: Last meeting was Tuesday regarding "Deployment blockers".

### Feature: Project Intelligence
**Scenario:** You have 50 scattered notes about "Project Alpha."
**What You See:** The "Projects" tab shows "Project Alpha" as a card.
**AI Output:**
> **Status**: Active (Deadline: Friday)
> **Team**: Sarah, John, Mike
> **Open Tasks**: 3 urgent tasks extracted from your notes.
> **Blocking Issues**: "Waiting on API keys" (extracted from a note 2 days ago).

---

## 3. Intelligent Retrieval (Search)

**Problem:** You remember *meaning* ("that thing about leadership"), not *keywords* ("servant leadership principles").
**Solution:** Hybrid Search (Vector + Graph).

### Feature: Semantic Search
**Scenario:** You are writing a blog post about management. You type: "How to be a good boss?"
**AI Action:**
*   It scans your database. It finds a note from 2 years ago titled "Steve Jobs Biography" where you wrote about "maintaining high standards."
*   It finds a meeting note with "Sarah" where she gave you feedback on "empathy."
**Result:** It returns relevant concepts, not just text matches.

### Feature: Graph Traversal (Context)
**Scenario:** You ask, "What did I promise John?"
**AI Action:**
*   Identifies entity "John".
*   Traverses edges: `(You)-[PROMISED]->(Task)-[RELATED_TO]->(John)`.
**Result:** "You promised John you would finish the API docs by Friday."

---

## 4. The Serendipity Engine (Creativity)

**Problem:** You get stuck in "tunnels" of thought. You write about code and only see code notes.
**Solution:** Active injection of divergent ideas.

**Scenario:** You are writing a technical spec for a "Queued Job System."
**The AI Nudge:** "Hey, this looks similar to your note about 'Mise en place' from your Cooking class last year. Both are about preparation and sequencing."
**The "Aha!" Moment:** You realize you can apply kitchen organization principles to your software architecture.

**How it Works:** The AI looks for "Structural Holes"—items that are semantically related (high vector similarity) but topologically distant in your graph key (different clusters).

---

## 5. Active Learning (Growth)

**Problem:** You take notes but forget them. You don't truly learn.
**Solution:** The AI acts as a tutor.

### Feature: The Feynman Agent
**Scenario:** You want to solidify your knowledge of "Graph Theory."
**User Action:** You toggle "Feynman Mode."
**AI Interaction:**
> **AI**: "Okay, explain Graph Theory to me like I'm 12 years old. What is a node?"
> **You**: "A node is like a dot."
> **AI**: "Good. But how do these dots talk to each other? What connects them?"
> **You**: "Edges."
> **AI**: "Exactly. Now, can you give me a real-world example of an edge in your own life?"

**Benefit:** This forces Active Recall, which is scientifically proven to increase retention.

### Feature: Daily Briefing
**Scenario:** Monday Morning. You sit down, feeling overwhelmed.
**AI Action:** Generates a "Morning Briefing."
> "Good Morning!
> 1.   **Refresher**: On Friday, you were deep in the 'Authentication' module. You left a note saying 'JWT tokens are expiring too fast.'
> 2.  **Schedule**: You have a meeting with the Design Team at 2 PM.
> 3.  **Suggestion**: Review the 'Auth Specs' before you start coding."

---

## Summary of Use Cases

| User Goal | Feature |
| :--- | :--- |
| **"I need to clear my head."** | Use **Dropzone** (Decompose & Route) |
| **"Who is this person?"** | Check **Person Profiles** in Brain World |
| **"What am I missing?"** | Check **Serendipity** suggestions |
| **"Teach me this."** | Activate **Feynman Mode** |
| **"Start my day."** | Read **Daily Briefing** |
| **"Review what I learned."** | Use **Spaced Repetition** flashcards |
| **"Break this down."** | **Task Decomposition** (auto-generates subtasks) |

---

## 6. Spaced Repetition (Memory)

**Problem:** You take notes but forget them within days.
**Solution:** The AI schedules review using the SM-2 algorithm.

**How it Works:**
1. Notes marked as "Learning" or "Resource" enter the review queue.
2. The system calculates optimal review intervals based on your performance.
3. You rate each card as Easy/Medium/Hard, adjusting future intervals.

**API:** `GET /api/resurface` returns cards due for review.

---

## 7. Task Decomposition (Planning)

**Problem:** Big ideas are overwhelming. You don't know where to start.
**Solution:** The AI breaks complex tasks into actionable subtasks.

**Scenario:** You type: "Build a mobile app with authentication, payments, and notifications."

**AI Output:**
```
Parent Task: Build Mobile App
├── Subtask 1: Implement authentication (urgency: 4)
├── Subtask 2: Integrate payment gateway (urgency: 3)
├── Subtask 3: Set up push notifications (urgency: 2)
└── Subtask 4: Design onboarding flow (urgency: 2)
```

---

## 8. Zettelkasten Auto-Atomization

**Problem:** You paste a long article but can't find specific insights later.
**Solution:** The AI splits long-form content into atomic, interconnected notes.

**Scenario:** You paste a 1000-word article about productivity.

**AI Action:**
1. Splits into 5-7 atomic notes (each containing one concept).
2. Links related atoms together: `(Atom A)-[:RELATED_TO]->(Atom B)`.
3. All atoms link back to the original: `(Atom)-[:ATOMIZED_FROM]->(Original)`.

**Benefit:** Your graph stays granular and searchable.

