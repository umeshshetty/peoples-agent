import asyncio
from knowledge_graph import knowledge_graph, ThoughtNode, Entity
from datetime import datetime, timedelta

async def verify_backend_phase2():
    print("🚀 Verifying Phase 2 Backend Features...")
    
    # ========================================================================
    # 1. Smart Re-surface Validation
    # ========================================================================
    print("\n🧠 [1/2] Verifying Smart Re-surface (Spaced Repetition)...")
    
    # 1. Seed Old Thought (Should be in queue)
    old_date = "1990-01-01T10:00:00"
    old_thought = ThoughtNode(
        id="test_ancient_phase2",
        content="This is an ancient thought.",
        summary="Ancient concept",
        timestamp=old_date,
        review_count=0
    )
    knowledge_graph.add_thought(old_thought)
    
    # 2. Check Queue
    queue = knowledge_graph.get_resurface_queue(limit=5)
    queue_ids = [t.id for t in queue]
    print(f"Queue: {queue_ids}")
    
    if "test_ancient_phase2" in queue_ids:
        print("✅ Re-surface Queue logic works (Oldest First).")
    else:
        print("❌ Re-surface Queue logic FAILED.")

    # ========================================================================
    # 2. Project Radar Validation
    # ========================================================================
    print("\n📡 [2/2] Verifying Project Radar...")
    
    # 1. Seed a Project with mentions
    project_thought = ThoughtNode(
        id="test_project_thought_1",
        content="Working on Project Mars today.",
        summary="Mars work",
        timestamp=datetime.now().isoformat(),
        entities=[Entity(name="Project Mars", type="Project")]
    )
    knowledge_graph.add_thought(project_thought)
    print("✓ Seeded project data for 'Project Mars'")
    
    # 2. Get Radar Data
    radar_data = knowledge_graph.get_project_radar_data()
    print(f"Radar Data: {[p['name'] for p in radar_data]}")
    
    mars_data = next((p for p in radar_data if p["name"] == "Project Mars"), None)
    
    if mars_data:
        print(f"✅ Found Project Mars: {mars_data}")
        if mars_data['velocity'] > 0:
            print("✅ Velocity metric calculates correctly (>0).")
        else:
            print("❌ Velocity metric failed (is 0).")
    else:
        print("❌ Project Mars NOT found in Radar data.")

if __name__ == "__main__":
    asyncio.run(verify_backend_phase2())
