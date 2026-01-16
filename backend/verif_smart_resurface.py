import asyncio
from knowledge_graph import knowledge_graph, ThoughtNode
from datetime import datetime, timedelta

async def verify_spaced_repetition():
    print("🧠 Verifying Smart Re-surface (Spaced Repetition)...")
    
    # 1. Seed Old Thought (Should be in queue)
    old_date = "1990-01-01T10:00:00"
    old_thought = ThoughtNode(
        id="test_ancient_1",
        content="This is an ancient thought that needs review.",
        summary="Ancient concept",
        timestamp=old_date,
        review_count=0
    )
    knowledge_graph.add_thought(old_thought)
    print("✓ Seeded old thought: 'test_ancient_1' (1990)")
    
    # 2. Seed Fresh Thought (Should NOT be in queue yet, maybe)
    # Actually, default query gets unreviewed items > 0 days? 
    # Let's check logic: review_count=0 OR (review_count>0 AND interval passed)
    # So a brand new thought might appear immediately if we don't filter by age for count=0
    # Let's see what happens.
    
    # 3. Check Queue
    print("\n[Step 1] Checking Resurface Queue...")
    queue = knowledge_graph.get_resurface_queue(limit=10)
    queue_ids = [t.id for t in queue]
    print(f"Queue IDs: {queue_ids}")
    
    if "test_ancient_1" in queue_ids:
        print("✅ SUCCESS: Old thought surfaced in queue.")
    else:
        print("❌ FAILURE: Old thought NOT in queue.")
        
    # 4. Mark as Reviewed
    print("\n[Step 2] Marking 'test_ancient_1' as Reviewed (Easy)...")
    knowledge_graph.mark_as_reviewed("test_ancient_1", "easy")
    
    # 5. Check Queue Again (Should be gone)
    print("\n[Step 3] Checking Queue after Review...")
    queue_2 = knowledge_graph.get_resurface_queue(limit=10)
    queue_ids_2 = [t.id for t in queue_2]
    print(f"Queue IDs: {queue_ids_2}")
    
    if "test_ancient_1" not in queue_ids_2:
        print("✅ SUCCESS: Reviewed thought removed from queue.")
    else:
        print("❌ FAILURE: Reviewed thought still in queue.")

if __name__ == "__main__":
    asyncio.run(verify_spaced_repetition())
