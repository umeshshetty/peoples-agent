"""
People's Agent - Background Worker
Handles heavy synthesis tasks asynchronously to reduce user-perceived latency.
"""

import asyncio
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
import threading

# Thread pool for background tasks
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="synthesis_")
_pending_tasks: Dict[str, asyncio.Task] = {}


def run_in_background(coro, task_id: str):
    """
    Fire-and-forget: Run a coroutine in the background without blocking.
    Results are logged, not returned.
    """
    async def wrapper():
        try:
            result = await coro
            print(f"   ► Background task {task_id} completed: {result.get('status', 'ok')}")
            return result
        except Exception as e:
            print(f"   ⚠ Background task {task_id} failed: {e}")
            return {"error": str(e)}
    
    # Get or create event loop
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(wrapper())
        _pending_tasks[task_id] = task
    except RuntimeError:
        # No running loop, create one in a thread
        def run_async():
            asyncio.run(wrapper())
        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()


def get_task_status(task_id: str) -> Dict[str, Any]:
    """Check status of a background task."""
    task = _pending_tasks.get(task_id)
    if not task:
        return {"status": "not_found"}
    if task.done():
        try:
            result = task.result()
            return {"status": "completed", "result": result}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
    return {"status": "running"}


def cleanup_completed_tasks():
    """Remove completed tasks from tracking dict."""
    to_remove = [tid for tid, task in _pending_tasks.items() if task.done()]
    for tid in to_remove:
        del _pending_tasks[tid]
