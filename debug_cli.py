"""Debug: time each step of the CLI workflow."""
import time

def step(msg, fn):
    print(f"⏱ {msg}...", end=" ", flush=True)
    t0 = time.time()
    try:
        result = fn()
        print(f"{time.time()-t0:.1f}s OK")
        return result
    except Exception as e:
        print(f"{time.time()-t0:.1f}s FAIL: {e}")
        raise

# Step 1: Init DB
from src.storage.database import init_db, seed_agent_configs
step("Init DB", lambda: (init_db(), seed_agent_configs()))

# Step 2: Create provider
from src.llm.deepseek import DeepSeekProvider
provider = step("Create DeepSeekProvider", lambda: DeepSeekProvider())

# Step 3: Run workflow
task = "用三句话介绍人工智能的历史"
print(f"⏱ Run workflow: '{task}'...", flush=True)
t0 = time.time()
from src.engine.graph import run_workflow
result = run_workflow(provider, task)
print(f"  Total: {time.time()-t0:.1f}s")

print(f"\nPlan: {len(result.get('plan',[]))} steps")
print(f"Results: {len(result.get('results',{}))} items")
errs = result.get('errors',[])
if errs:
    print(f"Errors: {errs}")
print(f"Output: {result.get('final_output','NONE')[:200]}")
