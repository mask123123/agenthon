"""Run the 146 extraction prompts (llm_ex/batch_*.json) against a REAL OpenAI-compatible endpoint (e.g. the House model via
build.nvidia.com for local testing) with the SAME client the image uses, write llm_ex/out_live_*.json, then score with
eval_llm_extract.py (set the glob there to out_live_*.json). Needs: MODEL_ENDPOINT (origin, no path), MODEL_TOKEN, MODEL_NAME.
   export MODEL_ENDPOINT=https://integrate.api.nvidia.com MODEL_NAME=nvidia/nemotron-3-super-120b-a12b MODEL_TOKEN=<your key>
   python run_llm_extract_live.py && sed -i '' 's#out_\\*.json#out_live_*.json#' eval_llm_extract.py && python eval_llm_extract.py
The key is read from the environment only; never write it to a file or commit it."""
import sys, glob, json, time
sys.path.insert(0, "/Users/apple/Documents/agenthon/submission")
from t2agent.house import House
h = House(max_requests=10_000, deadline_s=3600)
assert h.available, "set MODEL_ENDPOINT, MODEL_TOKEN, MODEL_NAME"
for f in sorted(glob.glob("llm_ex/batch_*.json")):
    out = []
    for it in json.load(open(f)):
        try:
            a = h.chat([{"role": "system", "content": it["system"]}, {"role": "user", "content": it["user"]}], max_tokens=300)
        except Exception as e:
            a = f"ERROR {type(e).__name__}"; h.fail_streak = 0
        out.append({"id": it["id"], "answer": a}); time.sleep(0.2)
    json.dump(out, open(f.replace("batch_", "out_live_"), "w"), indent=1); print("wrote", f.replace("batch_", "out_live_"), len(out))
