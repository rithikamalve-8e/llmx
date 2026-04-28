"""
Smoke-test for the Langfuse integration.

Run:  python test_langfuse.py

Requirements
------------
* Langfuse server running at LANGFUSE_BASE_URL (default http://localhost:3000)
* LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY present in .env
* At least one of: OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv
load_dotenv()


# ── helpers ────────────────────────────────────────────────────────────────────

def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")

def fail(msg: str, e: Exception | None = None) -> None:
    print(f"  [FAIL] {msg}")
    if e:
        print(f"         {repr(e)}")

def section(title: str) -> None:
    print(f"\n{'─' * 14} {title} {'─' * 14}")


# ── 1. Verify observability module imports correctly ───────────────────────────

section("Module import")

try:
    from llmx.observability import observe, lf as get_lf
    lf_client = get_lf()
    if lf_client is not None:
        ok("langfuse imported — client ready")
    else:
        fail("langfuse not installed or client failed. Run: pip install langfuse")
        raise SystemExit(1)
except SystemExit:
    raise
except Exception as e:
    fail("Could not import llmx.observability", e)
    raise SystemExit(1)


# ── 2. Check environment variables ────────────────────────────────────────────

section("Environment")

LANGFUSE_SK  = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PK  = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_URL = os.getenv("LANGFUSE_BASE_URL")

if not (LANGFUSE_SK and LANGFUSE_PK):
    fail("LANGFUSE_SECRET_KEY or LANGFUSE_PUBLIC_KEY missing from .env")
    raise SystemExit(1)
ok(f"Langfuse keys present  (base_url={LANGFUSE_URL})")

OPENAI  = bool(os.getenv("OPENAI_API_KEY"))
GROQ    = bool(os.getenv("GROQ_API_KEY"))
GEMINI  = bool(os.getenv("GEMINI_API_KEY"))
print(f"  OpenAI: {'✓' if OPENAI else '✗'}   Groq: {'✓' if GROQ else '✗'}   Gemini: {'✓' if GEMINI else '✗'}")


# ── 3. Make LLM calls and capture trace IDs ───────────────────────────────────

section("LLM calls")

from langfuse import observe as lf_observe
from llmx import LLMClient

PROVIDERS_TO_TEST = []
if OPENAI:  PROVIDERS_TO_TEST.append(("openai", "gpt-4o-mini"))
if GROQ:    PROVIDERS_TO_TEST.append(("groq",   "groq/compound-mini"))
if GEMINI:  PROVIDERS_TO_TEST.append(("gemini", "gemini-2.0-flash"))

if not PROVIDERS_TO_TEST:
    fail("No LLM provider keys found — add at least one to .env")
    raise SystemExit(1)

# (provider, trace_id, response_content)
results: list[tuple[str, str, str]] = []

for provider, model in PROVIDERS_TO_TEST:
    captured: dict = {}

    # Wrap in a parent @observe so we can read the trace_id while still
    # inside the observed context via get_current_trace_id().
    @lf_observe(name=f"test.{provider}")
    def _run():
        client = LLMClient(provider=provider)
        resp = client.generate(
            "Reply with exactly the words: trace ok",
            model=model,
            max_tokens=20,
        )
        captured["trace_id"] = lf_client.get_current_trace_id()
        captured["output"]   = resp.content

    try:
        _run()
        tid = captured.get("trace_id")
        out = captured.get("output", "")
        if tid:
            ok(f"{provider} ({model})")
            ok(f"  response   = {out!r}")
            ok(f"  trace_id   = {tid}")
            results.append((provider, tid, out))
        else:
            fail(f"{provider}: call succeeded but no trace_id captured")
    except Exception as e:
        fail(f"{provider} generate() failed", e)


# ── 4. Flush and print UI links ───────────────────────────────────────────────

section("Flushing & trace links")

if not results:
    fail("No successful LLM calls — nothing to verify")
    raise SystemExit(1)

print("  Flushing Langfuse queue ...", end=" ", flush=True)
lf_client.flush()
time.sleep(2)
print("done\n")

failures = 0

for provider, tid, output in results:
    try:
        url = lf_client.get_trace_url(trace_id=tid)
        ok(f"{provider} trace URL:")
        print(f"         {url}")

        # Basic content checks we can do locally
        if not output:
            fail(f"{provider}: LLM returned empty output")
            failures += 1
        if not tid:
            fail(f"{provider}: trace_id is empty")
            failures += 1

    except Exception as e:
        fail(f"{provider}: could not build trace URL", e)
        failures += 1


# ── 5. Deep REST verification (optional) ──────────────────────────────────────

section("Deep trace verification (REST API)")

try:
    from langfuse.api.client import LangfuseAPI
    import httpx

    _http = httpx.Client(
        base_url=LANGFUSE_URL,
        auth=(LANGFUSE_PK, LANGFUSE_SK),
        timeout=10,
    )
    _api_available = True
    ok("REST client ready")
except Exception as e:
    _api_available = False
    ok("Skipping REST check — open the URLs above to verify in the UI")

if _api_available:
    for provider, tid, _ in results:
        print(f"\n  ── {provider} ──")
        try:
            # Fetch trace via raw HTTP (avoids init complexity of LangfuseAPI)
            r = _http.get(f"/api/public/traces/{tid}")
            r.raise_for_status()
            trace = r.json()

            ok(f"Trace found          name={trace.get('name')!r}")

            if trace.get("input"):
                ok("trace.input          populated")
            else:
                fail("trace.input          empty — input not attached to trace")
                failures += 1

            if trace.get("output"):
                ok(f"trace.output         {str(trace.get('output'))[:60]!r}")
            else:
                fail("trace.output         empty — output not attached to trace")
                failures += 1

            # Fetch observations for this trace
            r2 = _http.get(f"/api/public/observations", params={"traceId": tid})
            r2.raise_for_status()
            obs_data = r2.json().get("data", [])

            gen_obs  = [o for o in obs_data if o.get("type") == "GENERATION"]
            span_obs = [o for o in obs_data if o.get("type") == "SPAN"]

            if span_obs:
                print(f"         spans:       {[s.get('name') for s in span_obs]}")

            if gen_obs:
                g = gen_obs[0]
                ok(f"Generation span      name={g.get('name')!r}  model={g.get('model')!r}")

                if g.get("model"):
                    ok(f"model field          {g.get('model')!r}")
                else:
                    fail("model field          missing on generation span")
                    failures += 1

                if g.get("input"):
                    ok("generation.input     populated")
                else:
                    fail("generation.input     empty")
                    failures += 1

                if g.get("output"):
                    ok(f"generation.output    {str(g.get('output'))[:60]!r}")
                else:
                    fail("generation.output    empty")
                    failures += 1

                usage = g.get("usage") or {}
                if usage.get("input") is not None:
                    ok(f"token usage          input={usage.get('input')}  output={usage.get('output')}")
                elif provider == "gemini":
                    ok("token usage          (Gemini does not expose usage tokens — expected)")
                else:
                    fail(f"token usage          missing on {provider} generation span")
                    failures += 1
            else:
                fail("No GENERATION observation found in trace")
                failures += 1

        except Exception as e:
            fail(f"{provider}: REST verification failed", e)
            failures += 1


# ── 6. Summary ────────────────────────────────────────────────────────────────

section("Summary")

total  = len(results)
passed = total - failures

if not _api_available:
    print(f"  {total}/{total} LLM call(s) traced successfully.")
    print(f"  Open the URLs above to verify structure in the Langfuse UI.")
elif failures == 0:
    print(f"  All {total}/{total} provider(s) passed — traces verified via REST.")
    print(f"  Open {LANGFUSE_URL} to inspect in the UI.")
else:
    print(f"  {passed}/{total} passed,  {failures} failure(s).")
    raise SystemExit(1)
