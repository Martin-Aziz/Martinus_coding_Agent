import logging
import os

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from .llm import LocalModelClient
from .memory import SQLiteMemoryStore
from .models import (
    ChatRequest,
    ChatResponse,
    InsightItem,
    InsightsRequest,
    InsightsResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryUpsertRequest,
    MemoryUpsertResponse,
    ReflectRequest,
    ReflectResponse,
    SelfConsistencyRequest,
    SelfConsistencyResponse,
    SummarizeRequest,
    SummarizeResponse,
)

logger = logging.getLogger(__name__)


def _compress_locally(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    # 65% head + 25% tail = 90% of budget, 10% for separator
    head = text[: int(max_chars * 0.65)]
    tail = text[-int(max_chars * 0.25):]
    return head + "\n... [compacted] ...\n" + tail


app = FastAPI(title="creator-ai-service", version="0.3.0")

# CORS — orchestrator (default localhost:8080/18080) and both UIs need to reach
# ai-service. Match the orchestrator's allow-list; loopback-only by default.
_AI_ALLOWED_ORIGINS = os.getenv(
    "AI_SERVICE_ALLOWED_ORIGINS",
    "http://localhost:8080,http://127.0.0.1:8080,"
    "http://localhost:18080,http://127.0.0.1:18080,"
    "http://localhost:8091,http://127.0.0.1:8091,"
    "http://127.0.0.1:1420,http://localhost:1420,"
    "tauri://localhost,http://tauri.localhost",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_AI_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Origin", "Content-Type", "Accept", "X-API-Key"],
)

# Optional API-key gate — matches the orchestrator pattern. Loopback-only by
# default so a professor never sees a 401 on their first health check.
_AI_API_KEY = os.getenv("AI_SERVICE_API_KEY", "").strip()
_AI_REQUIRE_API_KEY = os.getenv("AI_SERVICE_REQUIRE_API_KEY", "false").strip().lower() == "true"


@app.middleware("http")
async def _api_key_middleware(request: Request, call_next):
    """Loopback-friendly api-key gate. /healthz is always public."""
    if request.url.path in ("/healthz",) or request.method == "OPTIONS":
        return await call_next(request)
    if not _AI_API_KEY:
        if _AI_REQUIRE_API_KEY:
            from starlette.responses import JSONResponse
            return JSONResponse(
                status_code=503,
                content={"detail": "AI_SERVICE_REQUIRE_API_KEY=true but AI_SERVICE_API_KEY is empty"},
            )
        return await call_next(request)
    key = request.headers.get("X-API-Key", "").strip()
    import hmac
    if not hmac.compare_digest(key, _AI_API_KEY):
        from starlette.responses import JSONResponse
        return JSONResponse(status_code=401, content={"detail": "missing or invalid API key"})
    return await call_next(request)


store = SQLiteMemoryStore(
    os.getenv("MEMORY_DB_PATH", "./data/memory.sqlite3")
)
model_client = LocalModelClient()

# SELF_CONSISTENCY_ENABLED controls whether /v1/self-consistency is active.
# Default: false — the endpoint exists but returns 503 when disabled so callers
# know it is intentionally off. The 20-task ablation did not activate this
# endpoint; it remains available for future ablation conditions.
_SELF_CONSISTENCY_ENABLED = (
    os.getenv("SELF_CONSISTENCY_ENABLED", "false").lower() == "true"
)


@app.get("/healthz")
def health() -> dict:
    return {"ok": True, "service": "ai-service"}


@app.post("/v1/memory/upsert", response_model=MemoryUpsertResponse)
def memory_upsert(req: MemoryUpsertRequest) -> MemoryUpsertResponse:
    identifier = store.insert(
        session_id=req.session_id,
        content=req.content,
        tags=req.tags,
        workspace_path=req.workspace_path,
        source_phase=req.source_phase,
        source_provider=req.source_provider,
        source_model=req.source_model,
        outcome_label=req.outcome_label,
        metadata=req.metadata,
    )
    return MemoryUpsertResponse(id=identifier)


@app.post("/v1/memory/retrieve", response_model=MemorySearchResponse)
def memory_retrieve(
    req: MemorySearchRequest,
) -> MemorySearchResponse:
    results = store.retrieve(
        session_id=req.session_id,
        query=req.query,
        top_k=req.top_k,
        workspace_path=req.workspace_path,
        include_cross_session=req.include_cross_session,
        current_phase=req.current_phase,
    )
    return MemorySearchResponse(items=results)


@app.post("/v1/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest) -> SummarizeResponse:
    compacted = _compress_locally(req.text, req.max_chars)
    prompt = (
        "ROLE: You are a senior technical writer condensing an engineering agent's context window.\n\n"
        "REASONING PROTOCOL: Before writing the summary, identify: "
        "(1) the single most important fact from the context, "
        "(2) which constraint will most likely constrain the next action, "
        "(3) what the agent should do immediately next.\n\n"
        "Summarize this engineering context for a multi-file coding agent. "
        "Return exactly three sections (no other text):\n"
        "1. Current Objective: one sentence — what must be accomplished\n"
        "2. Key Constraints: up to 3 bullet points — hard limits that must not be violated\n"
        "3. Next Actions: up to 3 bullet points — concrete next steps in priority order\n\n"
        + compacted
    )

    try:
        provider, model, summary = await model_client.chat(
            prompt, task_type="summarize"
        )
    except Exception as exc:
        logger.warning("summarize LLM call failed, using fallback: %s", exc, exc_info=True)
        summary = compacted[: req.max_chars]
        provider = "fallback"
        model = "local-compactor"

    if not summary.strip():
        raise HTTPException(
            status_code=502, detail="empty summarize response"
        )

    return SummarizeResponse(
        provider=provider, model=model, summary=summary
    )


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    prompt = req.prompt
    if req.metadata:
        prompt = f"Metadata: {req.metadata}\n\n" + prompt

    provider, model, output = await model_client.chat(
        prompt,
        task_type=req.task_type,
        temperature=req.temperature,
    )
    if not output:
        logger.warning("empty model response from provider=%s model=%s", provider, model)
        raise HTTPException(
            status_code=502, detail="empty model response"
        )

    if req.session_id:
        store.insert(req.session_id, output, ["chat-output"])

    return ChatResponse(provider=provider, model=model, output=output)


@app.post(
    "/v1/self-consistency",
    response_model=SelfConsistencyResponse,
)
async def self_consistency(
    req: SelfConsistencyRequest,
) -> SelfConsistencyResponse:
    """
    Sample the model N times and return the majority-vote response.

    Implements Self-Consistency (Wang et al., 2023, arXiv:2203.11171):
    diverse reasoning paths are sampled at higher temperature, then
    aggregated to produce a more reliable answer than any single sample.
    Useful for Review-phase queries where single-sample output is noisy.

    Gated by SELF_CONSISTENCY_ENABLED env var (default: false).
    The 20-task ablation did NOT activate this endpoint; set the env var
    to true to enable it for future ablation conditions.
    """
    if not _SELF_CONSISTENCY_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Self-consistency is disabled. "
                "Set SELF_CONSISTENCY_ENABLED=true to enable."
            ),
        )
    provider, model, output = await model_client.self_consistency_chat(
        prompt=req.prompt,
        n=req.n,
        temperature=req.temperature,
    )
    if not output:
        raise HTTPException(
            status_code=502, detail="empty self-consistency response"
        )

    if req.session_id:
        store.insert(
            req.session_id, output, ["self-consistency-output"]
        )

    return SelfConsistencyResponse(
        provider=provider,
        model=model,
        output=output,
        samples=req.n,
    )


@app.post("/v1/reflect", response_model=ReflectResponse)
async def reflect(req: ReflectRequest) -> ReflectResponse:
    """
    Algorithm 4.6: Post-Run Reflection.

    Extract a concise "lesson learned" from a completed agent-run transcript
    (phase outputs truncated to 800 runes). The model generates a single-paragraph
    lesson at T=0.2, specifying the technique, pattern, or mistake observed and
    stating what should be done differently next time.

    The lesson is stored as a high-priority memory entry with:
    - source_phase=Reflection
    - tag=lesson-learned
    - outcome_prior: success→1.0, partial→0.6, failure→0.2
    - elevated importance in future retrievals

    Source: Algorithm 4.6, thesis §4.4.2. Inspired by Shinn et al. (2023) Reflexion.
    """
    prompt = (
        f"ROLE: You are a senior engineering coach reviewing a completed coding-agent run "
        f"to extract transferable lessons.\n\n"
        f"REASONING PROTOCOL: Before writing the lesson, analyze:\n"
        f"(1) What technique, pattern, or decision most influenced the outcome?\n"
        f"(2) Was the outcome caused by the approach chosen or by execution quality?\n"
        f"(3) What would a different approach have changed?\n\n"
        f"Task: {req.task}\n"
        f"Outcome: {req.outcome}\n\n"
        f"=== Run Transcript ===\n{req.transcript}\n\n"
        "Based on the transcript above, extract a single concise lesson learned "
        "that would help a coding agent perform better on similar tasks in the future.\n\n"
        "Requirements for the lesson:\n"
        "- One paragraph, maximum 3 sentences.\n"
        "- Be specific: name the technique, pattern, or mistake observed.\n"
        "- State what should be done differently (or repeated) next time.\n"
        "- Do not restate the task description.\n"
        "- Do not include the reasoning analysis in the output — output the lesson only.\n\n"
        "Output only the lesson text, no headers or labels."
    )

    try:
        provider, model, lesson = await model_client.chat(
            prompt, task_type="summarize"
        )
    except Exception as exc:
        logger.warning("reflect LLM call failed: %s", exc, exc_info=True)
        return ReflectResponse(provider="fallback", model="none", lesson="")

    lesson = lesson.strip()
    if not lesson:
        return ReflectResponse(provider=provider, model=model, lesson="")

    # Store the lesson as a high-priority memory entry.
    store.insert(
        session_id=req.session_id,
        content=lesson,
        tags=["reflection", "lesson-learned"],
        workspace_path=req.workspace_path,
        source_phase="Reflection",
        outcome_label=req.outcome,
        metadata={"task": req.task[:256]},
    )

    return ReflectResponse(provider=provider, model=model, lesson=lesson)


@app.post("/v1/insights", response_model=InsightsResponse)
async def consolidate_insights(req: InsightsRequest) -> InsightsResponse:
    """
    Algorithm 4.7: ExpeL-Style Insight Consolidation.

    When >= min_lessons lesson-learned entries have accumulated for a workspace,
    consolidate them into 3-7 distilled heuristics and store each with elevated
    importance. Prevents memory dilution as the lesson store grows; creates a
    compact, reusable skill library for future runs on the same workspace.

    Source: Zhao, A., Huang, D., Xu, Q., Lin, M., Liu, Y.-J., & Huang, G.
    (2024). ExpeL: LLM Agents Are Experiential Learners. AAAI 2024.
    arXiv:2308.10144.
    """
    results = store.retrieve(
        session_id="__insights__",
        query="lesson learned technique pattern mistake architecture",
        top_k=50,
        workspace_path=req.workspace_path,
        include_cross_session=True,
        current_phase="Reflection",
    )
    lessons_with_tag = [
        item for item in results
        if "lesson-learned" in item.get("tags", [])
    ]

    if len(lessons_with_tag) < req.min_lessons:
        return InsightsResponse(
            provider="none",
            model="none",
            insights=[],
            lessons_consolidated=0,
        )

    lesson_texts = "\n\n".join(
        f"{i + 1}. {item['content'][:400]}"
        for i, item in enumerate(lessons_with_tag[:40])
    )

    prompt = (
        "ROLE: You are a senior engineering coach distilling lessons into reusable heuristics.\n\n"
        "REASONING PROTOCOL: Before writing insights, identify:\n"
        "(1) Which lessons share the same root cause or pattern?\n"
        "(2) Which lessons are specific one-offs vs. broadly transferable?\n"
        "(3) What is the most actionable formulation of each pattern?\n\n"
        f"The following {len(lessons_with_tag)} lessons were learned from coding-agent runs "
        f"on workspace {req.workspace_path}:\n\n"
        f"{lesson_texts}\n\n"
        "Distil these into 3 to 7 concise, actionable heuristics that a coding agent "
        "should apply to future tasks on this workspace.\n\n"
        "Requirements for each heuristic:\n"
        "- One sentence, imperative form (e.g. 'Always validate X before Y').\n"
        "- Specific to the patterns observed — not generic advice.\n"
        "- Ordered from most- to least-impactful.\n\n"
        "Output only the numbered list, no headers or commentary."
    )

    try:
        provider, model, output = await model_client.chat(
            prompt, task_type="plan"
        )
    except Exception as exc:
        logger.warning("insights LLM call failed: %s", exc, exc_info=True)
        return InsightsResponse(
            provider="fallback", model="none", insights=[], lessons_consolidated=0
        )

    lines = [
        line.strip().lstrip("0123456789. ").strip()
        for line in output.strip().splitlines()
        if line.strip() and line.strip()[0].isdigit()
    ]

    stored_insights: list[InsightItem] = []
    for insight_text in lines:
        if not insight_text:
            continue
        entry_id = store.insert(
            session_id="__insights__",
            content=insight_text,
            tags=["insight", "consolidated"],
            workspace_path=req.workspace_path,
            source_phase="Insight",
            outcome_label="success",
            metadata={"consolidated_from": str(len(lessons_with_tag))},
        )
        stored_insights.append(InsightItem(text=insight_text, id=entry_id))

    return InsightsResponse(
        provider=provider,
        model=model,
        insights=stored_insights,
        lessons_consolidated=len(lessons_with_tag),
    )
