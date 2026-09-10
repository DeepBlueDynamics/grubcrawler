# Master Plan - gnosis-crawl Agent Module

## Scope
Build a dual-mode agent architecture where:
1. External agents remain first-class (`Mode A`, default).
2. Internal micro-agent loop is optional and policy-bound (`Mode B`, opt-in).

The same tool contracts and trace schema must work in both modes.

## Goals
1. Keep current external crawl APIs/tool calls backward compatible.
2. Add an internal bounded loop with strict policy gates and typed errors.
3. Emit replayable traces for all tool executions.
4. Keep internal agent disabled by default.

## Workstreams

### W1. Agent Core
Files:
- `app/agent/types.py`
- `app/agent/errors.py`
- `app/agent/dispatcher.py`
- `app/agent/engine.py`

Deliverables:
1. State machine primitives:
- `RunState`: `INIT`, `PLAN`, `EXECUTE_TOOL`, `OBSERVE`, `RESPOND`, `STOP`, `ERROR`
- `StopReason`: `max_steps`, `max_wall_time`, `max_failures`, `no_op_loop`, `policy_denied`, `completed`
2. Normalized actions/results:
- `AssistantAction` => `Respond(text)` or `ToolCalls([ToolCall])`
- `ToolCall` => `id`, `name`, `args`
- `ToolResult` => `ok|err`, `payload`, `error_code`, `retriable`
3. Loop runner:
- `run_task(task, config, context) -> RunResult`
- `plan(run_ctx) -> AssistantAction`
- `step(run_ctx, action) -> StepResult`
4. Stop conditions enforced every iteration.

### W2. Unified Tool Contract
Files:
- `app/tools/tool_registry.py`
- `app/agent/dispatcher.py`

Deliverables:
1. Normalized descriptor:
- `name`, `args_schema`, `timeout_ms`, `retry_profile`, `side_effect_class (read|write|auth)`
2. Shared usage by:
- External AHP tool execution path.
- Internal agent dispatcher.
3. Dispatcher behavior:
- Arg validation
- Timeout + retry handling
- Typed error normalization (no raw exceptions to loop callers)

### W3. Policy and Safety
Files:
- `app/policy/domain.py`
- `app/policy/gate.py`
- `app/policy/redaction.py`

Deliverables:
1. Domain allowlist + private network deny by default.
2. Pre-tool and pre-fetch policy checks.
3. Prompt-injection quarantine transform for crawled text before planning.
4. Secret redaction before logs and persisted outputs.

### W4. Observability and Trace
Files:
- `app/observability/trace.py`
- `app/observability/events.py`

Deliverables:
1. Shared step trace schema:
- `run_id`, `step_id`, `tool_name`, `args_hash`, `duration_ms`, `status`, `error_code`, `artifacts[]`, `policy_flags[]`
2. Persist trace/artifacts to session/job storage.
3. Replay-friendly run summary output.

### W5. API + Job Wiring
Files:
- `app/job_routes.py`
- `app/jobs.py`
- `app/models.py` (if request/response expansion needed)

Deliverables:
1. Convert `/api/wraith` from placeholder to real submission.
2. Add `JobType.AGENT_RUN`.
3. Add processor branch for internal agent execution.
4. Preserve external mode parity when internal agent is disabled.

### W6. Provider Abstraction
Files:
- `app/agent/providers/base.py`
- `app/agent/providers/openai_adapter.py`
- `app/agent/providers/anthropic_adapter.py`
- `app/agent/providers/ollama_adapter.py`

Deliverables:
1. Provider interface:
- `LLMAdapter.complete(actions, context) -> AssistantAction`
2. Normalized response mapping:
- OpenAI tool-calls
- Anthropic `tool_use`/`tool_result`
- Ollama `tool_calls`/tool messages
3. Fallback + retry:
- Retry transient once, then rotate provider.

Defaults:
1. OpenAI: `gpt-4.1-mini`
2. Anthropic: `claude-3-5-sonnet-latest`
3. Ollama: `llama3.1:8b-instruct`

### W7. Config and Startup Validation
Files:
- `app/config.py`
- startup validation path in `app/main.py` (or dedicated validator module)

Deliverables:
1. Add:
- `agent.internal.enabled` (default `false`)
- `agent.internal.max_steps` (default `12`)
- `agent.internal.max_wall_time_ms` (default `90000`)
- `agent.internal.max_failures` (default `3`)
- `agent.internal.allowed_tools` (explicit allowlist)
- policy flags (`allowed_domains`, private-range blocking, redaction, raw HTML persistence)
2. Fail closed on invalid config.

### W8. Ghost Protocol (Vision Fallback)
Files:
- `app/agent/ghost.py`
- `app/policy/gate.py` (extend with cloak-mode trigger rules)
- `app/agent/providers/base.py` (extend with vision interface)

Concept:
When a crawl hits an anti-bot wall (Cloudflare challenge, CAPTCHA, empty
SPA shell, bot-detection interstitial), the agent switches to "cloak mode":
instead of parsing the DOM, it takes a screenshot and sends the rendered
image to a vision-capable LLM to extract content from pixels. This sidesteps
DOM-based anti-bot entirely because the extraction reads the visual render,
not the markup.

Deliverables:
1. Cloak-mode trigger detection:
   - Detect Cloudflare/CAPTCHA/challenge pages from crawl result signals
     (`blocked`, `captcha_detected`, thin body, known block signatures).
   - Detect empty SPA shells (JS-rendered content that Playwright missed).
   - Configurable: auto-trigger on block, or manual via tool arg.
2. Screenshot capture pipeline:
   - Full-page screenshot via Playwright CDP (`Page.screenshot`).
   - Optional viewport-only mode for above-the-fold content.
   - Downscale to reasonable size (max 1280px wide) to control token cost.
3. Vision extraction:
   - `LLMAdapter.vision(image_bytes, prompt) -> str` — new method on provider interface.
   - Sends screenshot + extraction prompt to vision-capable model.
   - Provider routing: Claude (claude-sonnet-4-5-20250929) or GPT-4o for vision.
   - Extraction prompt templates: "Extract all visible text", "Extract pricing table",
     "Describe the page layout and main content".
4. Fallback chain in dispatcher:
   - Normal crawl → if blocked → ghost protocol screenshot → vision extract.
   - Trace records which mode produced the final content (`render_mode: "ghost"`).
   - Policy gate: ghost mode must be explicitly allowed per-run (`allow_ghost: true`).
5. Ghost tool:
   - `ghost_extract(url, prompt?, viewport_only?)` — standalone tool callable by
     the agent or external clients.
   - Returns extracted text + screenshot artifact reference.
6. Config flags:
   - `AGENT_GHOST_ENABLED` (default: `false`)
   - `AGENT_GHOST_AUTO_TRIGGER` (default: `true` — auto-trigger on detected blocks)
   - `AGENT_GHOST_VISION_PROVIDER` (default: inherits from `AGENT_PROVIDER`)
   - `AGENT_GHOST_MAX_IMAGE_WIDTH` (default: `1280`)

### W9. Live Browser Stream
Files:
- `app/browser_pool.py`
- `app/stream.py`
- `app/main.py` (mount WebSocket route)

Concept:
Persistent Chromium instance that stays warm across requests and exposes a
live video stream of its viewport over WebSocket or MJPEG. Agents or humans
can watch what the crawler sees in real time.

Deliverables:
1. Persistent browser pool:
   - Keep 1-N Chromium instances alive across requests.
   - Connect via `browser.connect_over_cdp()`.
   - Health check + auto-restart on crash.
2. CDP screencast relay:
   - `Page.startScreencast(format='jpeg', quality=25, maxWidth=854)`.
   - Chrome sends JPEG frames natively — no extra deps.
3. WebSocket endpoint:
   - `ws://host:8080/stream/{session_id}` — pushes JPEG frames to clients.
   - Low quality (JPEG q20-30), 480p max, 5-10 FPS.
4. MJPEG fallback:
   - `GET /stream/{session_id}/mjpeg` — `multipart/x-mixed-replace` for
     dumb clients (works in any `<img>` tag).
5. Config flags:
   - `BROWSER_POOL_SIZE` (default: `1`)
   - `BROWSER_STREAM_ENABLED` (default: `false`)
   - `BROWSER_STREAM_QUALITY` (default: `25`)
   - `BROWSER_STREAM_MAX_WIDTH` (default: `854`)

## Error Semantics
Use machine-readable typed errors:
1. `validation_error`
2. `policy_denied`
3. `tool_timeout`
4. `tool_unavailable`
5. `execution_error`

Each response includes retry hint metadata when applicable.

## Test Plan

### Unit
1. Stop condition triggers.
2. Policy gate decisions.
3. Dispatcher normalization/retry behavior.

### Integration
1. External mode parity with internal disabled.
2. Internal bounded multi-step completion.
3. Fail-closed behavior on policy violations.

### Provider Contract
1. OpenAI mapping fidelity.
2. Anthropic mapping fidelity.
3. Ollama mapping fidelity.

### Security
1. Prompt-injection quarantine blocks forbidden actions.
2. Secret redaction applied in logs and final outputs.

### Ghost Protocol
1. Cloak-mode triggers on known block signatures.
2. Vision extraction produces usable text from screenshot.
3. Ghost mode respects `allow_ghost` policy gate.
4. Fallback chain: normal crawl → ghost → error (no silent failures).

### Live Stream
1. WebSocket delivers frames at target FPS.
2. MJPEG fallback renders in a plain `<img>` tag.
3. Browser pool recovers from Chromium crashes.

## Rollout
1. Phase 1: internal loop + policy + trace (`disabled` by default). ✅
2. Phase 2: provider adapters enabled via env.
3. Phase 3: ghost protocol + vision fallback.
4. Phase 4: live browser stream.
5. Phase 5: hardening/perf tuning + soak tests.

## Acceptance Criteria
1. External clients remain backward compatible.
2. Internal mode solves bounded multi-step tasks.
3. Policy-denied actions are blocked and logged with reason.
4. Shared trace schema for both internal and external modes.
5. Internal mode requires explicit opt-in.
6. Ghost protocol extracts content from pages that block DOM scraping.
7. Ghost mode requires explicit opt-in (`allow_ghost` / `AGENT_GHOST_ENABLED`).
8. Live stream delivers real-time viewport visibility to connected clients.
