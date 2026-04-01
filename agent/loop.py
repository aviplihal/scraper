"""Agent loop — Qwen3 9B via Ollama with native tool calling in a ReAct pattern.

Flow per step:
  1. Send current messages (system + history) to Ollama with tool definitions.
  2. If the model returns tool_calls, execute each one and append tool results.
  3. If the model returns no tool_calls, it has finished reasoning — break.
  4. Repeat up to MAX_STEPS times.
"""

import json
import logging

import ollama

from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from tools.registry import TOOL_DEFINITIONS, ToolContext, dispatch_tool

logger = logging.getLogger(__name__)

MODEL      = "qwen3.5:9b"
MAX_STEPS  = 30


async def run_agent_loop(config: dict, source: str, ctx: ToolContext) -> dict:
    """Run the agent until it finishes or hits MAX_STEPS."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(config, source)},
    ]

    client = ollama.AsyncClient()
    follow_through_reminders = 0
    run_result = {
        "steps_run": 0,
        "status": "unknown",
        "stop_reason": "",
    }

    print("\n=== Job Start ===")
    print(f"Client    : {config['client_id']}")
    print(f"Source    : {source}")
    print(f"Model     : {MODEL}")
    print(f"Job       : {config['job']}")
    print(f"Max steps : {MAX_STEPS}\n")

    for step in range(1, MAX_STEPS + 1):
        run_result["steps_run"] = step
        logger.debug("Step %d / %d", step, MAX_STEPS)

        try:
            response = await client.chat(
                model=MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                options={"temperature": 0.1, "num_predict": 2048},
            )
        except Exception as exc:
            logger.error("Ollama call failed at step %d: %s", step, exc)
            print(f"[agent] ERROR: Ollama call failed — {exc}")
            run_result["status"] = "error"
            run_result["stop_reason"] = f"Ollama call failed: {exc}"
            break

        message = response.message

        # Print any reasoning text the model produced
        if message.content:
            print(f"[step {step}] {message.content.strip()}", flush=True)

        # Append the assistant message to history
        messages.append({"role": "assistant", "content": message.content or "", "tool_calls": message.tool_calls})

        if not message.tool_calls:
            if _should_request_follow_through(ctx) and follow_through_reminders < 2:
                follow_through_reminders += 1
                reminder = _build_follow_through_reminder(ctx)
                print(f"[agent] Reminder: {reminder}", flush=True)
                messages.append({"role": "user", "content": reminder})
                continue

            print(f"\n[agent] Agent finished after {step} step(s) — no further tool calls.")
            run_result["status"] = "completed"
            run_result["stop_reason"] = "Agent stopped making tool calls."
            break

        # Execute tool calls and append results
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            raw_args  = tool_call.function.arguments

            # Ollama may return arguments as a dict or as a JSON string
            if isinstance(raw_args, str):
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError:
                    arguments = {}
            else:
                arguments = raw_args or {}

            print(f"[step {step}] → {tool_name}({_fmt_args(arguments)})", flush=True)

            result = await dispatch_tool(tool_name, arguments, ctx)
            result_str = json.dumps(result, ensure_ascii=False)

            print(f"[step {step}] ← {result_str[:300]}", flush=True)

            messages.append({"role": "tool", "content": result_str})

            # fail_url signals we should stop processing the current URL
            if ctx.failed_url_flag:
                ctx.failed_url_flag = False

    else:
        print(f"\n[agent] MAX_STEPS ({MAX_STEPS}) reached — job stopped.")
        run_result["status"] = "max_steps"
        run_result["stop_reason"] = f"Reached max step limit ({MAX_STEPS})."

    if run_result["status"] == "unknown":
        run_result["status"] = "completed"
        run_result["stop_reason"] = "Run exited without an explicit stop reason."

    return run_result


def _fmt_args(args: dict) -> str:
    """Format tool arguments for terminal display (truncated)."""
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 80:
            v_str = v_str[:77] + "..."
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)


def _should_request_follow_through(ctx: ToolContext) -> bool:
    """Return True when fetched pages remain unprocessed."""
    return bool(_unprocessed_fetch_ids(ctx))


def _unprocessed_fetch_ids(ctx: ToolContext) -> list[str]:
    """Return fetched pages that have not been handled by list_links/parse_html/fail_url."""
    return [
        fetch_id
        for fetch_id in ctx.fetch_metadata
        if fetch_id not in ctx.processed_fetch_ids
    ]


def _build_follow_through_reminder(ctx: ToolContext) -> str:
    """Build a targeted reminder telling the model how to process fetched pages."""
    instructions: list[str] = []
    for fetch_id in _unprocessed_fetch_ids(ctx)[:3]:
        metadata = ctx.fetch_metadata.get(fetch_id, {})
        page_kind = metadata.get("page_kind", "unknown")
        url = metadata.get("final_url") or metadata.get("url") or "unknown URL"

        if page_kind in {"search_results", "directory"}:
            action = (
                f"{fetch_id} ({page_kind}) {url}: call list_links on this page to discover candidate profile/detail URLs."
            )
        elif page_kind == "profile":
            action = (
                f"{fetch_id} ({page_kind}) {url}: call parse_html on this detail/profile page, then save_result if the name is real."
            )
        else:
            action = (
                f"{fetch_id} ({page_kind}) {url}: call fail_url if it is blocked, irrelevant, not found, or listing-only."
            )
        instructions.append(action)

    summary = " ".join(instructions)
    return (
        "You have fetched pages that are still unprocessed. "
        "Use list_links on search_results/directory pages, parse_html on profile pages, "
        "and fail_url on blocked, irrelevant, listing-only, or not-found pages. "
        f"Outstanding pages: {summary}"
    )
