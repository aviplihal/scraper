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
from tools.registry import TOOL_DEFINITIONS, ToolContext, _curated_target_pool_exhausted, dispatch_tool

logger = logging.getLogger(__name__)

MODEL      = "qwen3.5:9b"
MAX_STEPS  = 100


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
    print(f"Lead target: {_saved_lead_count(ctx)}/{_lead_target(ctx)} viable leads")
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
            if await _try_automatic_profile_processing(ctx, messages, step):
                if _lead_target_reached(ctx):
                    print(
                        "[agent] Recovered from Ollama tool-call failure by automatically "
                        "processing outstanding profile pages and reaching the lead target.",
                        flush=True,
                    )
                    run_result["status"] = "completed"
                    run_result["stop_reason"] = _lead_target_stop_reason(ctx)
                else:
                    print(
                        "[agent] Automatic profile processing recovered what it could after an "
                        "Ollama tool-call failure, but the lead target is still unmet.",
                        flush=True,
                    )
                    run_result["status"] = "completed"
                    run_result["stop_reason"] = _under_target_stop_reason(
                        ctx,
                        "Recovered from Ollama tool-call failure but could not reach the lead target",
                    )
                break
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

            if await _auto_fail_remaining_non_actionable_pages(ctx, messages, step):
                if _should_request_follow_through(ctx):
                    continue

            if await _try_automatic_profile_processing(ctx, messages, step):
                if _lead_target_reached(ctx):
                    print(
                        "[agent] Automatic profile processing completed for outstanding fetched "
                        "profile pages and reached the lead target.",
                        flush=True,
                    )
                    run_result["status"] = "completed"
                    run_result["stop_reason"] = _lead_target_stop_reason(ctx)
                else:
                    print(
                        "[agent] Automatic profile processing completed for outstanding fetched "
                        "profile pages, but the lead target is still unmet.",
                        flush=True,
                    )
                    run_result["status"] = "completed"
                    run_result["stop_reason"] = _under_target_stop_reason(
                        ctx,
                        "Automatic profile processing completed but could not reach the lead target",
                    )
                break

            print(f"\n[agent] Agent finished after {step} step(s) — no further tool calls.")
            run_result["status"] = "completed"
            run_result["stop_reason"] = (
                _lead_target_stop_reason(ctx)
                if _lead_target_reached(ctx)
                else _under_target_stop_reason(ctx, "Agent stopped making tool calls")
            )
            break

        # Execute tool calls and append results
        reached_target_this_step = False
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

            if _lead_target_reached(ctx):
                reached_target_this_step = True
                run_result["status"] = "completed"
                run_result["stop_reason"] = _lead_target_stop_reason(ctx)
                print(
                    f"\n[agent] Lead target reached after step {step} — job stopped.",
                    flush=True,
                )
                break

        if reached_target_this_step:
            break

    else:
        print(f"\n[agent] MAX_STEPS ({MAX_STEPS}) reached — job stopped.")
        run_result["status"] = "max_steps"
        run_result["stop_reason"] = _under_target_stop_reason(
            ctx,
            f"Reached max step limit ({MAX_STEPS})",
        )

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
    return (
        _needs_target_suggestions(ctx)
        or _needs_target_fetch_follow_through(ctx)
        or bool(_unprocessed_fetch_ids(ctx))
    )


def _unprocessed_fetch_ids(ctx: ToolContext) -> list[str]:
    """Return fetched pages that have not been handled by list_links/parse_html/fail_url."""
    return [
        fetch_id
        for fetch_id in ctx.fetch_metadata
        if fetch_id not in ctx.processed_fetch_ids
    ]


def _build_follow_through_reminder(ctx: ToolContext) -> str:
    """Build a targeted reminder telling the model how to process fetched pages."""
    if _needs_target_suggestions(ctx):
        return (
            f"Progress: saved {_saved_lead_count(ctx)}/{_lead_target(ctx)} viable leads. "
            "You are in a website=NA run. Call suggest_targets first to get the curated "
            "starter target list before fetching pages."
        )

    if _needs_target_fetch_follow_through(ctx):
        pending_targets = []
        for target in ctx.suggested_targets[:3]:
            url = str(target.get("url", ""))
            normalized = url.strip()
            if normalized and normalized not in ctx.visited_urls:
                pending_targets.append(url)
        preview = " ".join(pending_targets[:3])
        return (
            f"Progress: saved {_saved_lead_count(ctx)}/{_lead_target(ctx)} viable leads. "
            "You already called suggest_targets, but you have not fetched enough curated starter targets yet. "
            "Fetch 1 to 2 of the highest-priority suggested targets now and continue from their results. "
            f"Suggested starter URLs: {preview}"
        )

    instructions: list[str] = []
    for fetch_id in _unprocessed_fetch_ids(ctx)[:3]:
        metadata = ctx.fetch_metadata.get(fetch_id, {})
        page_kind = metadata.get("page_kind", "unknown")
        url = metadata.get("final_url") or metadata.get("url") or "unknown URL"

        if page_kind in {"search_results", "directory", "company_directory", "company_page"}:
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
        f"Progress: saved {_saved_lead_count(ctx)}/{_lead_target(ctx)} viable leads. "
        "You have fetched pages that are still unprocessed. "
        "Use list_links on search_results/directory pages, parse_html on profile pages, "
        "and fail_url on blocked, irrelevant, listing-only, or not-found pages. "
        f"Outstanding pages: {summary}"
    )


async def _try_automatic_profile_processing(ctx: ToolContext, messages: list[dict], step: int) -> bool:
    """Process outstanding profile pages directly when the model fails to continue."""
    profile_fetch_ids = [
        fetch_id
        for fetch_id in _unprocessed_fetch_ids(ctx)
        if ctx.fetch_metadata.get(fetch_id, {}).get("page_kind") == "profile"
    ]
    if not profile_fetch_ids:
        return False

    field_names = list(ctx.client_config.get("fields", {}).keys())
    processed_any = False

    for fetch_id in profile_fetch_ids:
        metadata = ctx.fetch_metadata.get(fetch_id, {})
        url = metadata.get("final_url") or metadata.get("url") or ""
        parse_args = {"fetch_id": fetch_id, "field_names": field_names}
        print(f"[step {step}] → parse_html({_fmt_args(parse_args)}) [auto]", flush=True)
        parse_result = await dispatch_tool("parse_html", parse_args, ctx)
        parse_result_str = json.dumps(parse_result, ensure_ascii=False)
        print(f"[step {step}] ← {parse_result_str[:300]} [auto]", flush=True)
        messages.append({"role": "tool", "content": parse_result_str})

        fields = parse_result.get("fields", {})
        name = fields.get("name") if isinstance(fields, dict) else None
        if _looks_saveable_name(name):
            save_args = {"url": url, "data": fields}
            print(f"[step {step}] → save_result({_fmt_args(save_args)}) [auto]", flush=True)
            save_result = await dispatch_tool("save_result", save_args, ctx)
            save_result_str = json.dumps(save_result, ensure_ascii=False)
            print(f"[step {step}] ← {save_result_str[:300]} [auto]", flush=True)
            messages.append({"role": "tool", "content": save_result_str})
        else:
            fail_args = {
                "url": url,
                "reason": "Automatic profile processing could not extract a usable person identifier.",
            }
            print(f"[step {step}] → fail_url({_fmt_args(fail_args)}) [auto]", flush=True)
            fail_result = await dispatch_tool("fail_url", fail_args, ctx)
            fail_result_str = json.dumps(fail_result, ensure_ascii=False)
            print(f"[step {step}] ← {fail_result_str[:300]} [auto]", flush=True)
            messages.append({"role": "tool", "content": fail_result_str})

        processed_any = True
        if _lead_target_reached(ctx):
            break

    return processed_any


async def _auto_fail_remaining_non_actionable_pages(
    ctx: ToolContext,
    messages: list[dict],
    step: int,
) -> bool:
    """Automatically fail leftover blocked or non-discovery pages before ending a run."""
    non_actionable_kinds = {"blocked", "job_board", "landing_page", "article_or_news", "not_found"}
    pending = [
        fetch_id
        for fetch_id in _unprocessed_fetch_ids(ctx)
        if ctx.fetch_metadata.get(fetch_id, {}).get("page_kind") in non_actionable_kinds
    ]
    if not pending:
        return False

    processed_any = False
    for fetch_id in pending:
        metadata = ctx.fetch_metadata.get(fetch_id, {})
        url = metadata.get("final_url") or metadata.get("url") or ""
        page_kind = metadata.get("page_kind", "unknown")
        reason = f"Automatic cleanup: {page_kind} page is not a viable discovery target."
        fail_args = {"url": url, "reason": reason}
        print(f"[step {step}] → fail_url({_fmt_args(fail_args)}) [auto]", flush=True)
        fail_result = await dispatch_tool("fail_url", fail_args, ctx)
        fail_result_str = json.dumps(fail_result, ensure_ascii=False)
        print(f"[step {step}] ← {fail_result_str[:300]} [auto]", flush=True)
        messages.append({"role": "tool", "content": fail_result_str})
        processed_any = True

    return processed_any


def _looks_saveable_name(value: object) -> bool:
    """Return True when an extracted name is good enough to save."""
    if not isinstance(value, str):
        return False
    value = value.strip()
    if len(value) < 2:
        return False
    banned = {"senior software engineer", "software engineer", "developer", "engineer"}
    return value.lower() not in banned


def _lead_target(ctx: ToolContext) -> int:
    """Return the configured viable lead target for this run."""
    return int(ctx.client_config["min_leads"])


def _saved_lead_count(ctx: ToolContext) -> int:
    """Return how many viable new leads were saved in this run."""
    return int(getattr(ctx.sheets_writer, "saved_count", 0))


def _lead_target_reached(ctx: ToolContext) -> bool:
    """Return True once the run has saved enough viable new leads."""
    return _saved_lead_count(ctx) >= _lead_target(ctx)


def _lead_target_stop_reason(ctx: ToolContext) -> str:
    """Build a success stop reason once the viable lead target is met."""
    return (
        f"Reached lead target with {_saved_lead_count(ctx)}/{_lead_target(ctx)} "
        "viable new leads saved."
    )


def _under_target_stop_reason(ctx: ToolContext, prefix: str) -> str:
    """Build a consistent incomplete stop reason when the target is unmet."""
    if _curated_target_pool_exhausted(ctx):
        return (
            "Curated target pool exhausted before reaching the lead target "
            f"({_saved_lead_count(ctx)}/{_lead_target(ctx)} viable new leads saved)."
        )
    return (
        f"{prefix} before reaching the lead target "
        f"({_saved_lead_count(ctx)}/{_lead_target(ctx)} viable new leads saved)."
    )


def _needs_target_suggestions(ctx: ToolContext) -> bool:
    """Return True when a website=NA run has not yet requested curated targets."""
    return (
        ctx.source_mode in {"web", "human_emulator", "all"}
        and str(ctx.client_config.get("website", "NA")).upper() == "NA"
        and not ctx.suggest_targets_called
    )


def _needs_target_fetch_follow_through(ctx: ToolContext) -> bool:
    """Return True when curated starter targets were suggested but not yet fetched."""
    if not (
        ctx.source_mode in {"web", "human_emulator", "all"}
        and str(ctx.client_config.get("website", "NA")).upper() == "NA"
        and ctx.suggest_targets_called
        and ctx.suggested_target_urls
    ):
        return False

    visited = ctx.visited_urls
    fetched_targets = sum(1 for url in ctx.suggested_target_urls if url in visited)
    if fetched_targets >= min(2, len(ctx.suggested_target_urls)):
        return False

    return not _lead_target_reached(ctx)
