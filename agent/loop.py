"""Agent loop — Qwen3 9B via Ollama with native tool calling in a ReAct pattern.

Flow per step:
  1. Send current messages (system + history) to Ollama with tool definitions.
  2. If the model returns tool_calls, execute each one and append tool results.
  3. If the model returns no tool_calls, it has finished reasoning — break.
  4. Repeat up to MAX_STEPS times.
"""

import asyncio
import json
import logging
import re

import ollama

from agent.prompts import SYSTEM_PROMPT, build_user_prompt
from tools.registry import (
    TOOL_DEFINITIONS,
    ToolContext,
    _curated_target_pool_exhausted,
    _domain_for_url,
    _fetch_budget_for_url,
    _fetch_budget_key,
    _normalize_url,
    dispatch_tool,
)

logger = logging.getLogger(__name__)

MODEL      = "qwen3.5:9b"
MAX_STEPS  = 100
_PROMPT_TOKEN_COMPACT_THRESHOLD = 2500
_MESSAGE_COMPACT_THRESHOLD = 12
_MESSAGE_COMPACT_HARD_THRESHOLD = 22
_COMPACTION_MIN_NEW_MESSAGES = 10
_AUTO_PROFILE_BATCH_SIZE = 3
_FAKE_TOOL_CALL_PATTERN = re.compile(
    r"\b(fetch_page|fetch_url|list_links|parse_html|save_result|fail_url|suggest_targets|search|finish_run|finish_job)\s*\("
)
_RECOVERABLE_MODEL_ERROR_PATTERNS = (
    "xml syntax error",
    "closed by </parameter>",
    "closed by </function>",
)


async def run_agent_loop(config: dict, source: str, ctx: ToolContext) -> dict:
    """Run the agent until it finishes or hits MAX_STEPS."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_prompt(config, source)},
    ]

    client = ollama.AsyncClient()
    follow_through_reminders = 0
    last_follow_through_signature: tuple[int, int, int, int, int, str] | None = None
    run_result = {
        "steps_run": 0,
        "status": "unknown",
        "stop_reason": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "max_prompt_tokens": 0,
        "compactions": 0,
        "last_compaction_non_system_messages": 0,
    }

    print("\n=== Job Start ===")
    print(f"Client    : {config['client_id']}")
    print(f"Source    : {source}")
    print(f"Model     : {MODEL}")
    print(f"Job       : {config['job']}")
    print(f"Lead target: {_saved_lead_count(ctx)}/{_lead_target(ctx)} viable leads")
    print(f"Source phase: {ctx.source_phase}")
    print(f"Max steps : {MAX_STEPS}\n")

    for step in range(1, MAX_STEPS + 1):
        run_result["steps_run"] = step
        logger.debug("Step %d / %d", step, MAX_STEPS)
        _maybe_compact_messages(messages, ctx, run_result)

        try:
            response = await client.chat(
                model=MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                options={"temperature": 0.1, "num_predict": 2048},
            )
            _accumulate_token_usage(run_result, response)
        except asyncio.CancelledError:
            run_result["status"] = "interrupted"
            run_result["stop_reason"] = "Run interrupted while waiting for the model."
            raise
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
            if await _recover_from_model_call_error(ctx, messages, step, exc, run_result):
                continue
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
            if await _try_automatic_profile_processing(ctx, messages, step):
                if _lead_target_reached(ctx):
                    print(
                        "[agent] Automatic profile processing completed for outstanding fetched "
                        "profile pages and reached the lead target.",
                        flush=True,
                    )
                    run_result["status"] = "completed"
                    run_result["stop_reason"] = _lead_target_stop_reason(ctx)
                    break
                if _should_request_follow_through(ctx):
                    continue

            if await _auto_fail_remaining_non_actionable_pages(ctx, messages, step):
                if _should_request_follow_through(ctx):
                    continue

            current_signature = _follow_through_signature(ctx)
            if current_signature != last_follow_through_signature:
                follow_through_reminders = 0
                last_follow_through_signature = current_signature

            if _should_request_follow_through(ctx) and follow_through_reminders < 2:
                follow_through_reminders += 1
                reminder = (
                    _build_fake_tool_call_correction(ctx)
                    if _looks_like_fake_tool_calls(message.content or "")
                    else _build_follow_through_reminder(ctx)
                )
                print(f"[agent] Reminder: {reminder}", flush=True)
                messages.append({"role": "user", "content": reminder})
                continue

            if _maybe_switch_to_discovery_phase(ctx, messages):
                continue

            if _no_viable_next_actions(ctx):
                print(f"\n[agent] Agent finished after {step} step(s) — no viable next actions remained.")
                run_result["status"] = "completed"
                run_result["stop_reason"] = _under_target_stop_reason(
                    ctx,
                    "No actionable pages or fetchable candidate targets remained",
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

        reached_target_this_step = await _execute_tool_calls(
            step,
            message.tool_calls,
            ctx,
            messages,
            run_result,
        )
        follow_through_reminders = 0
        last_follow_through_signature = _follow_through_signature(ctx)
        if reached_target_this_step:
            break

        if _no_viable_next_actions(ctx):
            print(f"\n[agent] Agent finished after step {step} — no viable next actions remained.")
            run_result["status"] = "completed"
            run_result["stop_reason"] = _under_target_stop_reason(
                ctx,
                "No actionable pages or fetchable candidate targets remained",
            )
            break

        if await _try_automatic_profile_processing(ctx, messages, step):
            if _lead_target_reached(ctx):
                print(
                    f"\n[agent] Lead target reached after step {step} — job stopped.",
                    flush=True,
                )
                run_result["status"] = "completed"
                run_result["stop_reason"] = _lead_target_stop_reason(ctx)
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


async def _execute_tool_calls(
    step: int,
    tool_calls: list,
    ctx: ToolContext,
    messages: list[dict],
    run_result: dict,
) -> bool:
    """Execute model-emitted tool calls, batching safe I/O work concurrently."""
    index = 0
    while index < len(tool_calls):
        batch = [tool_calls[index]]
        batch_tool_name = _normalized_tool_name(tool_calls[index].function.name)
        index += 1
        while (
            index < len(tool_calls)
            and _normalized_tool_name(tool_calls[index].function.name) == batch_tool_name
            and batch_tool_name in {"fetch_page", "parse_html", "save_result"}
        ):
            batch.append(tool_calls[index])
            index += 1

        if len(batch) > 1 and batch_tool_name in {"fetch_page", "parse_html", "save_result"}:
            results = await _run_tool_batch_concurrently(step, batch_tool_name, batch, ctx)
        else:
            results = [await _run_single_tool_call(step, batch[0], ctx)]

        for tool_name, result in results:
            if tool_name == "finish_run":
                run_result["status"] = "completed"
                reason = str(result.get("reason") or "").strip()
                run_result["stop_reason"] = (
                    reason
                    if reason
                    else (
                        _lead_target_stop_reason(ctx)
                        if _lead_target_reached(ctx)
                        else _under_target_stop_reason(ctx, "Agent requested an early finish")
                    )
                )
                print(
                    f"\n[agent] Agent finished after step {step} — requested early completion.",
                    flush=True,
                )
                return True

            if tool_name == "suggest_targets" and "error" not in result:
                _print_targeting_brief(result)

            messages.append(
                {
                    "role": "tool",
                    "content": _tool_history_content(tool_name, result, ctx),
                }
            )

            if "error" not in result:
                # Reset reminder pressure once we make progress.
                pass

            if ctx.failed_url_flag:
                ctx.failed_url_flag = False

            if _lead_target_reached(ctx):
                run_result["status"] = "completed"
                run_result["stop_reason"] = _lead_target_stop_reason(ctx)
                print(
                    f"\n[agent] Lead target reached after step {step} — job stopped.",
                    flush=True,
                )
                return True

    return False


async def _run_tool_batch_concurrently(
    step: int,
    batch_tool_name: str,
    tool_calls: list,
    ctx: ToolContext,
) -> list[tuple[str, dict]]:
    """Run a safe same-tool batch concurrently and return results in call order."""
    semaphore = asyncio.Semaphore(2)

    async def _run(tool_call) -> tuple[str, dict]:
        async with semaphore:
            return await _run_single_tool_call(step, tool_call, ctx)

    return list(await asyncio.gather(*[_run(tool_call) for tool_call in tool_calls]))


async def _run_single_tool_call(step: int, tool_call, ctx: ToolContext) -> tuple[str, dict]:
    """Execute one tool call and print its terminal trace."""
    tool_name = _normalized_tool_name(tool_call.function.name)
    arguments = _tool_arguments(tool_call)

    if tool_name == "finish_run":
        result = {
            "status": "finish_requested",
            "reason": str(arguments.get("reason") or "").strip(),
        }
        print(f"[step {step}] → {tool_name}({_fmt_args(arguments)})", flush=True)
        result_str = json.dumps(result, ensure_ascii=False)
        print(f"[step {step}] ← {result_str[:300]}", flush=True)
        return tool_name, result

    print(f"[step {step}] → {tool_name}({_fmt_args(arguments)})", flush=True)
    result = await dispatch_tool(tool_name, arguments, ctx)
    result_str = json.dumps(result, ensure_ascii=False)
    print(f"[step {step}] ← {result_str[:300]}", flush=True)
    return tool_name, result


def _normalized_tool_name(tool_name: str) -> str:
    """Normalize model-emitted tool names to the supported registry names."""
    if tool_name == "fetch_url":
        return "fetch_page"
    if tool_name == "finish_job":
        return "finish_run"
    if tool_name == "search":
        return "suggest_targets"
    return tool_name


def _tool_arguments(tool_call) -> dict:
    """Normalize model-emitted tool arguments into a plain dict."""
    raw_args = tool_call.function.arguments
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {}
    return raw_args or {}


def _accumulate_token_usage(run_result: dict, response: object) -> None:
    """Accumulate Ollama prompt/completion token counts when available."""
    prompt_tokens = _safe_int(getattr(response, "prompt_eval_count", None))
    completion_tokens = _safe_int(getattr(response, "eval_count", None))

    run_result["last_prompt_tokens"] = prompt_tokens
    run_result["prompt_tokens"] += prompt_tokens
    run_result["completion_tokens"] += completion_tokens
    run_result["max_prompt_tokens"] = max(run_result.get("max_prompt_tokens", 0), prompt_tokens)
    run_result["total_tokens"] = (
        run_result["prompt_tokens"] + run_result["completion_tokens"]
    )


def _safe_int(value: object) -> int:
    """Convert Ollama numeric metadata to an int, defaulting to zero."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _maybe_compact_messages(messages: list[dict], ctx: ToolContext, run_result: dict) -> None:
    """Compact older conversation history when prompt state grows too large."""
    non_system_messages = max(0, len(messages) - 2)
    growth_since_compaction = non_system_messages - run_result.get("last_compaction_non_system_messages", 0)
    prompt_trigger = (
        run_result.get("last_prompt_tokens", 0) > _PROMPT_TOKEN_COMPACT_THRESHOLD
        and non_system_messages >= _MESSAGE_COMPACT_THRESHOLD
        and growth_since_compaction >= _COMPACTION_MIN_NEW_MESSAGES
    )
    message_trigger = non_system_messages >= _MESSAGE_COMPACT_HARD_THRESHOLD
    if not prompt_trigger and not message_trigger:
        return
    if len(messages) <= 3:
        return

    tail = messages[2:][-6:]
    summary_message = {
        "role": "user",
        "content": _conversation_state_summary(ctx),
    }
    messages[:] = messages[:2] + [summary_message] + tail
    run_result["compactions"] = run_result.get("compactions", 0) + 1
    run_result["last_compaction_non_system_messages"] = max(0, len(messages) - 2)


def _force_compact_messages(messages: list[dict], ctx: ToolContext, run_result: dict) -> None:
    """Compact history immediately after a model-format failure so the retry starts cleaner."""
    if len(messages) <= 3:
        return
    tail = messages[2:][-6:]
    summary_message = {
        "role": "user",
        "content": _conversation_state_summary(ctx),
    }
    messages[:] = messages[:2] + [summary_message] + tail
    run_result["compactions"] = run_result.get("compactions", 0) + 1
    run_result["last_compaction_non_system_messages"] = max(0, len(messages) - 2)


def _is_recoverable_model_call_error(exc: Exception) -> bool:
    """Return True for malformed structured-output failures that are worth one retry."""
    message = str(exc).lower()
    return any(pattern in message for pattern in _RECOVERABLE_MODEL_ERROR_PATTERNS)


async def _recover_from_model_call_error(
    ctx: ToolContext,
    messages: list[dict],
    step: int,
    exc: Exception,
    run_result: dict,
) -> bool:
    """Try one lightweight recovery pass after malformed model tool output."""
    if not _is_recoverable_model_call_error(exc):
        return False
    if run_result.get("model_error_retries", 0) >= 1:
        return False

    await _auto_fail_remaining_non_actionable_pages(ctx, messages, step)
    _force_compact_messages(messages, ctx, run_result)
    reminder = (
        "The previous model response failed because its tool-call markup was malformed. "
        "Continue with direct tool calls only. Process any outstanding fetched pages first, "
        "then continue from the current discovery seed. Do not emit XML-like markup or pseudo-tool calls."
    )
    messages.append({"role": "user", "content": reminder})
    run_result["model_error_retries"] = run_result.get("model_error_retries", 0) + 1
    print(
        f"[agent] Recovering from malformed model tool output at step {step}; compacting history and retrying once.",
        flush=True,
    )
    return True


def _conversation_state_summary(ctx: ToolContext) -> str:
    """Build a compact state summary that preserves the active scrape context."""
    keyword_brief = ctx.keyword_brief or {}
    primary_terms = ", ".join(keyword_brief.get("primary_terms", [])) or "n/a"
    secondary_terms = ", ".join(keyword_brief.get("secondary_terms", [])) or "n/a"
    allowed_domains = ", ".join(ctx.candidate_domains or sorted(ctx.allowed_domains)) or "n/a"
    avoid_domains = ", ".join(ctx.avoid_domains) or "none"
    low_yield = ", ".join(sorted(ctx.low_yield_platforms)) or "none"

    unprocessed = []
    for fetch_id in _unprocessed_fetch_ids(ctx)[:5]:
        metadata = ctx.fetch_metadata.get(fetch_id, {})
        url = metadata.get("final_url") or metadata.get("url") or "unknown"
        page_kind = metadata.get("page_kind", "unknown")
        unprocessed.append(f"{fetch_id}:{page_kind}:{url}")
    parsed_pending = []
    for fetch_id, fields in list(ctx.parsed_results.items())[-3:]:
        metadata = ctx.fetch_metadata.get(fetch_id, {})
        url = metadata.get("final_url") or metadata.get("url") or "unknown"
        parsed_pending.append(f"{fetch_id}:{url}:{json.dumps(fields, ensure_ascii=False)[:160]}")
    failures = [f"{item['reason']}:{item['url']}" for item in ctx.failed_urls[-5:]]

    return (
        "Context summary for the ongoing lead scrape.\n"
        f"Progress: saved={_saved_lead_count(ctx)}/{_lead_target(ctx)} viable leads.\n"
        f"Source phase: {ctx.source_phase}. Strategy: {ctx.target_strategy or 'unknown'}.\n"
        f"Keyword brief: primary={primary_terms}; secondary={secondary_terms}; area={keyword_brief.get('area', 'NA')}.\n"
        f"Candidate domains: {allowed_domains}. Avoid: {avoid_domains}. Low-yield platforms: {low_yield}.\n"
        f"Unprocessed fetches: {' | '.join(unprocessed) if unprocessed else 'none'}.\n"
        f"Parsed pending results: {' | '.join(parsed_pending) if parsed_pending else 'none'}.\n"
        f"Recent failures: {' | '.join(failures) if failures else 'none'}."
    )


def _tool_history_content(tool_name: str, result: dict, ctx: ToolContext) -> str:
    """Return a token-lean tool message for model history."""
    if "error" in result:
        return json.dumps(
            {
                "tool": tool_name,
                "error": result.get("error"),
                "url": result.get("url"),
                "arguments": result.get("arguments"),
            },
            ensure_ascii=False,
        )

    if tool_name == "suggest_targets":
        payload = {
            "tool": tool_name,
            "status": result.get("status", "ok"),
            "phase": result.get("phase"),
            "strategy": result.get("strategy"),
            "keyword_brief": result.get("keyword_brief"),
            "allowed_domains": result.get("allowed_domains"),
            "candidate_targets": [
                {
                    "url": target.get("url"),
                    "domain": target.get("domain"),
                    "source": target.get("source"),
                }
                for target in result.get("candidate_targets", [])[:3]
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    if tool_name == "fetch_page":
        return json.dumps(
            {
                "tool": tool_name,
                "fetch_id": result.get("fetch_id"),
                "url": result.get("url"),
                "final_url": result.get("final_url"),
                "title": result.get("title"),
                "page_kind": result.get("page_kind"),
                "preview": str(result.get("preview", ""))[:200],
                "exhausted": result.get("exhausted", False),
            },
            ensure_ascii=False,
        )

    if tool_name == "list_links":
        return json.dumps(
            {
                "tool": tool_name,
                "count": result.get("count"),
                "exhausted": result.get("exhausted", False),
                "links": result.get("links", [])[:5],
            },
            ensure_ascii=False,
        )

    if tool_name == "parse_html":
        return json.dumps(
            {
                "tool": tool_name,
                "fields": result.get("fields"),
                "cached": result.get("cached", False),
            },
            ensure_ascii=False,
        )

    if tool_name in {"save_result", "fail_url"}:
        return json.dumps(
            {
                "tool": tool_name,
                "status": result.get("status"),
                "url": result.get("url"),
                "reason": result.get("reason"),
            },
            ensure_ascii=False,
        )

    return json.dumps(result, ensure_ascii=False)


def _looks_like_fake_tool_calls(content: str) -> bool:
    """Return True when the model printed tool-shaped prose instead of real tool calls."""
    return bool(content and _FAKE_TOOL_CALL_PATTERN.search(content))


def _build_fake_tool_call_correction(ctx: ToolContext) -> str:
    """Return a narrow reminder when the model printed tool calls as prose/code blocks."""
    return (
        f"Progress: saved {_saved_lead_count(ctx)}/{_lead_target(ctx)} viable leads. "
        "You must call tools directly instead of printing code blocks or pseudo-tool calls. "
        "If profile pages are already fetched, parse them now. Otherwise continue from the current discovery page."
    )


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
        or bool(_remaining_discovered_profile_urls(ctx))
        or bool(_remaining_discovery_fetch_ids(ctx))
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
            "You are in a website=NA run. Call suggest_targets first to get the keyword brief "
            "and candidate domains before fetching pages."
        )

    if _unprocessed_fetch_ids(ctx):
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

    discovered_profile_urls = _remaining_discovered_profile_urls(ctx)
    if discovered_profile_urls:
        preview = " ".join(discovered_profile_urls[:5])
        fetch_count = min(5, max(2, _lead_target(ctx) - _saved_lead_count(ctx)))
        return (
            f"Progress: saved {_saved_lead_count(ctx)}/{_lead_target(ctx)} viable leads. "
            "Fetch the discovered profile/detail URLs from the current search pages before asking for new targets. "
            f"Fetch up to {fetch_count} of these next: {preview}"
        )

    if _remaining_discovery_fetch_ids(ctx):
        discovery_pages = []
        for fetch_id in _remaining_discovery_fetch_ids(ctx)[:2]:
            metadata = ctx.fetch_metadata.get(fetch_id, {})
            url = metadata.get("final_url") or metadata.get("url") or "unknown URL"
            page_kind = metadata.get("page_kind", "unknown")
            discovery_pages.append(
                f"{fetch_id} ({page_kind}) {url}: call list_links again to continue from unseen candidates on this page."
            )
        return (
            f"Progress: saved {_saved_lead_count(ctx)}/{_lead_target(ctx)} viable leads. "
            "Continue from the current discovery/search pages before asking for new targets. "
            f"Discovery pages with unseen candidates: {' '.join(discovery_pages)}"
        )

    if _needs_target_fetch_follow_through(ctx):
        preview = " ".join(_candidate_preview_urls(ctx, limit=3))
        if not preview:
            return (
                f"Progress: saved {_saved_lead_count(ctx)}/{_lead_target(ctx)} viable leads. "
                "No fetchable starter URLs remain in the current phase. Stop if no other actionable pages remain."
            )
        keyword_brief = ctx.keyword_brief or {}
        primary_terms = ", ".join(keyword_brief.get("primary_terms", [])) or "the target persona"
        domain_switch_note = ""
        switch_domain = _switch_candidate_domain(ctx)
        if switch_domain:
            domain_switch_note = (
                f" Prefer a different domain/source now because {switch_domain} has not produced "
                "a viable lead yet."
            )
        return (
            f"Progress: saved {_saved_lead_count(ctx)}/{_lead_target(ctx)} viable leads. "
            f"You already called suggest_targets for {ctx.source_phase}. Choose 1 to 2 starter targets that best match the keyword brief for {primary_terms}. "
            f"Suggested starter URLs: {preview}."
            + (
                " If this is pass 2 discovery, sample up to 3 viable leads from any new source before going deeper."
                if ctx.source_phase == "discovery"
                else ""
            )
            + domain_switch_note
        )

    return (
        f"Progress: saved {_saved_lead_count(ctx)}/{_lead_target(ctx)} viable leads. "
        "Continue with the candidate domains from suggest_targets until you reach the lead target "
        "or no useful work remains."
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

    processed_any = False

    remaining_gap = max(1, _lead_target(ctx) - _saved_lead_count(ctx))
    batch_size = min(len(profile_fetch_ids), max(_AUTO_PROFILE_BATCH_SIZE, min(6, remaining_gap)))
    batch = profile_fetch_ids[:batch_size]
    parse_records = await asyncio.gather(
        *[_auto_process_profile_fetch(step, fetch_id, ctx) for fetch_id in batch]
    )
    for fetch_id, records in zip(batch, parse_records):
        for tool_name, args, result in records:
            result_str = json.dumps(result, ensure_ascii=False)
            print(f"[step {step}] → {tool_name}({_fmt_args(args)}) [auto]", flush=True)
            print(f"[step {step}] ← {result_str[:300]} [auto]", flush=True)
            messages.append({"role": "tool", "content": _tool_history_content(tool_name, result, ctx)})
        processed_any = processed_any or bool(records)
        if _lead_target_reached(ctx):
            break

        if not records:
            continue

        parse_result = records[-1][2]
        fields = parse_result.get("fields", {})
        metadata = ctx.fetch_metadata.get(fetch_id, {})
        url = metadata.get("final_url") or metadata.get("url") or ""

        if _looks_saveable_name(fields.get("name") if isinstance(fields, dict) else None):
            follow_up_name = "save_result"
            follow_up_args = {"url": url, "data": fields}
        else:
            follow_up_name = "fail_url"
            follow_up_args = {
                "url": url,
                "reason": "Automatic profile processing could not extract a usable person identifier.",
            }

        follow_up_result = await dispatch_tool(follow_up_name, follow_up_args, ctx)
        follow_up_str = json.dumps(follow_up_result, ensure_ascii=False)
        print(f"[step {step}] → {follow_up_name}({_fmt_args(follow_up_args)}) [auto]", flush=True)
        print(f"[step {step}] ← {follow_up_str[:300]} [auto]", flush=True)
        messages.append({"role": "tool", "content": _tool_history_content(follow_up_name, follow_up_result, ctx)})
        processed_any = True
        if _lead_target_reached(ctx):
            break

    return processed_any


async def _auto_process_profile_fetch(
    step: int,  # noqa: ARG001
    fetch_id: str,
    ctx: ToolContext,
) -> list[tuple[str, dict, dict]]:
    """Parse and save/fail a single fetched profile page for auto-flush."""
    field_names = list(ctx.client_config.get("fields", {}).keys())
    metadata = ctx.fetch_metadata.get(fetch_id, {})
    records: list[tuple[str, dict, dict]] = []

    parse_args = {"fetch_id": fetch_id, "field_names": field_names}
    parse_result = await dispatch_tool("parse_html", parse_args, ctx)
    records.append(("parse_html", parse_args, parse_result))

    return records


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
        messages.append({"role": "tool", "content": _tool_history_content("fail_url", fail_result, ctx)})
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
            "Keyword-driven candidate pool exhausted before reaching the lead target "
            f"({_saved_lead_count(ctx)}/{_lead_target(ctx)} viable new leads saved)."
        )
    return (
        f"{prefix} before reaching the lead target "
        f"({_saved_lead_count(ctx)}/{_lead_target(ctx)} viable new leads saved)."
    )


def _maybe_switch_to_discovery_phase(ctx: ToolContext, messages: list[dict]) -> bool:
    """Move from approved/temp-source pass 1 into discovery pass 2 when pass 1 is exhausted."""
    if ctx.source_phase != "pass1" or ctx.source_state is None:
        return False
    if _lead_target_reached(ctx):
        return False
    pass1_exhausted = _curated_target_pool_exhausted(ctx)
    if ctx.suggest_targets_called and not (ctx.candidate_domains or ctx.allowed_domains):
        pass1_exhausted = True
    if not pass1_exhausted:
        return False

    ctx.source_phase = "discovery"
    ctx.suggest_targets_called = False
    ctx.suggested_targets.clear()
    ctx.suggested_target_urls.clear()
    ctx.allowed_domains.clear()
    ctx.candidate_domains.clear()
    ctx.avoid_domains.clear()
    ctx.keyword_brief.clear()
    ctx.target_strategy = None
    ctx.source_mix = None
    print(
        "[agent] Pass 1 approved and temporary seed sources are exhausted. Switching to pass 2 discovery.",
        flush=True,
    )
    messages.append(
        {
            "role": "user",
            "content": (
                "Pass 1 is exhausted and the lead target is still unmet. "
                "Start pass 2 now: call suggest_targets again, stay close to the approved source families, "
                "sample up to 3 viable leads from any new source before deeper scraping, and keep going until "
                "you reach the lead target or useful work is exhausted."
            ),
        }
    )
    return True


def _needs_target_suggestions(ctx: ToolContext) -> bool:
    """Return True when a website=NA run has not yet requested curated targets."""
    return (
        ctx.source_mode in {"web", "human_emulator", "all"}
        and str(ctx.client_config.get("website", "NA")).upper() == "NA"
        and not ctx.suggest_targets_called
    )


def _needs_target_fetch_follow_through(ctx: ToolContext) -> bool:
    """Return True when candidate domains remain unexplored and the target is still unmet."""
    if not (
        ctx.source_mode in {"web", "human_emulator", "all"}
        and str(ctx.client_config.get("website", "NA")).upper() == "NA"
        and ctx.suggest_targets_called
        and (ctx.candidate_domains or ctx.allowed_domains)
    ):
        return False

    if _lead_target_reached(ctx):
        return False

    if _remaining_discovery_fetch_ids(ctx):
        return False

    return bool(_remaining_candidate_target_urls(ctx) or _remaining_candidate_domains(ctx))


def _print_targeting_brief(result: dict) -> None:
    """Print a concise targeting brief after suggest_targets runs."""
    phase = str(result.get("phase", "")).strip()
    if phase:
        print(f"[agent] Source phase: {phase}", flush=True)
    strategy = str(result.get("strategy", "")).strip()
    if strategy:
        print(f"[agent] Target strategy: {strategy}", flush=True)

    keyword_brief = result.get("keyword_brief", {})
    if isinstance(keyword_brief, dict):
        primary = ", ".join(keyword_brief.get("primary_terms", [])) or "n/a"
        secondary = ", ".join(keyword_brief.get("secondary_terms", [])) or "n/a"
        area = keyword_brief.get("area", "NA")
        source_mode = keyword_brief.get("source_mode", "unknown")
        print(
            f"[agent] Keyword brief: primary={primary}; secondary={secondary}; area={area}; source_mode={source_mode}",
            flush=True,
        )

    candidate_domains = result.get("allowed_domains", [])
    if isinstance(candidate_domains, list) and candidate_domains:
        print(
            f"[agent] Candidate domains: {', '.join(str(domain) for domain in candidate_domains)}",
            flush=True,
        )


def _fetched_candidate_domains(ctx: ToolContext) -> set[str]:
    """Return candidate domains that have already been fetched in this run."""
    fetched_domains = {
        _domain_for_url(str(metadata.get("final_url") or metadata.get("url") or ""))
        for metadata in ctx.fetch_metadata.values()
    }
    return {domain for domain in fetched_domains if domain in ctx.allowed_domains}


def _remaining_candidate_domains(ctx: ToolContext) -> list[str]:
    """Return allowed candidate domains that remain unfetched and unbanned."""
    remaining: list[str] = []
    fetched_domains = _fetched_candidate_domains(ctx)
    for domain in ctx.candidate_domains or sorted(ctx.allowed_domains):
        if domain in getattr(ctx, "unavailable_domains", set()):
            continue
        outcome = ctx.domain_outcomes.get(domain)
        if outcome and outcome.banned_for_run:
            continue
        if _domain_on_low_yield_cooldown(domain, ctx):
            continue
        if _domain_budget_exhausted(domain, ctx):
            continue
        if domain not in fetched_domains:
            remaining.append(domain)
    return remaining


def _remaining_candidate_target_urls(ctx: ToolContext) -> list[str]:
    """Return starter target URLs that have not been fetched or exhausted yet."""
    remaining: list[str] = []
    seen: set[str] = set()
    for target in ctx.suggested_targets:
        if not isinstance(target, dict):
            continue
        url = str(target.get("url", "")).strip()
        if not url:
            continue
        normalized_url = _normalize_url(url)
        domain = _domain_for_url(url)
        if (
            normalized_url in seen
            or domain in getattr(ctx, "unavailable_domains", set())
            or _url_budget_exhausted(url, ctx)
            or normalized_url in getattr(ctx, "exhausted_discovery_urls", set())
            or normalized_url in getattr(ctx, "url_to_fetch_id", {})
            or normalized_url in getattr(ctx, "terminal_url_outcomes", {})
        ):
            continue
        seen.add(normalized_url)
        remaining.append(url)
    return remaining


def _remaining_discovery_fetch_ids(ctx: ToolContext) -> list[str]:
    """Return discovery pages that still have unseen candidate links."""
    remaining: list[str] = []
    for fetch_id, metadata in ctx.fetch_metadata.items():
        if metadata.get("page_kind") not in {"search_results", "directory", "company_directory", "company_page"}:
            continue
        if fetch_id in ctx.exhausted_discovery_fetches:
            continue
        remaining.append(fetch_id)
    return remaining


def _remaining_discovered_profile_urls(ctx: ToolContext) -> list[str]:
    """Return discovered profile/detail URLs that are ready to fetch next."""
    ready: list[str] = []
    for normalized_url, parent_fetch_id in ctx.discovered_link_parents.items():
        if parent_fetch_id not in ctx.fetch_metadata:
            continue
        metadata = ctx.fetch_metadata.get(parent_fetch_id, {})
        if metadata.get("page_kind") not in {"search_results", "directory", "company_directory", "company_page", "unknown"}:
            continue
        if normalized_url in ctx.visited_urls or normalized_url in getattr(ctx, "terminal_url_outcomes", {}):
            continue
        if normalized_url in getattr(ctx, "url_to_fetch_id", {}):
            continue
        if normalized_url in getattr(ctx, "exhausted_discovery_urls", set()):
            continue
        if _url_budget_exhausted(normalized_url, ctx):
            continue
        ready.append(normalized_url)
    return ready


def _domain_on_low_yield_cooldown(domain: str, ctx: ToolContext) -> bool:
    """Return True when a domain maps to a run-local low-yield social platform."""
    if domain == "linkedin.com" and "linkedin" in getattr(ctx, "low_yield_platforms", set()):
        return True
    if domain in {"x.com", "twitter.com"} and "x" in getattr(ctx, "low_yield_platforms", set()):
        return True
    return False


def _candidate_preview_urls(ctx: ToolContext, limit: int = 3) -> list[str]:
    """Return a few suggested starter URLs, preferring domains not fetched yet."""
    preview_urls: list[str] = []
    remaining_target_urls = _remaining_candidate_target_urls(ctx)
    remaining_domains = _remaining_candidate_domains(ctx)
    prioritized_targets = list(ctx.suggested_targets)
    remaining_target_url_set = {url for url in remaining_target_urls}

    if remaining_domains:
        for domain in remaining_domains:
            for target in prioritized_targets:
                if str(target.get("domain", "")).strip() != domain:
                    continue
                url = str(target.get("url", "")).strip()
                if (
                    not url
                    or url in preview_urls
                    or url not in remaining_target_url_set
                    or _domain_for_url(url) in getattr(ctx, "unavailable_domains", set())
                    or _url_budget_exhausted(url, ctx)
                    or _normalize_url(url) in getattr(ctx, "exhausted_discovery_urls", set())
                ):
                    continue
                preview_urls.append(url)
                break
            if len(preview_urls) >= limit:
                return preview_urls[:limit]

    for target in prioritized_targets:
        url = str(target.get("url", "")).strip()
        if (
            not url
            or url in preview_urls
            or (remaining_target_url_set and url not in remaining_target_url_set)
            or _domain_for_url(url) in getattr(ctx, "unavailable_domains", set())
            or _url_budget_exhausted(url, ctx)
            or _normalize_url(url) in getattr(ctx, "exhausted_discovery_urls", set())
        ):
            continue
        preview_urls.append(url)
        if len(preview_urls) >= limit:
            break
    return preview_urls


def _switch_candidate_domain(ctx: ToolContext) -> str | None:
    """Return a domain to move away from when it has not produced viable leads yet."""
    remaining_domains = _remaining_candidate_domains(ctx)
    if not remaining_domains:
        return None

    if not ctx.fetch_metadata:
        return None

    last_metadata = next(reversed(ctx.fetch_metadata.values()))
    last_domain = _domain_for_url(str(last_metadata.get("final_url") or last_metadata.get("url") or ""))
    if last_domain not in ctx.allowed_domains:
        return None

    outcome = ctx.domain_outcomes.get(last_domain)
    if outcome and outcome.saved_hits > 0:
        return None

    if ctx.domain_fetch_counts.get(last_domain, 0) < 1:
        return None

    return last_domain


def _follow_through_signature(ctx: ToolContext) -> tuple[int, int, int, int, int, str]:
    """Return a lightweight snapshot of actionable state for reminder budgeting."""
    return (
        len(_unprocessed_fetch_ids(ctx)),
        len(_remaining_discovery_fetch_ids(ctx)),
        len(_remaining_discovered_profile_urls(ctx)),
        len(_remaining_candidate_target_urls(ctx)) + len(_remaining_candidate_domains(ctx)),
        _saved_lead_count(ctx),
        ctx.source_phase,
    )


def _url_budget_exhausted(url: str, ctx: ToolContext) -> bool:
    """Return True when the URL's fetch budget bucket is exhausted."""
    budget_key = _fetch_budget_key(url)
    return ctx.fetch_budget_counts.get(budget_key, 0) >= _fetch_budget_for_url(url, ctx)


def _domain_budget_exhausted(domain: str, ctx: ToolContext) -> bool:
    """Return True when all suggested starter URLs for a domain are out of budget."""
    candidate_urls = [
        str(target.get("url", "")).strip()
        for target in ctx.suggested_targets
        if str(target.get("domain", "")).strip().lower() == domain and str(target.get("url", "")).strip()
    ]
    if not candidate_urls:
        return False
    return all(_url_budget_exhausted(url, ctx) for url in candidate_urls)


def _no_viable_next_actions(ctx: ToolContext) -> bool:
    """Return True when the run has no actionable fetched pages or fetchable next URLs left."""
    return not (
        _unprocessed_fetch_ids(ctx)
        or _remaining_discovery_fetch_ids(ctx)
        or _remaining_discovered_profile_urls(ctx)
        or _remaining_candidate_target_urls(ctx)
        or _remaining_candidate_domains(ctx)
    )
