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
