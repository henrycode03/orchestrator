"""Provider-free regression checks for the deployed OpenClaw C1 serializer seam."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OPENCLAW_PI_AI_ROOT = Path(
    "/usr/lib/node_modules/openclaw/node_modules/@mariozechner/pi-ai/dist"
)
COMPLETIONS = OPENCLAW_PI_AI_ROOT / "providers/openai-completions.js"
RESPONSES = OPENCLAW_PI_AI_ROOT / "providers/openai-responses.js"

NODE_PROBE = r"""
const kind = process.argv[1];
const empty = process.argv[2] === "empty";
const modulePath = kind === "completions"
  ? "COMPLETIONS_PATH"
  : "RESPONSES_PATH";
const imported = await import(modulePath);
const streamFactory = kind === "completions"
  ? imported.streamOpenAICompletions
  : imported.streamOpenAIResponses;
const model = {
  api: kind === "completions" ? "openai-completions" : "openai-responses",
  provider: "openai",
  id: "qwen-local",
  name: "qwen-local",
  baseUrl: "http://ai-gateway:8000/v1",
  reasoning: false,
  input: ["text"],
  contextWindow: 131072,
  maxTokens: 8,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};
const tool = {
  name: "lookup_rate",
  description: "Look up rate",
  parameters: {
    type: "object",
    properties: { q: { type: "string" } },
    required: ["q"],
    additionalProperties: false,
  },
};
let captured;
streamFactory(
  model,
  {
    messages: [{ role: "user", content: "provider-free serializer probe" }],
    tools: empty ? [] : [tool],
  },
  {
    apiKey: "runtime3-placeholder",
    ...(empty ? {} : { toolChoice: "none" }),
    onPayload(params) {
      captured = structuredClone(params);
      throw new Error("STOP_BEFORE_PROVIDER");
    },
  },
);
await new Promise((resolve) => setTimeout(resolve, 100));
process.stdout.write(JSON.stringify({
  tools: captured?.tools,
  tool_choice: captured?.tool_choice,
  keys: Object.keys(captured || {}),
}));
"""


def _capture(kind: str, empty: bool) -> dict:
    script = NODE_PROBE.replace("COMPLETIONS_PATH", str(COMPLETIONS)).replace(
        "RESPONSES_PATH", str(RESPONSES)
    )
    result = subprocess.run(
        [
            "node",
            "--input-type=module",
            "-e",
            script,
            kind,
            "empty" if empty else "tool",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    failures: list[str] = []
    for kind in ("completions", "responses"):
        empty = _capture(kind, empty=True)
        non_empty = _capture(kind, empty=False)
        if "tools" in empty:
            failures.append(f"{kind}: empty tools field was emitted")
        if "tool_choice" in empty:
            failures.append(f"{kind}: empty request introduced tool_choice")
        if not non_empty.get("tools"):
            failures.append(f"{kind}: legitimate tool was not serialized")
        tool_payload = non_empty.get("tools", [{}])[0]
        tool_name = tool_payload.get("function", {}).get("name") or tool_payload.get(
            "name"
        )
        if tool_name != "lookup_rate":
            failures.append(f"{kind}: tool name changed")
        expected_tool_choice = "none" if kind == "completions" else None
        if non_empty.get("tool_choice") != expected_tool_choice:
            failures.append(f"{kind}: non-empty tool_choice changed")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: C1 empty-tools omission and non-empty tool serialization")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
