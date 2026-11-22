"""Compare Keygen's returned assistant messages with the history rendered for the next turn."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate, openai_to_messages
from tools.strata_tokenizer import Tokenizer


def audit(trajectory, directory):
    tok = Tokenizer.from_dir(directory)
    template = ChatTemplate(directory / "chat_template.jinja")

    def render(messages):
        history, tools, kwargs = openai_to_messages({"messages": messages, "reasoning_effort": "xhigh"})
        return template.render(history, tools=tools, **kwargs)

    # Typed delimiters isolate this turn from the previous user text.
    base = [{"role": "user", "content": "Music."}]
    opening = render(base)
    messages = json.loads(trajectory.read_text())["messages"]
    turns = [m for m in messages if m["role"] == "assistant"]
    matched = 0
    for i, (previous, following) in enumerate(zip(turns, turns[1:]), 1):
        usage = previous["extra"]["response"]["usage"]
        next_usage = following["extra"]["response"]["usage"]
        available = usage["prompt_tokens"] + usage["completion_tokens"] - 1
        cached = next_usage["prompt_tokens_details"]["cached_tokens"]
        if cached == available:
            matched += 1
            continue
        message = {k: previous[k] for k in ("role", "content", "tool_calls", "reasoning_content") if k in previous}
        history = render(base + [message])
        assert history.startswith(opening)
        rerendered = tok.encode(history[len(opening):], parse_special=True)
        print(json.dumps(dict(assistant_turn=i, emitted=usage["completion_tokens"], rerendered=len(rerendered),
                              expected=available, next_cached=cached)))
    print(json.dumps(dict(assistant_turns=len(turns), followups=max(0, len(turns)-1), exact_prefix=matched)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, help="private Chat Completions trajectory.json from Keygen Bench")
    parser.add_argument("--tokenizer", required=True, type=Path, help="tokenizer directory exported from the served GGUF")
    args = parser.parse_args()
    audit(args.trajectory, args.tokenizer)
