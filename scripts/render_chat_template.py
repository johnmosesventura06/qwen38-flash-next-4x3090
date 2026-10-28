#!/usr/bin/env python3
"""Render a serving chat template for the :8000 lane with a chosen reasoning-effort tier.

Why this exists (2026-09-28): the checkpoint's own chat_template.jinja decides how
hard the model thinks. Its line

    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}

means "if the request did not name an effort, think at xhigh". vLLM has no
server-side effort default of its own -- it forwards whatever the request sent
into this template, and --default-chat-template-kwargs loses to any request-level
value. So the only way to move the lane's thinking tier is to serve a template with
that line re-pointed. This script writes that copy; it never touches the
checkpoint (which is mounted read-only in the container anyway).

  --src    checkpoint chat_template.jinja (read, never modified)
  --out    where the serving copy goes
  --effort xhigh | medium | low
  --lock   ignore what the client sends and hard-set the tier (client value dropped)

Exit 2 = the anchor line is not exactly one occurrence, i.e. the checkpoint is not
the template we think it is. Better to refuse the boot than to serve a silent
prompt-shape change.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

EFFORTS = ("xhigh", "medium", "low")

# The sentence the checkpoint injects per tier. Used to prove the render.
PROSE = {
    "xhigh": "Reasoning effort is set to xhigh",
    "low": "Reasoning effort is set to low",
    "medium": None,  # upstream ships no sentence for medium: it is the silent tier
}

# Indent | "{%- set resolved_reasoning_effort = " | value | "%}"
ANCHOR = re.compile(
    r"^([ \t]*)(\{%-?\s*set\s+resolved_reasoning_effort\s*=\s*)"
    r"(reasoning_effort\s*\|\s*default\(\s*['\"]xhigh['\"]\s*\))"
    r"(\s*%\})[ \t]*$",
    re.MULTILINE,
)


def fail(msg: str) -> None:
    print(f"[render_chat_template] REFUSE: {msg}", file=sys.stderr)
    sys.exit(2)


def render_check(text: str, effort: str, lock: bool) -> None:
    """Best-effort proof that the patched template renders as intended.

    The checkpoint template uses two HF-only helpers (tojson, raise_exception); we
    stub them so a plain jinja2 render is representative of what vLLM will see. A
    render failure here is NOT fatal -- the verified string patch is the real
    guarantee -- so this stays advisory.
    """
    from jinja2 import Environment

    env = Environment()
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False)
    env.globals["raise_exception"] = lambda *a: (_ for _ in ()).throw(
        RuntimeError(" ".join(str(x) for x in a))
    )
    tmpl = env.from_string(text)

    def render(**kw):
        return tmpl.render(messages=[{"role": "user", "content": "hi"}], **kw)

    if PROSE.get(effort):
        assert PROSE[effort] in render(), f"{effort} tier sentence missing"
    if effort != "xhigh":
        assert PROSE["xhigh"] not in render(), "xhigh sentence still present"
    if lock and effort != "xhigh":
        # A client still asking for xhigh must no longer get the xhigh sentence.
        assert PROSE["xhigh"] not in render(reasoning_effort="xhigh"), (
            "lock failed: client-sent xhigh still wins"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--effort", required=True, choices=EFFORTS)
    ap.add_argument("--lock", action="store_true")
    args = ap.parse_args()

    try:
        text = open(args.src, encoding="utf-8").read()
    except OSError as e:
        fail(f"cannot read {args.src}: {e}")

    hits = list(ANCHOR.finditer(text))
    if len(hits) != 1:
        fail(f"expected exactly 1 effort-default line in {args.src}, found {len(hits)}")

    m = hits[0]
    # --lock drops the client's value outright; plain mode only re-points the
    # default, so a request that names an effort still wins.
    new_value = (
        f"'{args.effort}'" if args.lock else f"reasoning_effort|default('{args.effort}')"
    )
    patched = text[: m.start()] + m.group(1) + m.group(2) + new_value + m.group(4) + text[m.end():]

    if patched == text:
        fail("patch was a no-op (source already at this tier?)")
    if ANCHOR.search(patched):
        fail("anchor survived the patch")

    try:
        render_check(patched, args.effort, args.lock)
        verdict = "render-verified"
    except Exception as e:  # advisory only
        verdict = f"RENDER-CHECK-FAILED ({type(e).__name__}: {e})"

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(patched)

    print(
        f"[render_chat_template] effort={args.effort} lock={int(args.lock)} {verdict}\n"
        f"  before: {m.group(0).strip()}\n"
        f"  after : {(m.group(1) + m.group(2) + new_value + m.group(4)).strip()}\n"
        f"  wrote : {args.out}"
    )


if __name__ == "__main__":
    main()
