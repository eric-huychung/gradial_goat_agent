"""THE NAIVE BASELINE — this is your starting point, and it scores zero.

It does the obvious thing: hand the task to the model, take whatever comes
back, submit it. One call. No tools. No loop.

Run it. Watch it fail every tile, confidently, with plausible-looking
answers. That failure is the entire premise of this hackathon: the model
cannot see a 4 MB log, cannot run code, cannot open a web page, and cannot
tell that it is guessing.

    python main.py

By default it samples MAX_TILES=3 tiles, one per board cell, then exits. It
does NOT walk the whole board: `jeopardy.open_tiles()` returns every open
variant (60 on the practice board, a few hundred on the scored one) because
that is the parallelism a real agent gets, and a serial baseline with no tools
would spend an hour and the entire submission rate limit re-proving the same
point. Set TASK_FILTER to aim it: TASK_FILTER=PR-H5,PR-W5 runs exactly those
two tiles. VERBOSE=1 shows the prompt, the reply, and the raw submit
response. See .env.example — those three knobs are the only ones this file
reads.

YOUR JOB is to turn this into an agent. Nothing below is precious — delete
all of it if you like. `jeopardy.py` next to this file has the plumbing
(board, tasks, files, submit, a model client) so you never have to think
about HTTP — and it must be INSIDE the zip you submit, because it is not
part of the hosted image (`cd starter_agent && zip -r ../agent.zip .`).

WHAT AN AGENT NEEDS THAT THIS DOESN'T HAVE
------------------------------------------
1. TOOLS. The model must be able to act, not just answer. At minimum:
     - run Python (parse the 4 MB file, decode the cipher, brute the search)
     - make HTTP requests, KEEPING COOKIES between them (a whole category
       is websites with logins and multi-step flows)
   With the Anthropic SDK that means passing `tools=[...]` with a JSON
   schema per tool, then handling `tool_use` blocks in the response and
   replying with `tool_result` blocks. See:
   https://docs.anthropic.com/en/docs/build-with-claude/tool-use

2. A LOOP. One model call is never enough. Call, execute the tool it asked
   for, feed the result back, repeat until it answers — with a turn cap so a
   stuck tile can't eat your whole budget.

3. VERIFICATION. Wrong answers cost 25% of the tile and trigger a doubling
   lockout. Make the agent prove its answer before submitting — and never
   let the model RETYPE an exact-match token; echo it straight from your
   code, or it will eventually fumble one character.

4. JUDGEMENT. Which tile next? Rows unlock on a timer, tiles are
   first-claim-wins, and a tile someone else takes mid-solve is wasted work.
   When do you give up and move on?

5. CONCURRENCY. Submissions are rate-limited; thinking is not. A serial
   agent watches other teams take the board while it works one tile.

6. SPECIALIZATION. A general loop is beaten by one that notices "this is a
   crypto tile" and brings the right tools and prompt to it.

Check your budget any time with `jeopardy.me()`.
"""
from __future__ import annotations

import os
import time

import jeopardy as jp

# ---- dev knobs, read by THIS file (see .env.example) ----------------------
# Three, and all three work. If you rewrite main.py they are yours to keep or
# drop — but don't leave a knob documented that the code ignores.

# Print the prompt, the model's full reply, and the raw submit response.
VERBOSE = os.environ.get("VERBOSE") == "1"
# Attempt exactly these tiles, in this order, ignoring MAX_TILES.
# e.g. TASK_FILTER=PR-H5,PR-W5 for the Heavy Compute and Dark Web 500s.
TASK_FILTER = [t.strip() for t in os.environ.get("TASK_FILTER", "").split(",")
               if t.strip()]
# How many tiles one run attempts when TASK_FILTER is empty.
MAX_TILES = int(os.environ.get("MAX_TILES", "3"))


def naive_attempt(task_id: str) -> bool:
    """One model call. No tools. The floor you are trying to beat."""
    detail = jp.task(task_id)
    workdir = jp.workdir(task_id)          # stable; see jeopardy.workdir
    names = jp.fetch_files(task_id, detail)

    prompt = (
        f"{detail['prompt']}\n\n"
        f"Files downloaded to {workdir}: {names or 'none'}\n"
        f"Answer checking: {detail.get('answer_format', 'exact')}\n\n"
        "Reply with ONLY the final answer — no working, no explanation."
    )
    if VERBOSE:
        jp.log(f"{task_id} ({detail.get('category')}, {detail.get('points')}pt)"
               f" prompt:\n{prompt}\n---")

    client = jp.anthropic_client()
    resp = client.messages.create(
        model=jp.MODEL, max_tokens=1000,
        messages=[{"role": "user", "content": prompt}])
    answer = "".join(b.text for b in resp.content if b.type == "text").strip()
    if VERBOSE:
        jp.log(f"{task_id} model replied:\n{answer}\n---")

    result = jp.submit(task_id, answer)
    jp.log(f"{task_id}: answered {answer[:60]!r} -> {result.get('result')}")
    if VERBOSE:
        jp.log(f"{task_id} full submit response: {result}")
    if result.get("result") == "forbidden":
        jp.log("  (a scored round is live — only your HOSTED agent may "
               "submit. Deploy with /api/agent/submit, or practise on the "
               "practice board.)")
    return result.get("result") == "correct"


def pick_tiles(b: dict) -> list[str]:
    """Which tiles this one pass attempts. Replace me — this is the strategy.

    `jp.open_tiles()` hands back EVERY open variant, because every one of them
    is claimable right now and a real agent works them in parallel. A baseline
    with no tools and no loop must not walk that list: at one submission per
    three seconds, a few hundred tiles is the whole event spent proving the
    same thing. So: one tile per cell — a spread of different tasks rather
    than eight clones of one — capped at MAX_TILES.

    Using the full width instead of throwing it away is most of the hackathon.
    """
    tiles = jp.open_tiles(b)
    if TASK_FILTER:
        open_now = {t["id"] for t in tiles}
        missing = [t for t in TASK_FILTER if t not in open_now]
        if missing:
            jp.log(f"TASK_FILTER: not open right now, skipping {missing}")
        return [t for t in TASK_FILTER if t in open_now]
    picked: list[str] = []
    seen_cells: set[tuple] = set()
    for t in tiles:
        cell = (t.get("category"), t.get("points"))
        if cell in seen_cells:
            continue                # already sampling this cell's task
        seen_cells.add(cell)
        picked.append(t["id"])
        if len(picked) >= MAX_TILES:
            break
    return picked


def main() -> None:
    b = jp.board()
    every = jp.open_tiles(b)
    cells = {(t.get("category"), t.get("points")) for t in every}
    jp.log(f"phase={b.get('phase')}: {len(every)} open tiles across "
           f"{len(cells)} cells — all of them claimable in parallel")
    tiles = pick_tiles(b)
    if not tiles:
        jp.log("nothing open — is the board live yet?")
        return
    jp.log(f"naive baseline is attempting {len(tiles)} of them, serially: "
           f"{tiles}")

    solved = 0
    for task_id in tiles:
        try:
            solved += naive_attempt(task_id)
        except jp.AuthError:
            raise                                  # fatal; fix the key
        except jp.TileUnavailable as e:
            jp.log(f"{task_id}: not available — {e}")
        except Exception as e:                     # noqa: BLE001
            jp.log(f"{task_id}: blew up — {e!r}")
        time.sleep(4)                              # submission rate limit

    jp.log(f"naive baseline: {solved}/{len(tiles)} attempted tiles solved")
    if solved == 0:
        jp.log("Exactly as expected. Now go build an agent — read the "
               "docstring at the top of this file.")


if __name__ == "__main__":
    try:
        main()
    except jp.AuthError as e:
        raise SystemExit(f"[auth] {e}")
