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

WHERE THINGS LIVE
-----------------
    config.py         the three dev knobs, read from the environment
    tile_selector.py  which tiles a pass attempts — the strategy
    solver.py         one tile: prompt, model call, submit
    runner.py         one pass over the board
"""
from __future__ import annotations

import jeopardy as jp

from config import AgentConfig
from runner import AgentRunner


def main() -> None:
    AgentRunner(AgentConfig.from_env()).run()


if __name__ == "__main__":
    try:
        main()
    except jp.AuthError as e:
        raise SystemExit(f"[auth] {e}")
