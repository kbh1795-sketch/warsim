"""무작위 플레이어끼리 여러 판을 두어 게임이 끝까지 도는지, 얼마나 빠른지 확인한다.

    python scripts/run_random.py --games 20
"""

import argparse
import collections
import time

import numpy as np

from warsim import Config, Game
from warsim.agents import RandomAgent
from warsim.replay import end_reason


def play(i, cfg):
    rng = np.random.default_rng(i)
    g = Game(cfg, rng=rng)
    g.deploy(0)
    g.deploy(1)
    agents = [RandomAgent(0, rng), RandomAgent(1, rng)]
    t0 = time.perf_counter()
    while not g.done:
        g.step(agents[0].act(g), agents[1].act(g))
    dt = time.perf_counter() - t0
    reason = end_reason(g)
    return g.winner, reason, g.turn, dt, len(g.pieces)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    args = ap.parse_args()
    cfg = Config()
    res = [play(i, cfg) for i in range(args.games)]
    winners = collections.Counter(r[0] for r in res)
    reasons = collections.Counter(r[1] for r in res)
    turns = sum(r[2] for r in res)
    secs = sum(r[3] for r in res)
    print(f"games            {args.games}")
    print(f"winner           {dict(winners)}")
    print(f"end reason       {dict(reasons)}")
    print(f"mean turns/game  {turns / args.games:.1f}")
    print(f"mean sec/game    {secs / args.games:.2f}")
    print(f"turns/sec        {turns / secs:.0f}")


if __name__ == "__main__":
    main()
