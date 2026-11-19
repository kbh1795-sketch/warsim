"""여러 판을 동시에(락스텝) 돌리는 도구. CPU 에서 신경망 호출을 묶어서 한 번에 한다."""

import numpy as np
import torch

from ..agents import RandomAgent
from ..game import Game
from .deploy import TEMPLATES, placement
from .encode import decode, encode, stack
from .model import to_torch


class NetPolicy:
    def __init__(self, net, greedy=False, bandit=None):
        self.net, self.greedy, self.bandit = net, greedy, bandit

    def template(self, rng):
        return self.bandit.choose(rng) if self.bandit is not None else int(rng.integers(len(TEMPLATES)))

    def orders(self, items):
        """items: [(game, side)] -> [orders]."""
        if not items:
            return []
        encs = [encode(g, s) for g, s in items]
        acts, _, _ = self.net.act(to_torch(stack(encs)), greedy=self.greedy)
        acts = {k: v.numpy() for k, v in acts.items()}
        return [decode(g, s, e, {k: v[i] for k, v in acts.items()})
                for i, ((g, s), e) in enumerate(zip(items, encs))]


class RandomPolicy:
    def __init__(self, rng):
        self.rng = rng

    def template(self, rng):
        return int(rng.integers(len(TEMPLATES)))  # 배치 유형은 무작위로 (보급선이 닿는 배치)

    def orders(self, items):
        return [RandomAgent(s, self.rng).act(g) for g, s in items]


def new_game(cfg, rng, templates):
    g = Game(cfg, rng=rng)
    for side in (0, 1):
        t = templates[side]
        g.deploy(side, None if t is None else placement(g, side, t, rng))
    return g


def match(pa, pb, cfg, n, rng):
    """pa 대 pb 를 n 판 (진영 번갈아). pa 기준 (승, 무, 패)."""
    games, a_side = [], []
    for k in range(n):
        sa = k % 2
        pol = [None, None]
        pol[sa], pol[1 - sa] = pa, pb
        games.append(new_game(cfg, rng, [pol[0].template(rng), pol[1].template(rng)]))
        a_side.append(sa)
    with torch.no_grad():
        while not all(g.done for g in games):
            live = [i for i, g in enumerate(games) if not g.done]
            oa = pa.orders([(games[i], a_side[i]) for i in live])
            ob = pb.orders([(games[i], 1 - a_side[i]) for i in live])
            for j, i in enumerate(live):
                o = [None, None]
                o[a_side[i]], o[1 - a_side[i]] = oa[j], ob[j]
                games[i].step(o[0], o[1])
    w = sum(g.result(s) > 0 for g, s in zip(games, a_side))
    d = sum(g.result(s) == 0 for g, s in zip(games, a_side))   # 시간 종료는 패배로 센다
    return w, d, n - w - d


def score(wdl):
    w, d, l = wdl
    return (w + 0.5 * d) / max(w + d + l, 1)
