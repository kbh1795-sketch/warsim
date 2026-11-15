"""초기 배치 유형과 밴딧 (플루토의 오프닝 선택 방식).

플루토는 빌드를 신경망이 아니라 상대별 승패 기록을 보는 밴딧으로 골랐다.
여기서는 배치 유형 몇 개를 두고, 승패 기록으로 톰슨 샘플링을 해서 고른다.
(무승부는 반 승으로 센다.)
"""

import json
from pathlib import Path

import numpy as np

from ..game import SUPPORT_BN
from ..hexgrid import distance
from ..units import FSC, HQ_KINDS, SUPPLY

TEMPLATES = ["전방 집중", "좌익 집중", "우익 집중", "종심 배치", "균등 분산"]


def placement(game, side, template, rng):
    """배치 유형 -> {pid: hex}. 신경망 관점(내 후방 = 0행)에서 정하고 게임 좌표로 바꾼다.

    대대마다 중심점을 정하고, 대대본부는 중심점에, 소속 중대는 중심점 2칸 안에 둔다
    (처음부터 통신이 끊기지 않도록). 여단본부·통신중대·포병은 후방 중앙, 지원대대는
    기동대대들의 가운데 후방, 전방지원중대는 자기 대대와 지원대대 사이.
    """
    c = game.cfg
    W, D = c.width, c.deploy_depth
    name = TEMPLATES[template]

    def to_game(col, row):
        col = min(max(col, 0), W - 1)
        row = min(max(row, 0), D - 1)
        return (col, row) if side == 0 else (W - 1 - col, c.length - 1 - row)

    def anchor(maneuver):
        if not maneuver:
            return int(rng.integers(W // 3, 2 * W // 3)), int(rng.integers(0, 3))
        if name == "전방 집중":
            return int(rng.integers(W // 4, 3 * W // 4)), int(rng.integers(D - 3, D))
        if name == "좌익 집중":
            return int(rng.integers(0, W // 3)), int(rng.integers(D // 2, D))
        if name == "우익 집중":
            return int(rng.integers(2 * W // 3, W)), int(rng.integers(D // 2, D))
        if name == "종심 배치":
            return int(rng.integers(W // 4, 3 * W // 4)), int(rng.integers(0, D))
        return int(rng.integers(0, W)), int(rng.integers(0, D))

    pieces = game.side_pieces(side)
    bns = sorted({p.bn for p in pieces})
    centers = {bn: anchor(any(p.spec.maneuver for p in pieces if p.bn == bn)) for bn in bns}

    # 지원대대(배급·정비·의무)는 기동대대 중심들의 가운데 후방에 둔다
    man = [centers[bn] for bn in bns if any(p.spec.maneuver for p in pieces if p.bn == bn)]
    sup = (int(round(np.mean([m[0] for m in man]))) if man else W // 2, int(rng.integers(0, 3)))
    centers[SUPPORT_BN] = sup

    # 보급선이 처음부터 닿도록 (거리는 게임 좌표로 잰다: 육각 격자는 뒤집으면 거리가 달라짐), 
    # 각 대대 중심을 지원대대 중심 가까이 당긴다
    # (전방지원중대를 가운데에 두면 배급중대와 부대 양쪽 모두 보급 거리 안)
    reach = game.cfg.supply_range
    for bn in bns:
        if bn == SUPPORT_BN:
            continue
        lim = reach - 2 if bn < 0 else 2 * (reach - 3)  # 부대는 중심에서 최대 3칸 흩어짐
        col, row = centers[bn]
        t = 1.0
        while distance(to_game(int(round(sup[0] + (col - sup[0]) * t)),
                               int(round(sup[1] + (row - sup[1]) * t))), to_game(*sup)) > lim:
            t -= 0.05
        centers[bn] = (int(round(sup[0] + (col - sup[0]) * t)),
                       int(round(sup[1] + (row - sup[1]) * t)))

    out = {}
    for p in pieces:
        col, row = centers[p.bn]
        if p.kind == FSC:  # 전방지원중대: 자기 대대와 지원대대의 가운데
            col, row = (col + sup[0]) // 2, (row + sup[1]) // 2
        elif p.kind not in HQ_KINDS and p.kind != SUPPLY:  # 배급중대는 지원대대 중심에
            col += int(rng.integers(-2, 3))
            row += int(rng.integers(-2, 3))
        out[p.pid] = to_game(col, row)
    # 흩어 놓다가 자기 전방지원중대 보급 거리를 벗어난 말은 대대 중심으로 되돌린다
    fsc = {p.bn: out[p.pid] for p in pieces if p.kind == FSC}
    for p in pieces:
        if p.bn in fsc and p.kind != FSC and distance(out[p.pid], fsc[p.bn]) > reach:
            out[p.pid] = to_game(*centers[p.bn])
    return out


class DeploymentBandit:
    def __init__(self, n=len(TEMPLATES)):
        self.wins = np.zeros(n)
        self.games = np.zeros(n)

    def choose(self, rng):
        a = self.wins + 1
        b = self.games - self.wins + 1
        return int(np.argmax(rng.beta(a, b)))

    def update(self, template, result):
        """result: 1 승, 0.5 무, 0 패."""
        self.games[template] += 1
        self.wins[template] += result

    def table(self):
        return [{"배치": TEMPLATES[i], "판수": int(self.games[i]),
                 "승률": round(float(self.wins[i] / self.games[i]), 3) if self.games[i] else None}
                for i in range(len(TEMPLATES))]

    def save(self, path):
        Path(path).write_text(json.dumps({"wins": self.wins.tolist(), "games": self.games.tolist()}))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        b = cls(len(d["wins"]))
        b.wins, b.games = np.array(d["wins"]), np.array(d["games"])
        return b
