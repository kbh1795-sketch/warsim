"""초기 배치 유형과 밴딧 (플루토의 오프닝 선택 방식).

플루토는 빌드를 신경망이 아니라 상대별 승패 기록을 보는 밴딧으로 골랐다.
여기서는 배치 유형 몇 개를 두고, 승패 기록으로 톰슨 샘플링을 해서 고른다.
(무승부는 반 승으로 센다.)
"""

import json
from pathlib import Path

import numpy as np

from ..units import HQ_KINDS

TEMPLATES = ["전방 집중", "좌익 집중", "우익 집중", "종심 배치", "균등 분산"]


def placement(game, side, template, rng):
    """배치 유형 -> {pid: hex}. 신경망 관점(내 후방 = 0행)에서 정하고 게임 좌표로 바꾼다.

    대대마다 중심점을 정하고, 대대본부는 중심점에, 소속 중대는 중심점 2칸 안에 둔다
    (처음부터 통신이 끊기지 않도록). 여단본부·통신중대·포병·지원대대는 후방 중앙.
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
    out = {}
    for p in pieces:
        col, row = centers[p.bn]
        if p.kind not in HQ_KINDS:
            col += int(rng.integers(-2, 3))
            row += int(rng.integers(-2, 3))
        out[p.pid] = to_game(col, row)
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
