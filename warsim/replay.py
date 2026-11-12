"""게임 한 판을 기록해서, 브라우저로 여는 재생 파일(HTML 한 개)로 저장한다."""

import json
from pathlib import Path

_TEMPLATE = Path(__file__).with_name("viewer.html")


class Recorder:
    def __init__(self, game):
        self.game = game
        self.frames = [game.snapshot()]

    def record(self):
        self.frames.append(self.game.snapshot())

    def save(self, path, reason=""):
        g, c = self.game, self.game.cfg
        data = {
            "width": c.width, "length": c.length, "turn_hours": c.turn_hours,
            "max_turns": c.max_turns, "hold_turns": c.objective_hold_turns,
            "deploy_depth": c.deploy_depth,
            "terrain": g.map.terrain.tolist(),
            "base": [list(b) for b in g.base], "objective": [list(o) for o in g.objective],
            "reason": reason, "frames": self.frames,
        }
        html = _TEMPLATE.read_text(encoding="utf-8").replace(
            "/*__DATA__*/null", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
        Path(path).write_text(html, encoding="utf-8")
        return path


def end_reason(game):
    why = [r for r in game.defeat_reason if r]
    if why:
        return why[0]
    if max(game.hold) >= game.cfg.objective_hold_turns:
        return "목표 점유"
    return "시간 종료"
