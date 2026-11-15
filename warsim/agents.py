"""학습 전 확인용 플레이어."""

from .hexgrid import distance
from .units import ARTY, BDE_HQ, BN_HQ, BRADLEY, DISMOUNT, FSC, INF, MAINT, MED, SIGNAL, SUPPLY


class RandomAgent:
    """규칙 안에서 무작위로 두되, 절반은 상대 목표 쪽으로 움직이는 플레이어."""

    def __init__(self, side, rng):
        self.side = side
        self.rng = rng

    def _toward(self, game, pc, target):
        opts = game.move_options(pc)
        best = min(distance(h, target) for h in opts)
        cands = [h for h in opts if distance(h, target) == best]
        return cands[self.rng.integers(len(cands))]  # 동점은 무작위 (진영 쏠림 방지)

    def _random_hex(self, game, pc):
        opts = list(game.move_options(pc))
        return opts[self.rng.integers(len(opts))]

    def act(self, game):
        r = self.rng.random
        obs = game.observe(self.side)
        enemies = obs["enemy"]
        goal = obs["enemy_objective"]
        orders = []
        own = obs["own"]
        for pc in own:
            if pc.kind in (BDE_HQ, SIGNAL, SUPPLY, MAINT, MED):
                continue  # 여단본부·통신중대·지원대대 중대는 제자리
            if pc.kind in (BN_HQ, FSC):
                mates = [o for o in own if o.bn == pc.bn and o.kind not in (BN_HQ, FSC)]
                if mates:  # 대대본부·전방지원중대는 소속 중대 중 하나를 따라감
                    m = mates[self.rng.integers(len(mates))]
                    orders.append(("move", pc.pid, self._toward(game, pc, m.hex)))
                continue
            near = [e for e in enemies if distance(pc.hex, e.hex) <= game.cfg.direct_fire_range]
            if pc.kind == ARTY:
                inrange = [e for e in enemies if distance(pc.hex, e.hex) <= game.cfg.artillery_range]
                if inrange and r() < 0.7:
                    e = inrange[self.rng.integers(len(inrange))]
                    orders.append(("fire", pc.pid, e.hex))
                elif r() < 0.3:
                    orders.append(("move", pc.pid, self._toward(game, pc, goal)))
                continue
            if near and pc.ammo > 0 and r() < 0.7:
                e = near[self.rng.integers(len(near))]
                orders.append(("attack", pc.pid, e.pid))
                continue
            x = r()
            if x < 0.02 and pc.platoons > 1:
                orders.append(("split", pc.pid, 1, self._random_hex(game, pc)))
            elif x < 0.04 and pc.kind == INF:
                orders.append(("dismount", pc.pid))
            elif x < 0.08:
                mates = [o for o in own if o.hex == pc.hex and o.pid != pc.pid]
                for m in mates:
                    if m.kind == pc.kind:
                        orders.append(("merge", pc.pid, m.pid))
                        break
                    if pc.kind == BRADLEY and m.kind == DISMOUNT:
                        orders.append(("mount", pc.pid, m.pid))
                        break
            elif x < 0.55:
                orders.append(("move", pc.pid, self._toward(game, pc, goal)))
            elif x < 0.85:
                orders.append(("move", pc.pid, self._random_hex(game, pc)))
        return orders
