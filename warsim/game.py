"""게임 엔진.

한 턴 처리 순서 (양측 동시):
    1. 구조 변경: 합류, 탑승, 분리, 하차
    2. 이동
    3. 시야 계산
    4. 사격: 직사 공격, 자동 응사, 포병  -> 피해는 한꺼번에 계산 (동시)
    5. 피해 적용
    6. 보급 (자동): 기지 -> 배급중대 -> 전방지원중대 -> 대대 부대, 거리 안이면 저절로
    7. 소모: 식량, 연료 / 굶주림
    8. 정지 턴 수 갱신, 수리 진행
    9. 제거: 격멸(30% 미만), 굶주림
   10. 목표 점유, 승리 판정
   11. 통신 연결 다시 계산 (다음 턴 명령·정보에 적용)

보급: 명령이 필요 없다. 기지(여단 바깥 상급 지원부대)가 배급중대를 늘 채우고,
      배급중대는 8칸 안의 전방지원중대와 여단 직할·지원대대 말을, 전방지원중대는 8칸 안의
      자기 대대 말을 채운다. 신경망은 이 말들을 어디에 둘지만 정한다.
피해: 잃은 전투력의 50% 완파(영구), 30% 소파(24시간), 20% 중파(72시간).
      소파는 자기 전방지원중대(없으면 정비중대) 8칸 안, 중파는 정비중대와 이어져야 수리된다.

통신: 말은 자기 대대본부와 8칸 안, 대대본부는 여단본부나 여단 통신중대와 35칸 안이고
      사이에 산지가 가로막지 않아야 연결된다. 끊긴 말은 새 명령을 못 받고 마지막 명령
      (이동·공격·사격)을 이어가며, 그 말이 본 적은 지휘관이 모른다.

승리: 상대 여단본부 제거, 상대 전체 전투력 30% 이하, 상대 기동부대 전멸,
      상대 후방 목표 점유 유지 중 하나.

명령 형식 (진영별 리스트):
    ("move", pid, hex)
    ("attack", pid, target_pid)
    ("fire", pid, hex)                 포병
    ("split", pid, k[, hex])           k개 소대를 떼어냄. hex 를 주면 떼어낸 말이 그 칸으로 이동
    ("merge", pid, other_pid)          같은 칸, 같은 종류
    ("dismount", pid)                  보병 -> 브래들리 + 하차 보병
    ("mount", bradley_pid, dismount_pid)
"""

import numpy as np

from .config import Config
from .hexgrid import HexMap, distance
from .units import (ARTY, BDE_HQ, BN_HQ, BRADLEY, DISMOUNT, DISMOUNT_SHARE, INF,
                    FSC, KIND_SPECS, MAINT, MED, NO_SPLIT, RECON, SHARED, SIGNAL,
                    SUPPLY, SUPPORT_KINDS, TANK, Piece)

# 미군 ABCT (2016 이후) 대대별 중대 구성 [근거] battleorder.org, MCoE SM 3-90
BATTALIONS = [
    ("보병형 제병협동", [INF, INF, TANK]),
    ("전차형 제병협동 1", [TANK, TANK, INF]),
    ("전차형 제병협동 2", [TANK, TANK, INF]),
    ("기병", [RECON, RECON, RECON]),
    ("포병", [ARTY, ARTY, ARTY]),
    ("지원", [SUPPLY, MAINT, MED]),   # 여단지원대대: 배급·정비·의무중대 (본부중대 = 대대본부)
]
SUPPORT_BN = 5
COMPANY_PLATOONS = (TANK, INF, RECON, SUPPLY)  # 소대 3개로 시작하는 중대
CONTINUING = ("move", "attack", "fire")  # 통신이 끊겨도 이어가는 명령


class Game:
    def __init__(self, cfg=None, rng=None, log=False):
        self.cfg = cfg or Config()
        self.rng = rng if rng is not None else np.random.default_rng()
        self.log_enabled = log
        self.reset()

    # ------------------------------------------------------------ 시작
    def reset(self):
        c = self.cfg
        self.map = HexMap.generate(c.width, c.length, c.mountain_fraction, self.rng)
        self.turn = 0
        self.pieces = {}
        self._next_id = 0
        self.done = False
        self.winner = None
        self.defeat_reason = [None, None]
        self.hold = [0, 0]  # hold[s]: 진영 s 가 상대 목표를 점유한 연속 턴 수
        self.history = []
        self.events = []  # 이번 턴에 일어난 일 (가시화용)
        self.base = [(c.width // 2, 0), (c.width - 1 - c.width // 2, c.length - 1)]
        self.objective = [(c.width // 4, c.objective_depth),
                          (c.width - 1 - c.width // 4, c.length - 1 - c.objective_depth)]
        p = c.platoons_per_company
        for side in (0, 1):
            self._new_piece(side, BDE_HQ, None, 1)
            self._new_piece(side, SIGNAL, None, 1)
            for bn, (_, kinds) in enumerate(BATTALIONS):
                self._new_piece(side, BN_HQ, None, 1, bn)
                for kind in kinds:
                    self._new_piece(side, kind, None, p if kind in COMPANY_PLATOONS else 1, bn)
                if bn != SUPPORT_BN:
                    self._new_piece(side, FSC, None, 1, bn)  # 전방지원중대
            self._set_cargo(side)
        self.start_total = [sum(q.power for q in self.side_pieces(s)) for s in (0, 1)]
        self.last_order = {}
        self.conn = [set(), set()]
        self.deployed = [False, False]
        return self

    def _new_piece(self, side, kind, hexpos, platoons, bn=-1):
        s = KIND_SPECS[kind]
        pc = Piece(self._next_id, side, kind, hexpos, platoons,
                   s.power * platoons, s.power * platoons, bn=bn)
        pc.food = pc.food_cap(self.cfg)
        pc.ammo = pc.ammo_cap() if s.can_attack or kind == ARTY else 0.0
        pc.fuel = pc.fuel_cap()
        self.pieces[pc.pid] = pc
        self._next_id += 1
        return pc

    def _need(self, pc):
        """한 번 가득 채우는 데 드는 물자 (1일분)."""
        return (pc.food_cap(self.cfg), pc.ammo_cap() if pc.armed() else 0.0, pc.fuel_cap())

    def _set_cargo(self, side):
        """전방지원중대는 자기 대대 1일분, 배급중대는 여단 전체 1일분을 싣는다."""
        own = self.side_pieces(side)
        for f in own:
            if f.kind == FSC:
                need = [self._need(q) for q in own if q.bn == f.bn]
            elif f.kind == SUPPLY:
                need = [self._need(q) for q in own]
            else:
                continue
            f.cap_food, f.cap_ammo, f.cap_fuel = (sum(x[i] for x in need) for i in range(3))
            f.fill_cargo()

    def deploy_zone(self, side):
        c = self.cfg
        rows = range(0, c.deploy_depth) if side == 0 else range(c.length - c.deploy_depth, c.length)
        return [(col, r) for r in rows for col in range(c.width)]

    def deploy(self, side, placement=None):
        """placement: {pid: hex}. 빠진 말은 배치 구역 안 무작위."""
        zone = self.deploy_zone(side)
        zset = set(zone)
        placement = placement or {}
        for pc in self.side_pieces(side):
            h = placement.get(pc.pid)
            pc.hex = h if h in zset else zone[self.rng.integers(len(zone))]
        self.deployed[side] = True
        if all(self.deployed):
            self._update_comm()

    # ------------------------------------------------------------ 통신
    def _linked(self, a, b, rng):
        return distance(a.hex, b.hex) <= rng and (
            not self.cfg.comm_los or self.map.line_of_sight(a.hex, b.hex))

    def comm(self, side):
        """통신이 연결된 말의 pid 집합."""
        c = self.cfg
        own = self.side_pieces(side)
        bde = [p for p in own if p.kind == BDE_HQ]
        if not bde:
            return set()
        net = list(bde)
        net += [p for p in own if p.kind == SIGNAL and self._linked(p, bde[0], c.comm_bn_range)]
        ok = {p.pid for p in net}
        hq = {}
        for p in own:
            if p.kind == BN_HQ and any(self._linked(p, n, c.comm_bn_range) for n in net):
                ok.add(p.pid)
                hq[p.bn] = p
        for p in own:
            if p.kind in (BDE_HQ, BN_HQ, SIGNAL):
                continue
            h = hq.get(p.bn)
            if h is not None and self._linked(p, h, c.comm_unit_range):
                ok.add(p.pid)
        return ok

    def _update_comm(self):
        self.conn = [self.comm(0), self.comm(1)]

    def _comm_filter(self, side, orders):
        """연결된 말의 명령만 받고, 끊긴 말은 마지막 명령을 이어간다."""
        conn = self.conn[side]
        out, ordered = [], set()
        for o in orders:
            pid = o[1]
            if pid not in conn:
                continue
            out.append(o)
            if o[0] in CONTINUING:
                self.last_order[pid] = o
                ordered.add(pid)
        for pc in self.side_pieces(side):
            if pc.pid in conn:
                if pc.pid not in ordered:
                    self.last_order.pop(pc.pid, None)  # 명령 없음 = 제자리
                continue
            lo = self.last_order.get(pc.pid)
            if lo is None:
                continue
            if lo[0] == "move":
                dest = tuple(lo[2])
                if dest == pc.hex:
                    self.last_order.pop(pc.pid, None)
                    continue
                opts = self.move_options(pc)
                h = min(opts, key=lambda x: (distance(x, dest), opts[x]))
                if h != pc.hex:
                    out.append(("move", pc.pid, h))
            else:
                out.append(lo)
        return out

    # ------------------------------------------------------------ 조회
    def side_pieces(self, side):
        return [p for p in self.pieces.values() if p.side == side]

    def vision(self, pc):
        c = self.cfg
        if pc.kind == RECON:
            return c.vision_recon
        if pc.kind == ARTY or pc.kind in SUPPORT_KINDS:
            return c.vision_rear
        return c.vision_default

    def known_enemies(self, side):
        """지휘관이 아는 적: 통신이 연결된 말이 본 적만."""
        conn = self.conn[side]
        own = [p for p in self.side_pieces(side) if p.pid in conn]
        seen = set()
        for e in self.side_pieces(1 - side):
            for o in own:
                d = distance(o.hex, e.hex)
                if d <= self.vision(o) and (e.kind != DISMOUNT or d <= self.cfg.dismount_detect):
                    seen.add(e.pid)
                    break
        return seen

    def visible_enemies(self, side):
        own = self.side_pieces(side)
        seen = set()
        for e in self.side_pieces(1 - side):
            for o in own:
                d = distance(o.hex, e.hex)
                if d <= self.vision(o) and (e.kind != DISMOUNT or d <= self.cfg.dismount_detect):
                    seen.add(e.pid)
                    break
        return seen

    def observe(self, side):
        """진영 side 지휘관이 아는 정보: 자기 말 전체(연결 여부 포함) + 연결된 말이 본 적."""
        seen = self.known_enemies(side)
        return {
            "turn": self.turn,
            "own": [p for p in self.side_pieces(side)],
            "connected": set(self.conn[side]),
            "enemy": [self.pieces[i] for i in seen],
            "terrain": self.map.terrain,
            "base": self.base[side],
            "own_objective": self.objective[side],
            "enemy_objective": self.objective[1 - side],
        }

    def move_options(self, pc):
        """이번 턴에 갈 수 있는 칸 {hex: 비용}."""
        c = self.cfg
        if pc.spec.foot:
            return self.map.reachable(pc.hex, c.foot_mp, c.foot_mountain_cost)
        if pc.fuel <= 0:
            return {pc.hex: 0.0}
        return self.map.reachable(pc.hex, c.vehicle_mp, c.vehicle_mountain_cost)

    def snapshot(self):
        """가시화용 현재 상태 (JSON 으로 저장 가능한 형태)."""
        seen = [sorted(self.known_enemies(0)), sorted(self.known_enemies(1))]
        pcs = []
        for p in self.pieces.values():
            pcs.append([p.pid, p.side, p.kind, p.hex[0], p.hex[1], p.platoons,
                        round(p.power, 1), round(p.start_power, 1),
                        round(p.food / max(p.food_cap(self.cfg), 1e-9), 3),
                        round(p.ammo / max(p.ammo_cap(), 1e-9), 3) if p.ammo_cap() else 0,
                        round(p.fuel / p.fuel_cap(), 3) if p.fuel_cap() else -1,
                        int(p.moved), p.stationary, p.starving,
                        int(p.pid in self.conn[p.side]), p.bn])
        ev = [[e[0], e[1], list(e[2]), list(e[3]) if isinstance(e[3], tuple) else e[3]]
              for e in self.events]
        return {"turn": self.turn, "pieces": pcs, "seen": seen, "events": ev,
                "hold": list(self.hold), "done": self.done, "winner": self.winner}

    # ------------------------------------------------------------ 턴
    def step(self, orders0, orders1):
        assert all(self.deployed), "deploy() 를 먼저 호출"
        if self.done:
            return self.done, self.winner
        orders = [list(orders0 or []), list(orders1 or [])]
        self.events = []
        for pc in self.pieces.values():
            pc.moved = False
        for side in (0, 1):
            orders[side] = self._comm_filter(side, orders[side])
        for side in (0, 1):
            orders[side] = self._structural(side, orders[side])
        for side in (0, 1):
            self._moves(side, orders[side])
        seen = [self.visible_enemies(0), self.visible_enemies(1)]
        self._combat(orders, seen)
        for side in (0, 1):
            self._resupply(side)
        self._consume()
        for side in (0, 1):
            self._repair(side)
        self._remove()
        self._check_victory()
        self._update_comm()
        self.turn += 1
        if not self.done and self.turn >= self.cfg.max_turns:
            self.done = True
        return self.done, self.winner

    def _log(self, side, order):
        if self.log_enabled:
            self.history.append((self.turn, side, order))

    def _own(self, side, pid):
        pc = self.pieces.get(pid)
        return pc if pc is not None and pc.side == side else None

    # 1. 구조 변경
    def _structural(self, side, orders):
        rest = []
        for o in orders:
            kind = o[0]
            if kind == "merge":
                a, b = self._own(side, o[1]), self._own(side, o[2])
                if (a and b and a is not b and a.kind == b.kind and a.hex == b.hex
                        and a.kind not in NO_SPLIT and a.bn == b.bn):
                    self._absorb(a, b)
                    self._log(side, o)
            elif kind == "mount":
                a, b = self._own(side, o[1]), self._own(side, o[2])
                if (a and b and a.kind == BRADLEY and b.kind == DISMOUNT and a.hex == b.hex
                        and a.platoons == b.platoons and a.bn == b.bn):
                    self._absorb(a, b)
                    a.kind = INF
                    self._log(side, o)
            elif kind == "split":
                a = self._own(side, o[1])
                k = int(o[2]) if len(o) > 2 else 0
                if a and a.kind not in NO_SPLIT and 1 <= k < a.platoons:
                    new = self._new_piece(side, a.kind, a.hex, k, a.bn)
                    share = a.scale(k / a.platoons)
                    for key, v in share.items():
                        setattr(new, key, v)
                    a.platoons -= k
                    new.stationary = a.stationary
                    new.starving = a.starving
                    self._log(side, o)
                    if len(o) > 3:
                        rest.append(("move", new.pid, o[3]))
                        self.last_order[new.pid] = ("move", new.pid, o[3])
            elif kind == "dismount":
                a = self._own(side, o[1])
                if a and a.kind == INF:
                    new = self._new_piece(side, DISMOUNT, a.hex, a.platoons, a.bn)
                    share = a.scale(DISMOUNT_SHARE)
                    for key, v in share.items():
                        setattr(new, key, v)
                    a.fuel += new.fuel  # 연료는 모두 차량에
                    new.fuel = 0.0
                    a.kind = BRADLEY
                    new.stationary = a.stationary
                    new.starving = a.starving
                    self._log(side, o)
            else:
                rest.append(o)
        return rest

    def _absorb(self, a, b):
        for key in SHARED:
            setattr(a, key, getattr(a, key) + getattr(b, key))
        a.repairs += b.repairs
        a.platoons += b.platoons
        a.stationary = min(a.stationary, b.stationary)
        a.starving = max(a.starving, b.starving)
        del self.pieces[b.pid]

    # 2. 이동
    def _moves(self, side, orders):
        done = set()
        for o in orders:
            if o[0] != "move":
                continue
            pc = self._own(side, o[1])
            if pc is None or pc.pid in done:
                continue
            dest = tuple(o[2])
            if dest == pc.hex:
                continue
            opts = self.move_options(pc)
            if dest in opts:
                pc.hex = dest
                pc.moved = True
                done.add(pc.pid)
                self._log(side, o)

    # 4~5. 사격
    def _eff_power(self, pc):
        c = self.cfg
        if pc.ammo <= 0:
            return 0.0
        p = pc.power * self._starve_factor(pc)
        if pc.spec.entrench and not pc.moved:
            if pc.stationary >= c.entrench_turns:
                p *= c.entrench_mult
            if self.map.is_mountain(pc.hex):
                p *= c.mountain_mult
                if pc.kind == DISMOUNT:
                    p *= c.dismount_mountain_mult
        return p

    def _starve_factor(self, pc):
        if pc.starving <= 0:
            return 1.0
        return max(0.0, 1.0 - pc.starving / self.cfg.starve_turns)

    def _combat(self, orders, seen):
        c = self.cfg
        incoming = {}      # target pid -> 기대 피해
        fired = set()      # 이번 턴 사격한 말 (탄약 소모)
        attackers_of = {}  # target pid -> [attacker pid] (직사)
        eff = {pid: self._eff_power(pc) for pid, pc in self.pieces.items()}

        def add(tid, amount):
            incoming[tid] = incoming.get(tid, 0.0) + amount

        for side in (0, 1):
            for o in orders[side]:
                if o[0] == "attack":
                    a = self._own(side, o[1])
                    t = self.pieces.get(o[2])
                    if (a is None or t is None or t.side == side or a.pid in fired
                            or not a.spec.can_attack or eff[a.pid] <= 0
                            or t.pid not in seen[side]
                            or distance(a.hex, t.hex) > c.direct_fire_range):
                        continue
                    add(t.pid, c.attrition_coeff * eff[a.pid])
                    self.events.append(("direct", side, a.hex, t.hex))
                    attackers_of.setdefault(t.pid, []).append(a.pid)
                    fired.add(a.pid)
                    self._log(side, o)
                elif o[0] == "fire":
                    a = self._own(side, o[1])
                    h = tuple(o[2])
                    if (a is None or a.kind != ARTY or a.moved or a.pid in fired
                            or eff[a.pid] <= 0 or distance(a.hex, h) > c.artillery_range):
                        continue
                    targets = [t for t in self.pieces.values() if t.side != side and t.hex == h]
                    fired.add(a.pid)
                    self._log(side, o)
                    self.events.append(("arty", side, a.hex, h))
                    total = sum(t.power for t in targets)
                    if total <= 0:
                        continue
                    dmg = c.attrition_coeff * eff[a.pid]
                    for t in targets:
                        add(t.pid, dmg * t.power / total)

        # 자동 응사: 직사 공격을 받았고 이번 턴 스스로 쏘지 않은 말은 가장 강한 공격자에게 응사
        for tid, atk in attackers_of.items():
            t = self.pieces[tid]
            if t.pid in fired or not t.spec.can_attack or eff[t.pid] <= 0:
                continue
            best = max(atk, key=lambda pid: self.pieces[pid].power)
            add(best, c.attrition_coeff * eff[t.pid])
            self.events.append(("return", t.side, t.hex, self.pieces[best].hex))
            fired.add(t.pid)

        for pid in fired:
            pc = self.pieces[pid]
            use = c.ammo_per_fire * (1 if pc.kind == ARTY else pc.platoons)
            pc.ammo = max(0.0, pc.ammo - use)

        # 피해 적용: 마르코프 확률적 란체스터 (피해 단위 개수 ~ 포아송)
        for tid, exp_dmg in incoming.items():
            t = self.pieces[tid]
            q = t.spec.quantum
            n = self.rng.poisson(exp_dmg / q)
            loss = min(t.power, n * q)
            if loss <= 0:
                continue
            frac = loss / t.power if t.power > 0 else 1.0
            t.power -= loss
            # 완파 / 소파 / 중파 로 나눈다. 소파·중파는 수리되면 돌아온다.
            light = loss * c.dmg_light
            heavy = loss * max(0.0, 1.0 - c.dmg_destroyed - c.dmg_light)
            if light > 0:
                t.repairs.append([light, "소", c.repair_light_turns])
            if heavy > 0:
                t.repairs.append([heavy, "중", c.repair_heavy_turns])
            if t.kind in (SUPPLY, FSC):
                t.cargo_food *= 1 - frac
                t.cargo_ammo *= 1 - frac
                t.cargo_fuel *= 1 - frac

    # 6. 보급 (자동)
    def _near(self, pc, cands):
        """pc 에서 보급 거리 안에 있는 후보 중 가장 가까운 것."""
        r = self.cfg.supply_range
        best = [q for q in cands if q is not pc and distance(pc.hex, q.hex) <= r]
        return min(best, key=lambda q: distance(pc.hex, q.hex)) if best else None

    @staticmethod
    def _top_up(src, dst, food, ammo, fuel):
        """src 의 적재 물자로 dst 를 (food, ammo, fuel) 목표까지 채운다. 실제 준 양을 돌려준다."""
        g = [0.0, 0.0, 0.0]
        for i, (have, want, cargo) in enumerate((("food", food, "cargo_food"),
                                                 ("ammo", ammo, "cargo_ammo"),
                                                 ("fuel", fuel, "cargo_fuel"))):
            give = min(max(0.0, want - getattr(dst, have)), getattr(src, cargo))
            setattr(dst, have, getattr(dst, have) + give)
            setattr(src, cargo, getattr(src, cargo) - give)
            g[i] = give
        return g

    def _resupply(self, side):
        own = self.side_pieces(side)
        dist = [p for p in own if p.kind == SUPPLY]
        fsc = {p.bn: p for p in own if p.kind == FSC}
        # 기지(상급 지원부대)가 배급중대를 늘 채운다
        for d in dist:
            d.fill_cargo()
        # 배급중대 -> 전방지원중대 적재 보충
        for f in fsc.values():
            d = self._near(f, dist)
            if d is not None:
                for k in ("food", "ammo", "fuel"):
                    have, cap = "cargo_" + k, "cap_" + k
                    give = min(max(0.0, getattr(f, cap) - getattr(f, have)), getattr(d, have))
                    setattr(f, have, getattr(f, have) + give)
                    setattr(d, have, getattr(d, have) - give)
        # 말 -> 자기 물자: 자기 대대 전방지원중대, 없으면 배급중대 (거리 안)
        for pc in own:
            if pc.kind in (SUPPLY, FSC):
                src = pc          # 실은 물자로 자기도 먹는다
            else:
                f = fsc.get(pc.bn)
                src = f if f is not None and distance(pc.hex, f.hex) <= self.cfg.supply_range \
                    else self._near(pc, dist)
            if src is None:
                continue
            food, ammo, fuel = self._need(pc)
            self._top_up(src, pc, food, ammo, fuel)
            if pc.food > 0:
                pc.starving = 0

    # 7~8. 소모
    def _consume(self):
        h = self.cfg.turn_hours
        for pc in self.pieces.values():
            pc.food = max(0.0, pc.food - pc.platoons * h / 24.0)
            if pc.food <= 0:
                pc.starving += 1
            sp = pc.spec
            gph = sp.gph_move if pc.moved else sp.gph_idle
            pc.fuel = max(0.0, pc.fuel - sp.vehicles * pc.platoons * gph * h)
            pc.stationary = 0 if pc.moved else pc.stationary + 1

    # 8. 수리
    def _repair(self, side):
        """소파: 자기 대대 전방지원중대(없으면 정비중대) 8칸 안이면 진행.
        중파: 정비중대가 이 말이나 이 말의 전방지원중대 8칸 안이면 진행 (후송 수리)."""
        r = self.cfg.supply_range
        own = self.side_pieces(side)
        fsc = {p.bn: p for p in own if p.kind == FSC}
        maint = [p for p in own if p.kind == MAINT]
        for pc in own:
            if not pc.repairs:
                continue
            f = fsc.get(pc.bn)
            f = f if f is not None and f is not pc and distance(pc.hex, f.hex) <= r else None
            m = self._near(pc, maint) or (self._near(f, maint) if f is not None else None)
            ok = {"소": f is not None or m is not None, "중": m is not None}
            keep = []
            for item in pc.repairs:
                if ok[item[1]]:
                    item[2] -= 1
                if item[2] <= 0:
                    pc.power = min(pc.start_power, pc.power + item[0])
                    self.events.append(("repair", side, pc.hex, pc.hex))
                else:
                    keep.append(item)
            pc.repairs = keep

    # 9. 제거
    def _remove(self):
        bp = self.cfg.breakpoint
        dead = [pid for pid, pc in self.pieces.items()
                if pc.power < bp * pc.start_power or self._starve_factor(pc) <= 0]
        for pid in dead:
            pc = self.pieces[pid]
            why = "starved" if self._starve_factor(pc) <= 0 else "broken"
            self.events.append(("lost", pc.side, pc.hex, why))
            del self.pieces[pid]
            self.last_order.pop(pid, None)

    # 10. 승리
    def _check_victory(self):
        lost = []
        self.defeat_reason = [None, None]
        for s in (0, 1):
            own = self.side_pieces(s)
            why = None
            if not any(p.kind == BDE_HQ for p in own):
                why = "여단본부 제거"
            elif sum(p.power for p in own) <= self.cfg.power_defeat * self.start_total[s]:
                why = "전투력 30% 이하"
            elif not any(p.spec.maneuver for p in own):
                why = "전멸"
            self.defeat_reason[s] = why
            lost.append(why is not None)
        if lost[0] or lost[1]:
            self.done = True
            self.winner = None if lost[0] == lost[1] else (1 if lost[0] else 0)
            return
        for s in (0, 1):
            target = self.objective[1 - s]  # s 가 노리는 곳 = 상대 후방 목표
            here = [p for p in self.pieces.values() if p.hex == target]
            mine = any(p.side == s and p.spec.maneuver for p in here)
            theirs = any(p.side != s for p in here)
            self.hold[s] = self.hold[s] + 1 if (mine and not theirs) else 0
        wins = [self.hold[s] >= self.cfg.objective_hold_turns for s in (0, 1)]
        if wins[0] or wins[1]:
            self.done = True
            self.winner = None if wins[0] and wins[1] else (0 if wins[0] else 1)
