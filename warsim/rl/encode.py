"""신경망 입력(관측)과 출력(행동)을 게임과 연결한다.

관점: 진영 1 은 지도를 점대칭으로 뒤집어서 본다. 그래서 신경망은 항상
"내 후방이 0행, 적 후방이 마지막 행"인 같은 그림을 본다. 방향도 반대로 바뀐다.

관측 (안개 + 통신 제약 적용: 통신이 연결된 말이 본 적만 보인다)
    planes  (C, L, W)   지도 평면
    tokens  (P, F_TOK)  내 말 하나당 한 줄
    glob    (F_GLB,)    전체 상황
    atk     (P, KA, F_ATK)  말마다 직사 표적 후보
    fire    (P, KF, F_FIRE) 포병마다 사격 칸 후보

행동 (내 말 하나당)
    type  9가지: 대기, 이동, 공격, 포병사격, 분리, 합류, 하차, 탑승, (보급: 자동이 되어 늘 막힘)
    move_dir 6, move_dist 4 (1, 3, 6, 12칸 쪽으로)
    atk  KA 중 하나, fire KF 중 하나
"""

import numpy as np

from ..doctrine import DOCTRINE_KINDS
from ..hexgrid import _CUBE_DIRS, distance, to_cube
from ..hexgrid import MINE, MOUNTAIN, RIVER
from ..units import (ARTY, BN_HQ, BRADLEY, CARGO_KINDS, DISMOUNT, INF, KIND_NAMES,
                     KIND_SPECS, NO_SPLIT)

N_KIND = len(KIND_NAMES)
N_BN = 7        # 대대 6 + 여단 직할
P_MAX = 80      # 한 진영 최대 말 수 (시작 31, 소대 단위로 다 쪼개면 약 70)
KA = 8          # 직사 표적 후보 수
KF = 16         # 포병 사격 칸 후보 수
DISTS = (1, 3, 6, 12)
HOLD, MOVE, ATTACK, FIRE, SPLIT, MERGE, DISMOUNT_A, MOUNT, DELIVER = range(9)
N_TYPE = 9
C_PLANES = 1 + 2 * N_KIND + 2 + 4 + 2   # 마지막 2장: 강(지나갈 수 없음), 지뢰지대
F_TOK = N_KIND + N_BN + 16
F_GLB = 10
F_ATK = 6 + N_KIND
F_FIRE = 4 + N_KIND
POWER_SCALE = 1000.0


def mirror(side, h, cfg):
    return h if side == 0 else (cfg.width - 1 - h[0], cfg.length - 1 - h[1])


def game_dir(side, d):
    return d if side == 0 else (d + 3) % 6


def _cube_dist(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


class Encoded:
    """한 진영의 한 턴 관측과, 행동을 명령으로 바꾸는 데 필요한 정보."""
    __slots__ = ("planes", "tokens", "glob", "atk", "fire", "type_mask", "atk_mask",
                 "fire_mask", "piece_mask", "pids", "atk_ids", "fire_hex", "merge_with",
                 "mount_with", "n", "pos", "ctrl_mask")


def encode(game, side):
    cfg = game.cfg
    W, L = cfg.width, cfg.length
    own = game.side_pieces(side)[:P_MAX]
    seen_ids = game.known_enemies(side)
    conn = game.conn[side]
    enemies = [game.pieces[i] for i in seen_ids]

    e = Encoded()
    planes = np.zeros((C_PLANES, L, W), dtype=np.float32)
    terr = game.map.terrain if side == 0 else game.map.terrain[::-1, ::-1]
    planes[0] = terr == MOUNTAIN
    planes[C_PLANES - 2] = terr == RIVER
    planes[C_PLANES - 1] = terr == MINE
    K = N_KIND
    for p in own:
        c, r = mirror(side, p.hex, cfg)
        planes[1 + p.kind, r, c] += p.power / POWER_SCALE
    for p in enemies:
        c, r = mirror(side, p.hex, cfg)
        planes[1 + K + p.kind, r, c] += p.power / POWER_SCALE
    for p in own:
        if p.pid not in conn:
            continue
        for h in game.map.within(p.hex, game.vision(p)):
            c, r = mirror(side, h, cfg)
            planes[1 + 2 * K, r, c] = 1.0
        if p.kind == BN_HQ:  # 대대본부 통신 범위 (거리만)
            for h in game.map.within(p.hex, cfg.comm_unit_range):
                c, r = mirror(side, h, cfg)
                planes[2 + 2 * K, r, c] = 1.0
    b = 3 + 2 * K
    for ch, h in ((b, game.objective[side]), (b + 1, game.objective[1 - side]),
                  (b + 2, game.base[side]), (b + 3, game.base[1 - side])):
        c, r = mirror(side, h, cfg)
        planes[ch, r, c] = 1.0
    e.planes = planes

    tokens = np.zeros((P_MAX, F_TOK), dtype=np.float32)
    piece_mask = np.zeros(P_MAX, dtype=bool)
    ctrl_mask = np.zeros(P_MAX, dtype=bool)   # 통신 연결: 이번 턴 명령을 줄 수 있는 말
    type_mask = np.zeros((P_MAX, N_TYPE), dtype=bool)
    type_mask[:, HOLD] = True
    atk = np.zeros((P_MAX, KA, F_ATK), dtype=np.float32)
    atk_mask = np.zeros((P_MAX, KA), dtype=bool)
    fire = np.zeros((P_MAX, KF, F_FIRE), dtype=np.float32)
    fire_mask = np.zeros((P_MAX, KF), dtype=bool)
    pids = np.full(P_MAX, -1, dtype=np.int64)
    atk_ids = np.full((P_MAX, KA), -1, dtype=np.int64)
    fire_hex = [[None] * KF for _ in range(P_MAX)]
    merge_with = np.full(P_MAX, -1, dtype=np.int64)
    mount_with = np.full(P_MAX, -1, dtype=np.int64)

    by_hex = {}
    for p in own:
        by_hex.setdefault(p.hex, []).append(p)

    pos = np.zeros(P_MAX, dtype=np.int64)  # 지도 평면에서 r*W + c
    for i, p in enumerate(own):
        pids[i] = p.pid
        piece_mask[i] = True
        c, r = mirror(side, p.hex, cfg)
        pos[i] = r * W + c
        sp = KIND_SPECS[p.kind]
        t = tokens[i]
        t[p.kind] = 1
        t[N_KIND + (p.bn if p.bn >= 0 else N_BN - 1)] = 1
        k = N_KIND + N_BN
        ok = p.pid in conn
        ctrl_mask[i] = ok and not (game.cfg.doctrine and p.kind in DOCTRINE_KINDS)
        t[k + 13] = float(ok)
        t[k + 14] = p.cargo_food / p.cap_food if p.kind in CARGO_KINDS and p.cap_food else 0.0
        t[k + 15] = sum(x[0] for x in p.repairs) / max(p.start_power, 1e-9)  # 수리 대기 비율
        t[k:k + 13] = (c / W, r / L, p.platoons / 3.0, p.power / max(p.start_power, 1e-9),
                       p.power / POWER_SCALE, p.food / max(p.food_cap(cfg), 1e-9),
                       p.ammo / max(p.ammo_cap(), 1e-9) if p.ammo_cap() else 0.0,
                       p.fuel / p.fuel_cap() if p.fuel_cap() else 1.0,
                       float(p.moved), min(p.stationary, 4) / 4.0, p.starving / cfg.starve_turns,
                       float(game.map.is_mountain(p.hex)), float(sp.foot))
        if not ok:
            continue   # 끊긴 말: 신경망은 보지만 명령은 못 줌 (마지막 명령을 이어감)

        # 이동 가능
        if len(game.move_options(p)) > 1:
            type_mask[i, MOVE] = True
        # 직사 표적 후보
        if sp.can_attack and p.ammo > 0 and p.power > 0:
            cands = [q for q in enemies if distance(p.hex, q.hex) <= cfg.direct_fire_range]
            cands.sort(key=lambda q: -q.power)
            for j, q in enumerate(cands[:KA]):
                qc, qr = mirror(side, q.hex, cfg)
                atk[i, j, :6] = ((qc - c) / 2.0, (qr - r) / 2.0, distance(p.hex, q.hex),
                                 q.power / POWER_SCALE, q.power / max(q.start_power, 1e-9),
                                 q.platoons / 3.0)
                atk[i, j, 6 + q.kind] = 1
                atk_mask[i, j] = True
                atk_ids[i, j] = q.pid
            if cands:
                type_mask[i, ATTACK] = True
        # 포병 사격 후보 (보이는 적이 있는 칸)
        if p.kind == ARTY and p.ammo > 0:
            hexes = {}
            for q in enemies:
                if distance(p.hex, q.hex) <= cfg.artillery_range:
                    hexes.setdefault(q.hex, []).append(q)
            ranked = sorted(hexes.items(), key=lambda kv: -sum(q.power for q in kv[1]))
            for j, (h, qs) in enumerate(ranked[:KF]):
                hc, hr = mirror(side, h, cfg)
                fire[i, j, :4] = ((hc - c) / cfg.artillery_range, (hr - r) / cfg.artillery_range,
                                  distance(p.hex, h) / cfg.artillery_range,
                                  sum(q.power for q in qs) / POWER_SCALE)
                for q in qs:
                    fire[i, j, 4 + q.kind] += q.power / POWER_SCALE
                fire_mask[i, j] = True
                fire_hex[i][j] = h
            if ranked:
                type_mask[i, FIRE] = True
        # 분리 / 합류 / 하차 / 탑승 / 보급
        mates = [q for q in by_hex[p.hex] if q.pid != p.pid]
        if p.kind not in NO_SPLIT and p.platoons > 1:
            type_mask[i, SPLIT] = True
        same = [q for q in mates if q.kind == p.kind and q.bn == p.bn
                and p.kind not in NO_SPLIT and q.pid in conn]
        if same:
            type_mask[i, MERGE] = True
            merge_with[i] = same[0].pid
        if p.kind == INF:
            type_mask[i, DISMOUNT_A] = True
        if p.kind == BRADLEY:
            dm = [q for q in mates if q.kind == DISMOUNT and q.platoons == p.platoons
                  and q.bn == p.bn]
            if dm:
                type_mask[i, MOUNT] = True
                mount_with[i] = dm[0].pid

    own_pw = sum(p.power for p in own)
    en_pw = sum(p.power for p in enemies)
    e.glob = np.array([game.turn / cfg.max_turns,
                       game.hold[side] / cfg.objective_hold_turns,
                       game.hold[1 - side] / cfg.objective_hold_turns,
                       own_pw / (POWER_SCALE * 20), en_pw / (POWER_SCALE * 20),
                       len(own) / P_MAX, len(enemies) / P_MAX,
                       own_pw / game.start_total[side],
                       ctrl_mask.sum() / max(len(own), 1), 1.0], dtype=np.float32)
    e.tokens, e.piece_mask, e.type_mask = tokens, piece_mask, type_mask
    e.ctrl_mask = ctrl_mask
    e.atk, e.atk_mask, e.fire, e.fire_mask = atk, atk_mask, fire, fire_mask
    e.pids, e.atk_ids, e.fire_hex = pids, atk_ids, fire_hex
    e.merge_with, e.mount_with = merge_with, mount_with
    e.n = len(own)
    e.pos = pos
    return e


OBS_KEYS = ("planes", "tokens", "glob", "atk", "fire", "type_mask", "atk_mask",
            "fire_mask", "piece_mask", "pos", "ctrl_mask")


def stack(encs):
    """여러 관측을 배치 배열 dict 로 묶는다."""
    return {k: np.stack([getattr(e, k) for e in encs]) for k in OBS_KEYS}


def _move_target(game, side, pc, d, dist_idx):
    """방향 d(신경망 관점)로 DISTS[dist_idx] 칸 간 지점에 가장 가까운 도달 가능 칸."""
    gd = _CUBE_DIRS[game_dir(side, d)]
    x, y, z = to_cube(*pc.hex)
    k = DISTS[dist_idx]
    goal = (x + gd[0] * k, y + gd[1] * k, z + gd[2] * k)
    opts = game.move_options(pc)
    return min(opts, key=lambda h: (_cube_dist(to_cube(*h), goal), opts[h]))


def decode(game, side, enc, act):
    """act: dict of int arrays (P_MAX,) -> 명령 리스트."""
    orders = []
    for i in range(enc.n):
        if not enc.ctrl_mask[i]:
            continue
        pid = int(enc.pids[i])
        pc = game.pieces.get(pid)
        if pc is None:
            continue
        t = int(act["type"][i])
        if t == MOVE:
            h = _move_target(game, side, pc, int(act["move_dir"][i]), int(act["move_dist"][i]))
            if h != pc.hex:
                orders.append(("move", pid, h))
        elif t == ATTACK:
            tid = int(enc.atk_ids[i, int(act["atk"][i])])
            if tid >= 0:
                orders.append(("attack", pid, tid))
        elif t == FIRE:
            h = enc.fire_hex[i][int(act["fire"][i])]
            if h is not None:
                orders.append(("fire", pid, h))
        elif t == SPLIT:
            orders.append(("split", pid, 1))
        elif t == MERGE and enc.merge_with[i] >= 0:
            orders.append(("merge", pid, int(enc.merge_with[i])))
        elif t == DISMOUNT_A:
            orders.append(("dismount", pid))
        elif t == MOUNT and enc.mount_with[i] >= 0:
            orders.append(("mount", pid, int(enc.mount_with[i])))
    return orders
