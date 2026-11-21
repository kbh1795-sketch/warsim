"""교리(부대 운용 예규): 지휘·지원 말은 신경망이 아니라 정해진 규칙대로 움직인다.

실제 군대처럼 보급·지휘소 이동 같은 일상 업무는 예규가 맡고, 여단장(신경망)은
기동부대(전차·보병·브래들리·하차보병·정찰·포병)로 전술만 정한다.

규칙 (매 턴, 통신과 상관없이 — 예규는 명령 없이도 지켜진다):
  대대본부     : 자기 대대 기동 중대들의 무게중심에서 뒤로 BN_BACK 칸
  전방지원중대 : 자기 대대 기동 중대들의 무게중심에서 뒤로 FSC_BACK 칸
  배급·정비·의무중대 : 전방지원중대들의 무게중심에서 뒤로 BSA_BACK 칸
                       (배급중대에서 보급 거리 밖인 전방지원중대는 배급중대 쪽으로 당김)
  여단본부·통신중대   : 배급중대 옆 (뒤로 1칸)
  이동은 목표 칸까지 갈 수 있는 칸 중 가장 가까운 곳으로, 지뢰지대는 피한다.
  이미 목표에서 KEEP 칸 안이면 움직이지 않는다.
"""

from .hexgrid import distance
from .units import (BDE_HQ, BN_HQ, FSC, MAINT, MED, SIGNAL, SUPPLY)

DOCTRINE_KINDS = (BDE_HQ, BN_HQ, SIGNAL, FSC, SUPPLY, MAINT, MED)

# [개념] 뒤로 떨어지는 칸 수. 보급 거리 8칸 안에 들도록 정함:
#   기동 중대 -> 전방지원중대 3칸, 전방지원중대 -> 배급중대 4칸 (army.mil field trains:
#   전투 보급지원 지휘소가 야전 보급 거점보다 5~6km 앞)
BN_BACK = 1
FSC_BACK = 3
BSA_BACK = 4
KEEP = 1


def _centroid(hexes):
    n = len(hexes)
    return (round(sum(h[0] for h in hexes) / n), round(sum(h[1] for h in hexes) / n))


def _back(game, side, h, k):
    """자기 진영 뒤쪽(기지 방향)으로 k 칸."""
    c = game.cfg
    row = h[1] - k if side == 0 else h[1] + k
    return (min(max(h[0], 0), c.width - 1), min(max(row, 0), c.length - 1))


def _toward(game, pc, target):
    if distance(pc.hex, target) <= KEEP:
        return None
    opts = [h for h in game.move_options(pc) if not game.map.is_mine(h)]
    if not opts:
        return None
    best = min(opts, key=lambda h: distance(h, target))
    if distance(best, target) >= distance(pc.hex, target):
        return None
    return ("move", pc.pid, best)


def doctrine_orders(game, side):
    own = game.side_pieces(side)
    man = {}
    for p in own:
        if p.kind not in DOCTRINE_KINDS:
            man.setdefault(p.bn, []).append(p.hex)
    orders = []

    def go(pc, target):
        o = _toward(game, pc, target)
        if o:
            orders.append(o)

    # 전방지원중대·대대본부 목표
    goal = {}
    for p in own:
        if p.kind in (BN_HQ, FSC) and p.bn in man:
            goal[p.pid] = _back(game, side, _centroid(man[p.bn]),
                                BN_BACK if p.kind == BN_HQ else FSC_BACK)
    fsc = [p for p in own if p.kind == FSC and p.pid in goal]
    if fsc:
        bsa = _back(game, side, _centroid([goal[p.pid] for p in fsc]), BSA_BACK)
        # 배급중대가 모든 전방지원중대에 닿도록, 너무 먼 전방지원중대는 배급중대 쪽으로 당긴다
        reach = game.cfg.supply_range - 1
        for p in fsc:
            g = goal[p.pid]
            t = 1.0
            while distance(g, bsa) > reach and t > 0:
                t -= 0.1
                g = (round(bsa[0] + (goal[p.pid][0] - bsa[0]) * t),
                     round(bsa[1] + (goal[p.pid][1] - bsa[1]) * t))
            goal[p.pid] = g
        for p in own:
            if p.kind in (SUPPLY, MAINT, MED):
                goal[p.pid] = bsa
            elif p.kind in (BDE_HQ, SIGNAL):
                goal[p.pid] = _back(game, side, bsa, 1)
    for p in own:
        if p.pid in goal:
            go(p, goal[p.pid])
    return orders
