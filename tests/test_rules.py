import numpy as np
import pytest

from warsim import Config, Game
from warsim.hexgrid import FLAT, MOUNTAIN, HexMap, distance
from warsim.hexgrid import line
from warsim.units import ARTY, BDE_HQ, BN_HQ, BRADLEY, DISMOUNT, INF, SIGNAL, SUPPLY, TANK


def flat_game(comm=False, **kw):
    """comm=False 면 통신 범위를 무한으로 두어 통신 제약 없이 다른 규칙만 시험한다."""
    if not comm:
        kw.setdefault("comm_unit_range", 999)
        kw.setdefault("comm_bn_range", 999)
    g = Game(Config(mountain_fraction=0.0, **kw), rng=np.random.default_rng(0))
    g.deploy(0)
    g.deploy(1)
    return g


def put(g, pid, h):
    g.pieces[pid].hex = h


def first(g, side, kind):
    return next(p for p in g.side_pieces(side) if p.kind == kind)


def clear_others(g, keep):
    for pid in list(g.pieces):
        if pid not in keep:
            del g.pieces[pid]


# ---------------- 격자 ----------------
def test_hex_distance_and_neighbors():
    m = HexMap(10, 10, np.zeros((10, 10), dtype=np.int8))
    for h in [(0, 0), (5, 5), (9, 9), (3, 4)]:
        for n in m.neighbors(h):
            assert distance(h, n) == 1
    assert len(m.neighbors((5, 5))) == 6
    assert len(m.within((5, 5), 2)) == 19


def test_mountain_slows_vehicles():
    t = np.full((20, 20), MOUNTAIN, dtype=np.int8)
    m = HexMap(20, 20, t)
    far = max(distance((10, 10), h) for h in m.reachable((10, 10), 12, 1.5))
    assert far == 8
    t[:] = FLAT
    m = HexMap(20, 20, t)
    far = max(distance((10, 10), h) for h in m.reachable((10, 10), 12, 1.5))
    assert far == 12


def test_map_is_point_symmetric():
    g = Game(Config(), rng=np.random.default_rng(3))
    t = g.map.terrain
    assert np.array_equal(t, t[::-1, ::-1])


# ---------------- 분리·합류·하차 ----------------
def test_split_and_merge_conserve_power_and_supplies():
    g = flat_game()
    a = first(g, 0, TANK)
    before = (a.power, a.food, a.fuel)
    g.step([("split", a.pid, 1)], [])
    parts = [p for p in g.side_pieces(0) if p.kind == TANK and p.hex == a.hex]
    assert sum(p.platoons for p in parts) >= 3
    new = max(parts, key=lambda p: p.pid)
    assert new.platoons == 1 and a.platoons == 2
    assert new.power + a.power == pytest.approx(before[0])
    g.step([("merge", a.pid, new.pid)], [])
    assert a.platoons == 3 and new.pid not in g.pieces


def test_dismount_and_mount():
    g = flat_game()
    a = first(g, 0, INF)
    total = a.power
    g.step([("dismount", a.pid)], [])
    d = next(p for p in g.side_pieces(0) if p.kind == DISMOUNT)
    assert a.kind == BRADLEY
    assert a.power + d.power == pytest.approx(total)
    assert d.fuel == 0 and d.power > a.power  # 하차 보병 비중 55%
    g.step([("mount", a.pid, d.pid)], [])
    assert a.kind == INF and d.pid not in g.pieces


def test_dismounted_moves_one_hex():
    g = flat_game()
    a = first(g, 0, INF)
    put(g, a.pid, (15, 5))
    g.step([("dismount", a.pid)], [])
    d = next(p for p in g.side_pieces(0) if p.kind == DISMOUNT)
    opts = g.move_options(d)
    assert max(distance(d.hex, h) for h in opts) == 1


# ---------------- 전투 ----------------
def test_attack_damages_and_return_fire():
    g = flat_game()
    a, b = first(g, 0, TANK), first(g, 1, TANK)
    clear_others(g, {a.pid, b.pid})
    put(g, a.pid, (10, 20)); put(g, b.pid, (11, 20))
    losses_a, losses_b = [], []
    for i in range(200):
        g.rng = np.random.default_rng(i)
        a.power = b.power = a.start_power
        a.ammo = b.ammo = 3.0
        g.done = False
        g._combat([[("attack", a.pid, b.pid)], []], [{b.pid}, set()])
        losses_a.append(a.start_power - a.power)
        losses_b.append(b.start_power - b.power)
    exp = 0.05 * a.start_power
    assert np.mean(losses_b) == pytest.approx(exp, rel=0.25)
    assert np.mean(losses_a) > 0  # 자동 응사


def test_cannot_attack_unseen_or_far():
    g = flat_game()
    a, b = first(g, 0, TANK), first(g, 1, TANK)
    clear_others(g, {a.pid, b.pid})
    put(g, a.pid, (10, 20)); put(g, b.pid, (10, 25))
    p0 = b.power
    g.step([("attack", a.pid, b.pid)], [])
    assert b.power == p0


def test_artillery_cannot_move_and_fire():
    g = flat_game()
    art, b = first(g, 0, ARTY), first(g, 1, TANK)
    clear_others(g, {art.pid, b.pid, first(g, 0, TANK).pid})
    put(g, art.pid, (10, 5)); put(g, b.pid, (10, 20))
    ammo = art.ammo
    g.step([("move", art.pid, (10, 6)), ("fire", art.pid, (10, 20))], [])
    assert art.ammo == ammo


def test_entrench_and_mountain_multiply_power():
    g = flat_game()
    a = first(g, 0, TANK)
    a.stationary = 1
    assert g._eff_power(a) == pytest.approx(a.power * 1.3)
    g.map.terrain[a.hex[1], a.hex[0]] = MOUNTAIN
    assert g._eff_power(a) == pytest.approx(a.power * 1.3 * 1.5)
    a.moved = True
    assert g._eff_power(a) == pytest.approx(a.power)


def test_breakpoint_removes_below_30_percent():
    g = flat_game()
    a = first(g, 0, TANK)
    a.power = 0.29 * a.start_power
    g.step([], [])
    assert a.pid not in g.pieces


def test_no_ammo_no_damage():
    g = flat_game()
    a = first(g, 0, TANK)
    a.ammo = 0
    assert g._eff_power(a) == 0


# ---------------- 보급 ----------------
def test_starvation_removes_after_two_days():
    g = flat_game()
    a = first(g, 0, TANK)
    a.food = 0
    for _ in range(95):
        a.food = 0
        g.step([], [])
    assert a.pid in g.pieces
    assert g._starve_factor(a) < 0.05
    a.food = 0
    g.step([], [])
    assert a.pid not in g.pieces


def test_no_fuel_no_move_but_can_fire():
    g = flat_game()
    a = first(g, 0, TANK)
    a.fuel = 0
    assert list(g.move_options(a)) == [a.hex]
    assert g._eff_power(a) > 0


def test_supply_delivery_same_hex():
    g = flat_game()
    a, s = first(g, 0, TANK), first(g, 0, SUPPLY)
    put(g, s.pid, a.hex)
    a.food = 0.0
    a.fuel = 0.0
    g.step([("deliver", s.pid)], [])
    assert a.food > 0 and a.fuel > 0


def test_supply_refills_at_base():
    g = flat_game()
    s = first(g, 0, SUPPLY)
    s.cargo_food = 0
    put(g, s.pid, g.base[0])
    g.step([], [])
    assert s.cargo_food == pytest.approx(g.cfg.supply_food_cap * s.platoons)


# ---------------- 승리 ----------------
def test_objective_hold_wins():
    g = flat_game(objective_hold_turns=3)
    a = first(g, 0, TANK)
    target = g.objective[1]
    for p in g.side_pieces(1):
        p.hex = (0, g.cfg.length - 1)
    put(g, a.pid, target)
    for _ in range(3):
        done, w = g.step([], [])
    assert done and w == 0


def test_annihilation_wins():
    g = flat_game()
    for pid in [p.pid for p in g.side_pieces(1)]:
        del g.pieces[pid]
    done, w = g.step([], [])
    assert done and w == 0


def test_brigade_hq_destroyed_loses():
    g = flat_game()
    del g.pieces[first(g, 1, BDE_HQ).pid]
    done, w = g.step([], [])
    assert done and w == 0 and g.defeat_reason[1] == "여단본부 제거"


def test_total_power_30_percent_loses():
    g = flat_game()
    for p in g.side_pieces(1):
        if p.kind == TANK:
            del g.pieces[p.pid]
    for p in g.side_pieces(1):
        if p.kind not in (BDE_HQ,):
            p.power = 0.31 * p.start_power   # 각자 격멸선(30%) 위, 전체로는 30% 근처
    total = sum(p.power for p in g.side_pieces(1))
    assert total <= 0.3 * g.start_total[1]
    done, w = g.step([], [])
    assert done and w == 0 and g.defeat_reason[1] == "전투력 30% 이하"


# ---------------- 통신 ----------------
def _comm_setup():
    g = flat_game(comm=True)
    bde, sig = first(g, 0, BDE_HQ), first(g, 0, SIGNAL)
    a = first(g, 0, TANK)
    hq = next(p for p in g.side_pieces(0) if p.kind == BN_HQ and p.bn == a.bn)
    put(g, bde.pid, (15, 2)); put(g, sig.pid, (15, 2))
    put(g, hq.pid, (15, 20)); put(g, a.pid, (15, 25))
    return g, bde, sig, hq, a


def test_comm_ranges():
    g, bde, sig, hq, a = _comm_setup()
    c = g.comm(0)
    assert hq.pid in c and a.pid in c          # 대대본부 18칸(35 이내), 전차 5칸(8 이내)
    put(g, a.pid, (15, 29))                    # 대대본부와 9칸
    assert a.pid not in g.comm(0)
    put(g, a.pid, (15, 25)); put(g, hq.pid, (15, 40))   # 여단본부와 38칸
    assert hq.pid not in g.comm(0) and a.pid not in g.comm(0)
    put(g, sig.pid, (15, 15))                  # 통신중대를 앞으로 -> 다시 연결
    assert hq.pid in g.comm(0)


def test_mountain_blocks_comm_unless_on_high_ground():
    g, bde, sig, hq, a = _comm_setup()
    mid = line(hq.hex, a.hex)[2]
    g.map.terrain[mid[1], mid[0]] = MOUNTAIN
    g.map._los_cache.clear()
    assert a.pid not in g.comm(0)
    g.map.terrain[a.hex[1], a.hex[0]] = MOUNTAIN   # 전차가 산 위로
    g.map._los_cache.clear()
    assert a.pid in g.comm(0)


def test_cut_off_unit_keeps_last_order_and_ignores_new():
    g, bde, sig, hq, a = _comm_setup()
    g._update_comm()
    dest = (15, 40)
    g.step([("move", a.pid, (15, 33))], [])     # 연결 상태에서 명령: 8칸 전진
    assert a.hex == (15, 33)
    assert a.pid not in g.conn[0]               # 대대본부와 13칸 -> 끊김
    g.step([("move", a.pid, (15, 25))], [])     # 새 명령은 무시
    assert a.hex == (15, 33)                    # 목적지 도착 후 제자리
    g.last_order[a.pid] = ("move", a.pid, dest)
    g.step([], [])
    assert a.hex == dest                        # 마지막 이동 명령은 계속 따라감


def test_cut_off_sightings_unknown_to_commander():
    g, bde, sig, hq, a = _comm_setup()
    e = first(g, 1, TANK)
    put(g, e.pid, (16, 26))
    g._update_comm()
    assert e.pid in g.known_enemies(0)
    put(g, a.pid, (15, 35)); put(g, e.pid, (16, 36))
    g._update_comm()
    assert e.pid in g.visible_enemies(0) and e.pid not in g.known_enemies(0)
