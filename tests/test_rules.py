import numpy as np
import pytest

from warsim import Config, Game
from warsim.hexgrid import FLAT, MOUNTAIN, HexMap, distance
from warsim.hexgrid import line
from warsim.hexgrid import CROSSING, MINE, RIVER
from warsim.units import (ARTY, BDE_HQ, BN_HQ, BRADLEY, DISMOUNT, FSC, INF, MAINT, SIGNAL,
                          SUPPLY, SUPPORT_KINDS, TANK)


def flat_game(comm=False, **kw):
    """comm=False 면 통신 범위를 무한으로 두어 통신 제약 없이 다른 규칙만 시험한다."""
    if not comm:
        kw.setdefault("comm_unit_range", 999)
        kw.setdefault("comm_bn_range", 999)
    kw.setdefault("doctrine", False)   # 규칙 시험은 말을 직접 놓으므로 예규 이동은 끔
    kw.setdefault("river_crossings", 0)
    kw.setdefault("mine_fraction", 0.0)
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
    for p in list(g.pieces.values()):
        if p.kind in SUPPORT_KINDS:   # 보급이 끊긴 상황
            del g.pieces[p.pid]
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


def supply_line(g, side=0):
    """전차 1개 중대, 그 대대의 전방지원중대, 배급중대, 정비중대를 한 줄로 세운다 (각 6칸 간격)."""
    a = first(g, side, TANK)
    f = next(p for p in g.side_pieces(side) if p.kind == FSC and p.bn == a.bn)
    d, m = first(g, side, SUPPLY), first(g, side, MAINT)
    put(g, a.pid, (10, 40)); put(g, f.pid, (10, 34)); put(g, d.pid, (10, 28)); put(g, m.pid, (10, 30))
    return a, f, d, m


def test_auto_resupply_chain_keeps_unit_fed():
    """공격받지 않으면 굶지 않는다: 부대 1일분이 다 떨어질 시간이 지나도 배가 차 있다."""
    g = flat_game()
    a, f, d, m = supply_line(g)
    for _ in range(200):
        g.step([], [])
    assert a.pid in g.pieces and a.starving == 0 and a.food > 0
    assert a.fuel == pytest.approx(a.fuel_cap(), rel=0.01)


def test_resupply_needs_range():
    g = flat_game()
    a, f, d, m = supply_line(g)
    put(g, f.pid, (10, 40 - g.cfg.supply_range - 1))   # 전방지원중대가 너무 멀다
    a.food = 0.0
    g.step([], [])
    assert a.starving == 1


def test_fsc_refilled_by_distribution_company():
    g = flat_game()
    a, f, d, m = supply_line(g)
    f.cargo_food = 0.0
    g.step([], [])
    assert f.cargo_food > 0.5 * f.cap_food
    put(g, d.pid, (10, 34 - g.cfg.supply_range - 1))  # 배급중대가 멀면 못 채운다
    f.cargo_food = 0.0
    g.step([], [])
    assert f.cargo_food == pytest.approx(0.0, abs=1e-9)


def test_light_repair_needs_fsc_and_takes_a_day():
    g = flat_game()
    a, f, d, m = supply_line(g)
    put(g, m.pid, (0, 0))           # 정비중대는 멀리
    start = a.power
    a.power -= 30.0
    a.repairs.append([30.0, "소", g.cfg.repair_light_turns])
    for _ in range(g.cfg.repair_light_turns - 1):
        g.step([], [])
    assert a.power == pytest.approx(start - 30.0)
    g.step([], [])
    assert a.power == pytest.approx(start)


def test_heavy_repair_needs_maintenance_company():
    g = flat_game()
    a, f, d, m = supply_line(g)
    put(g, m.pid, (0, 0))           # 정비중대가 멀면 중파는 멈춘다
    a.power -= 30.0
    a.repairs.append([30.0, "중", 2])
    for _ in range(5):
        g.step([], [])
    assert a.repairs and a.repairs[0][2] == 2
    put(g, m.pid, (10, 30))         # 전방지원중대(10,34) 8칸 안 -> 후송 수리 진행
    g.step([], [])
    g.step([], [])
    assert not a.repairs


def test_combat_damage_creates_repairs():
    g = flat_game()
    a = first(g, 0, TANK)
    e = first(g, 1, TANK)
    clear_others(g, {a.pid, e.pid, first(g, 0, BDE_HQ).pid, first(g, 1, BDE_HQ).pid})
    g.start_total = [1.0, 1.0]   # 말을 지웠으니 '전체 전투력 30%' 패배가 바로 나지 않게
    put(g, a.pid, (10, 30)); put(g, e.pid, (10, 31))
    for _ in range(20):
        if a.pid not in g.pieces or e.pid not in g.pieces or e.repairs:
            break
        g.step([("attack", a.pid, e.pid)], [])
    lost = e.start_power - e.power
    assert e.pid in g.pieces and lost > 0
    assert sum(x[0] for x in e.repairs) == pytest.approx(lost * (1 - g.cfg.dmg_destroyed))


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


# ---------------- 강·지뢰지대 ----------------
def test_map_is_point_symmetric_with_river_and_mines():
    g = Game(Config(), rng=np.random.default_rng(3))
    t = g.map.terrain
    assert (t == t[::-1, ::-1]).all()
    assert (t == RIVER).any() and (t == CROSSING).any() and (t == MINE).any()
    zone = list(g.deploy_zone(0)) + list(g.deploy_zone(1))
    assert all(t[r, c] not in (RIVER, MINE) for c, r in zone)


def test_river_blocks_except_crossing():
    g = flat_game()
    W, mid = g.cfg.width, g.cfg.length // 2
    t = g.map.terrain
    t[mid - 1, :] = RIVER
    t[mid, :] = RIVER
    t[mid - 1, 5] = t[mid, 5] = CROSSING
    g.map.clear_cache()
    a = first(g, 0, TANK)
    put(g, a.pid, (9, mid - 2))
    opts = g.move_options(a)
    assert all(t[h[1], h[0]] != RIVER for h in opts)
    assert (5, mid) in opts                      # 도하 지점은 건널 수 있다
    assert any(h[1] > mid for h in opts)         # 도하 지점을 지나 건너편으로


def test_minefield_stops_and_damages():
    g = flat_game()
    a = first(g, 0, TANK)
    put(g, a.pid, (10, 20))
    t = g.map.terrain
    t[21, 10] = MINE
    g.map.clear_cache()
    opts = g.move_options(a)
    assert (10, 21) in opts
    start = a.power
    g.step([("move", a.pid, (10, 21))], [])
    assert a.hex == (10, 21)
    lost = start - a.power
    assert lost > 0
    assert sum(x[0] for x in a.repairs) == pytest.approx(lost * (1 - g.cfg.dmg_destroyed))


def test_timeout_is_a_loss_for_both():
    """기한 안에 아무도 목표를 달성하지 못하면 양쪽 모두 패배."""
    g = flat_game(max_turns=3)
    for _ in range(3):
        g.step([], [])
    assert g.done and g.timed_out and g.winner is None
    assert g.result(0) == -1.0 and g.result(1) == -1.0


# ---------------- 교리 (예규) ----------------
def test_doctrine_moves_support_and_ignores_net_orders():
    """예규가 켜지면 지휘·지원 말은 신경망 명령을 무시하고 기동 중대 뒤를 따라간다."""
    g = flat_game(doctrine=True)
    a = first(g, 0, TANK)
    f = next(p for p in g.side_pieces(0) if p.kind == FSC and p.bn == a.bn)
    hq = first(g, 0, BDE_HQ)
    for p in g.side_pieces(0):
        if p.bn == a.bn and p.kind not in (FSC, BN_HQ):
            put(g, p.pid, (10, 20))
    put(g, f.pid, (10, 2))
    start_hq = hq.hex
    g.step([("move", hq.pid, (0, 5))], [])     # 신경망이 여단본부를 옮기려 해도
    assert hq.hex != (0, 5) or start_hq == (0, 5)
    for _ in range(5):
        g.step([], [])
    assert f.hex[1] > 2 and f.hex[1] < 20                    # 앞으로 따라붙되 기동 중대보다 뒤
    assert distance(f.hex, (10, 20)) <= g.cfg.supply_range


def test_doctrine_kinds_not_controlled_by_net():
    from warsim.rl.encode import encode
    from warsim.doctrine import DOCTRINE_KINDS
    g = flat_game(doctrine=True)
    enc = encode(g, 0)
    for i, pid in enumerate(enc.pids):
        if enc.piece_mask[i] and g.pieces[int(pid)].kind in DOCTRINE_KINDS:
            assert not enc.ctrl_mask[i]
