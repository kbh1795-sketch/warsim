import numpy as np
import torch

from warsim import Config, Game
from warsim.rl.deploy import DeploymentBandit, placement
from warsim.rl.encode import P_MAX, MOVE, decode, encode, mirror, stack
from warsim.rl.model import Net, to_torch
from warsim.rl.policy import NetPolicy, RandomPolicy, match


def _game(seed=0):
    g = Game(Config(), rng=np.random.default_rng(seed))
    g.deploy(0)
    g.deploy(1)
    return g


def test_mirror_symmetric_view():
    g = _game()
    e0, e1 = encode(g, 0), encode(g, 1)
    # 지형은 점대칭이라 양쪽 관점에서 같은 그림
    assert np.array_equal(e0.planes[0], e1.planes[0])
    # 자기 목표·기지 평면도 같은 위치
    assert np.array_equal(e0.planes[23], e1.planes[23])
    assert np.array_equal(e0.planes[25], e1.planes[25])


def test_move_direction_flips_for_side1():
    """신경망 관점 '앞'(행 증가)이 진영 1 에서는 실제로 행 감소."""
    g = _game()
    for side in (0, 1):
        enc = encode(g, side)
        i = int(np.flatnonzero(enc.ctrl_mask)[0])   # 신경망이 지휘하는 첫 말
        p = g.pieces[int(enc.pids[i])]
        r0 = mirror(side, p.hex, g.cfg)[1]
        act = {k: np.zeros(P_MAX, int) for k in ("type", "move_dir", "move_dist", "atk", "fire")}
        act["type"][i] = MOVE
        act["move_dist"][i] = 3
        best = None
        for d in range(6):
            act["move_dir"][i] = d
            o = decode(g, side, enc, act)
            if o:
                r1 = mirror(side, o[0][2], g.cfg)[1]
                best = max(best or -99, r1 - r0)
        assert best > 0


def test_net_act_shapes_and_masks():
    g = _game()
    net = Net()
    o = to_torch(stack([encode(g, 0), encode(g, 1)]))
    a, lp, v = net.act(o)
    assert lp.shape == (2, P_MAX) and v.shape == (2,)
    assert (v.abs() <= 1).all()
    tm = o["type_mask"]
    chosen = tm.gather(-1, a["type"].unsqueeze(-1)).squeeze(-1)
    assert chosen[o["piece_mask"]].all()
    lp2, _, _ = net.evaluate(o, a)
    assert torch.allclose(lp, lp2, atol=1e-5)


def test_placement_in_zone_and_bandit():
    g = Game(Config(), rng=np.random.default_rng(1))
    rng = np.random.default_rng(1)
    for side in (0, 1):
        pl = placement(g, side, 1, rng)
        zone = set(g.deploy_zone(side))
        assert all(h in zone for h in pl.values())
    b = DeploymentBandit()
    for _ in range(50):
        b.update(2, 1.0)
        b.update(0, 0.0)
    assert sum(b.choose(rng) == 2 for _ in range(100)) > 80


def test_match_runs():
    rng = np.random.default_rng(0)
    w, d, l = match(NetPolicy(Net()), RandomPolicy(rng), Config(max_turns=20), 2, rng)
    assert w + d + l == 2
