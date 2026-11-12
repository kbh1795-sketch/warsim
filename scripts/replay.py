"""한 판을 두고, 재생 파일을 만들어 브라우저로 연다.

    python scripts/replay.py                                    # 무작위 대 무작위
    python scripts/replay.py --blue runs/run0/latest.pt         # 학습 신경망(청) 대 무작위(홍)
    python scripts/replay.py --blue runs/run0/latest.pt --red runs/run0/champions/champion_002.pt
    python scripts/replay.py --game 7 --no-open
"""

import argparse
import webbrowser
from pathlib import Path

import numpy as np
import torch

from warsim import Config
from warsim.replay import Recorder, end_reason
from warsim.rl.deploy import DeploymentBandit, TEMPLATES
from warsim.rl.model import Net
from warsim.rl.policy import NetPolicy, RandomPolicy, new_game


def load_policy(path, rng):
    if not path:
        return RandomPolicy(rng)
    net = Net()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    bp = Path(str(path).replace(".pt", "_bandit.json"))
    return NetPolicy(net, bandit=DeploymentBandit.load(bp) if bp.exists() else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, default=0, help="판 번호 (같은 번호면 같은 판이 재현됨)")
    ap.add_argument("--blue", default="", help="청(진영 0) 신경망 .pt (없으면 무작위)")
    ap.add_argument("--red", default="", help="홍(진영 1) 신경망 .pt (없으면 무작위)")
    ap.add_argument("--out", default="replays")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.game)
    pols = [load_policy(args.blue, rng), load_policy(args.red, rng)]
    tmpl = [p.template(rng) for p in pols]
    g = new_game(Config(), rng, tmpl)
    rec = Recorder(g)
    with torch.no_grad():
        while not g.done:
            g.step(pols[0].orders([(g, 0)])[0], pols[1].orders([(g, 1)])[0])
            rec.record()

    Path(args.out).mkdir(exist_ok=True)
    path = Path(args.out) / f"game_{args.game}.html"
    rec.save(path, end_reason(g))
    names = [TEMPLATES[t] if t is not None else "무작위" for t in tmpl]
    print(f"{path}  ({g.turn}턴, 승자 {g.winner}, {end_reason(g)}, 배치 청 {names[0]} / 홍 {names[1]})")
    if not args.no_open:
        webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()
