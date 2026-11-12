"""셀프플레이 강화학습 (CPU). 같은 --out 으로 다시 실행하면 마지막 저장 지점부터 이어서 한다.

    python scripts/train.py --updates 200 --out runs/run0
    python scripts/train.py --updates 5 --envs 4 --rollout 16 --eval-every 5 --out runs/smoke

결과: runs/<이름>/log.jsonl, champions/champion_NNN.pt (+ 배치 밴딧 기록), latest.pt
"""

import argparse

from warsim.rl.train import Trainer, TrainConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-games", type=int, default=8)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--rng", type=int, default=0, help="난수 번호 (독립 반복 실행용)")
    ap.add_argument("--out", default="runs/run0")
    a = ap.parse_args()
    tc = TrainConfig(n_envs=a.envs, rollout=a.rollout, updates=a.updates,
                     eval_every=a.eval_every, eval_games=a.eval_games, threads=a.threads,
                     out=a.out, rng=a.rng)
    Trainer(tc).run()


if __name__ == "__main__":
    main()
