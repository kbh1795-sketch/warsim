"""PPO 셀프플레이 학습 (플루토 방식).

- 처음부터 셀프플레이. 사람 기보나 교리를 넣지 않는다.
- 보상은 끝났을 때만: 승 +1, 패 -1, 무 0. 중간 보상 없음.
- 상대: 절반은 현재 자기 자신, 절반은 저장된 챔피언 중 하나.
- 배치 유형은 밴딧이 승패 기록으로 고른다.
- 일정 주기마다 현재 신경망이 모든 챔피언을 상대로 승률 50% 초과면 새 챔피언으로 저장.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..config import Config
from .deploy import DeploymentBandit
from .encode import OBS_KEYS, decode, encode, stack
from .model import Net, to_torch
from .policy import NetPolicy, RandomPolicy, match, new_game, score

ACT_KEYS = ("type", "move_dir", "move_dist", "atk", "fire")


@dataclass
class TrainConfig:
    n_envs: int = 8            # 동시에 돌리는 판 수
    rollout: int = 32          # 갱신 한 번에 판마다 모으는 턴 수
    updates: int = 200
    lr: float = 3e-4
    gamma: float = 0.999
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 3
    minibatch: int = 64
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    self_play: float = 0.5     # 상대가 현재 자신일 확률
    eval_every: int = 25       # 챔피언 심사 주기 (갱신 횟수)
    eval_games: int = 8        # 챔피언 하나당 심사 판수
    threads: int = 0           # 0 이면 torch 기본값
    out: str = "runs/run0"
    rng: int = 0


class Slot:
    """진행 중인 판 하나."""

    def __init__(self):
        self.game = None
        self.opp = None      # None 이면 현재 자신 (양 진영 모두 학습 대상)
        self.tmpl = [None, None]

    def learn_sides(self):
        return (0, 1) if self.opp is None else (self.learner,)


class Trainer:
    def __init__(self, tc: TrainConfig, cfg: Config = None):
        self.tc, self.cfg = tc, cfg or Config()
        if tc.threads:
            torch.set_num_threads(tc.threads)
        self.rng = np.random.default_rng(tc.rng)
        torch.manual_seed(tc.rng)
        self.net = Net()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=tc.lr)
        self.bandit = DeploymentBandit()
        self.out = Path(tc.out)
        (self.out / "champions").mkdir(parents=True, exist_ok=True)
        self.champions = []   # [(이름, Net)]
        self.log = []
        self.games_done = 0
        self.start_update = 1
        ck = self.out / "checkpoint.pt"
        if ck.exists():
            self._resume(ck)
        else:
            self._add_champion()  # 0번 챔피언 = 학습 전 신경망
        self.slots = [Slot() for _ in range(tc.n_envs)]
        for s in self.slots:
            self._reset(s)

    # ------------------------------------------------------------ 이어 하기
    def _checkpoint(self, u):
        torch.save({"net": self.net.state_dict(), "opt": self.opt.state_dict(),
                    "bandit": [self.bandit.wins, self.bandit.games],
                    "champions": [n for n, _ in self.champions],
                    "games_done": self.games_done, "update": u,
                    "rng": self.rng.bit_generator.state}, self.out / "checkpoint.pt")

    def _resume(self, ck):
        d = torch.load(ck, weights_only=False)
        self.net.load_state_dict(d["net"])
        self.opt.load_state_dict(d["opt"])
        self.bandit.wins, self.bandit.games = d["bandit"]
        for n in d["champions"]:
            net = Net()
            net.load_state_dict(torch.load(self.out / "champions" / f"{n}.pt"))
            net.eval()
            self.champions.append((n, net))
        self.games_done = d["games_done"]
        self.start_update = d["update"] + 1
        self.rng.bit_generator.state = d["rng"]
        print(f"이어 하기: 갱신 {d['update']}, 챔피언 {len(self.champions)}개", flush=True)

    # ------------------------------------------------------------ 챔피언
    def _add_champion(self):
        name = f"champion_{len(self.champions):03d}"
        torch.save(self.net.state_dict(), self.out / "champions" / f"{name}.pt")
        self.bandit.save(self.out / "champions" / f"{name}_bandit.json")
        frozen = Net()
        frozen.load_state_dict(self.net.state_dict())
        frozen.eval()
        self.champions.append((name, frozen))
        return name

    # ------------------------------------------------------------ 판 관리
    def _reset(self, s):
        rng = self.rng
        if rng.random() < self.tc.self_play:
            s.opp = None
            s.tmpl = [self.bandit.choose(rng), self.bandit.choose(rng)]
        else:
            s.opp = self.champions[rng.integers(len(self.champions))][1]
            s.learner = int(rng.integers(2))
            s.tmpl = [None, None]
            s.tmpl[s.learner] = self.bandit.choose(rng)
            s.tmpl[1 - s.learner] = int(rng.integers(len(self.bandit.wins)))
        s.game = new_game(self.cfg, rng, s.tmpl)

    # ------------------------------------------------------------ 수집
    def collect(self):
        T, E = self.tc.rollout, len(self.slots)
        rew = np.zeros((T, E, 2), np.float32)
        val = np.zeros((T + 1, E, 2), np.float32)
        done = np.zeros((T, E, 2), bool)
        valid = np.zeros((T, E, 2), bool)
        samples = {}  # (t, e, side) -> (obs dict, act dict, logp)
        self.net.eval()
        for t in range(T):
            items, keys = [], []
            opp_items = {}
            for e, s in enumerate(self.slots):
                for side in s.learn_sides():
                    items.append((s.game, side))
                    keys.append((e, side))
                if s.opp is not None:
                    opp_items.setdefault(id(s.opp), (s.opp, []))[1].append((e, 1 - s.learner))
            encs = [encode(g, sd) for g, sd in items]
            obs = stack(encs)
            acts, logp, v = self.net.act(to_torch(obs))
            acts = {k: a.numpy() for k, a in acts.items()}
            logp, v = logp.numpy(), v.numpy()
            orders = {}
            for i, ((g, sd), en, (e, side)) in enumerate(zip(items, encs, keys)):
                a = {k: acts[k][i] for k in ACT_KEYS}
                orders[(e, side)] = decode(g, sd, en, a)
                o = {k: obs[k][i] for k in OBS_KEYS}
                o["planes"] = o["planes"].astype(np.float16)
                samples[(t, e, side)] = (o, a, logp[i])
                val[t, e, side] = v[i]
                valid[t, e, side] = True
            for net, lst in opp_items.values():
                pol = NetPolicy(net)
                outs = pol.orders([(self.slots[e].game, sd) for e, sd in lst])
                for (e, sd), od in zip(lst, outs):
                    orders[(e, sd)] = od
            for e, s in enumerate(self.slots):
                s.game.step(orders[(e, 0)], orders[(e, 1)])
                if s.game.done:
                    w = s.game.winner
                    for side in s.learn_sides():
                        r = s.game.result(side)
                        rew[t, e, side] = r
                        done[t, e, side] = True
                        self.bandit.update(s.tmpl[side], (r + 1) / 2)
                    self.games_done += 1
                    self.results.append((s.opp is None, w, s.learn_sides(), s.game.turn))
                    self._reset(s)
        # 마지막 상태 가치 (부트스트랩)
        items, keys = [], []
        for e, s in enumerate(self.slots):
            for side in s.learn_sides():
                items.append((s.game, side))
                keys.append((e, side))
        _, _, v = self.net.act(to_torch(stack([encode(g, sd) for g, sd in items])))
        for (e, side), vv in zip(keys, v.numpy()):
            val[T, e, side] = vv

        # GAE. 학습 대상이 아니던 칸은 판이 바뀐 직후라 앞 칸에서 done 으로 끊겨 있다.
        adv = np.zeros((T, E, 2), np.float32)
        last = np.zeros((E, 2), np.float32)
        g, lam = self.tc.gamma, self.tc.lam
        for t in reversed(range(T)):
            nonterm = 1.0 - done[t]
            delta = rew[t] + g * val[t + 1] * nonterm - val[t]
            last = delta + g * lam * nonterm * last
            last = np.where(valid[t], last, 0.0)
            adv[t] = last
        ret = adv + val[:T]
        batch = [(samples[k], adv[k], ret[k]) for k in samples]
        return batch

    # ------------------------------------------------------------ 갱신
    def update(self, batch):
        tc = self.tc
        self.net.train()
        n = len(batch)
        advs = np.array([b[1] for b in batch], np.float32)
        # 보상이 끝에만 있어서 대부분 0 이다. 표준편차로 나누면 가치 오차(잡음)가
        # 크게 부풀려져 정책이 아무 근거 없이 굳는다. 그래서 정규화하지 않는다.
        stats = []
        for _ in range(tc.epochs):
            idx = self.rng.permutation(n)
            for st in range(0, n, tc.minibatch):
                mb = idx[st:st + tc.minibatch]
                o = to_torch({k: np.stack([batch[i][0][0][k] for i in mb]) for k in OBS_KEYS})
                a = to_torch({k: np.stack([batch[i][0][1][k] for i in mb]) for k in ACT_KEYS})
                old = torch.from_numpy(np.stack([batch[i][0][2] for i in mb]))
                ad = torch.from_numpy(advs[mb]).unsqueeze(1)
                rt = torch.from_numpy(np.array([batch[i][2] for i in mb], np.float32))
                lp, ent, v = self.net.evaluate(o, a)
                pm = o["ctrl_mask"].float()
                ratio = torch.exp(lp - old)
                s1, s2 = ratio * ad, torch.clamp(ratio, 1 - tc.clip, 1 + tc.clip) * ad
                pl = -(torch.min(s1, s2) * pm).sum() / pm.sum().clamp(min=1)
                vl = ((v - rt) ** 2).mean()
                loss = pl + tc.vf_coef * vl - tc.ent_coef * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                with torch.no_grad():
                    clipfrac = (((ratio - 1).abs() > tc.clip).float() * pm).sum() / pm.sum()
                stats.append((pl.item(), vl.item(), ent.mean().item(), clipfrac.item()))
        return np.mean(stats, 0)

    # ------------------------------------------------------------ 심사
    def gate(self):
        """모든 챔피언에게 승률 50% 초과면 새 챔피언."""
        cur = NetPolicy(self.net, bandit=self.bandit)
        self.net.eval()
        rates = {}
        for name, ch in self.champions:
            bpath = self.out / "champions" / f"{name}_bandit.json"
            r = score(match(cur, NetPolicy(ch, bandit=DeploymentBandit.load(bpath)),
                            self.cfg, self.tc.eval_games, self.rng))
            rates[name] = r
            if r <= 0.5:
                return None, rates
        return self._add_champion(), rates

    def vs_random(self, n=8):
        self.net.eval()
        return match(NetPolicy(self.net, bandit=self.bandit), RandomPolicy(self.rng),
                     self.cfg, n, self.rng)

    # ------------------------------------------------------------ 반복
    def run(self):
        tc = self.tc
        (self.out / "train_config.json").write_text(json.dumps(asdict(tc), ensure_ascii=False))
        t0 = time.time()
        for u in range(self.start_update, tc.updates + 1):
            self.results = []
            tc0 = time.time()
            batch = self.collect()
            tcol = time.time() - tc0
            pl, vl, ent, cf = self.update(batch)
            rec = {"update": u, "samples": len(batch), "games_done": self.games_done,
                   "policy_loss": round(float(pl), 4), "value_loss": round(float(vl), 4),
                   "entropy": round(float(ent), 3), "clipfrac": round(float(cf), 3),
                   "collect_s": round(tcol, 1), "total_s": round(time.time() - t0, 1),
                   "mean_len": round(float(np.mean([r[3] for r in self.results])), 1)
                   if self.results else None}
            if u % tc.eval_every == 0:
                name, rates = self.gate()
                w, d, l = self.vs_random()
                rec.update({"gate": {k: round(v, 3) for k, v in rates.items()},
                            "new_champion": name, "vs_random": [int(w), int(d), int(l)],
                            "bandit": self.bandit.table()})
                torch.save(self.net.state_dict(), self.out / "latest.pt")
                self.bandit.save(self.out / "latest_bandit.json")
                self._checkpoint(u)
            self.log.append(rec)
            with open(self.out / "log.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(json.dumps(rec, ensure_ascii=False), flush=True)
        torch.save(self.net.state_dict(), self.out / "latest.pt")
        self.bandit.save(self.out / "latest_bandit.json")
