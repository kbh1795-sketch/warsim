"""신경망 하나가 모든 말을 지휘한다 (플루토 방식).

    지도 평면 --CNN--> 칸별 특징 --(말 위치에서 꺼냄)--+
    말 정보 ------------------------------------------+--> 트랜스포머(말끼리 서로 봄)
                                                       --> 말마다 행동 머리
    전체 요약 토큰 ------------------------------------> 가치 머리 (tanh, -1 패 ~ +1 승)

행동은 말마다 나눠서 고른다: 먼저 종류(9), 종류가 이동이면 방향·거리,
공격이면 표적 후보 중 하나, 포병사격이면 사격 칸 후보 중 하나.
할 수 없는 행동은 마스크로 막는다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import (HOLD, ATTACK, C_PLANES, DISTS, F_ATK, F_FIRE, F_GLB, F_TOK, FIRE, MOVE,
                     N_TYPE)

NEG = -1e9
HOLD_BIAS = 2.0


def _masked_logits(logits, mask):
    return logits.masked_fill(~mask, NEG)


def _cat_logp_ent(logits, mask, idx):
    lg = _masked_logits(logits, mask)
    logp_all = F.log_softmax(lg, -1)
    logp = logp_all.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    p = logp_all.exp()
    ent = -(p * logp_all.masked_fill(~mask, 0.0)).sum(-1)
    return logp, ent, p


class Net(nn.Module):
    def __init__(self, d=64, cnn=32, layers=2, heads=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(C_PLANES, cnn, 3, padding=1), nn.ReLU(),
            nn.Conv2d(cnn, cnn, 3, padding=1), nn.ReLU(),
            nn.Conv2d(cnn, cnn, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(cnn, cnn, 3, padding=4, dilation=4), nn.ReLU(),
        )
        self.tok_in = nn.Linear(F_TOK + cnn, d)
        self.glob_in = nn.Linear(F_GLB + 2 * cnn, d)
        enc_layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True)
        self.tf = nn.TransformerEncoder(enc_layer, layers, enable_nested_tensor=False)
        self.type_head = nn.Linear(d, N_TYPE)
        self.dir_head = nn.Linear(d, 6)
        self.dist_head = nn.Linear(d, len(DISTS))
        self.atk_in = nn.Linear(F_ATK, d)
        self.atk_out = nn.Sequential(nn.ReLU(), nn.Linear(d, 1))
        self.fire_in = nn.Linear(F_FIRE, d)
        self.fire_out = nn.Sequential(nn.ReLU(), nn.Linear(d, 1))
        self.value = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        # 처음엔 '대기'를 더 자주 고르게 시작 (학습하면서 바뀜) [학습 초기값]
        with torch.no_grad():
            self.type_head.bias[HOLD] += HOLD_BIAS

    def forward(self, o):
        B, P = o["tokens"].shape[:2]
        fmap = self.cnn(o["planes"])                       # (B, cnn, L, W)
        flat = fmap.flatten(2)                             # (B, cnn, L*W)
        at = torch.gather(flat, 2, o["pos"].unsqueeze(1).expand(-1, flat.shape[1], -1))
        x = self.tok_in(torch.cat([o["tokens"], at.transpose(1, 2)], -1))
        g = self.glob_in(torch.cat([o["glob"], flat.mean(2), flat.amax(2)], -1)).unsqueeze(1)
        seq = torch.cat([g, x], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool), ~o["piece_mask"]], 1)
        h = self.tf(seq, src_key_padding_mask=pad)
        hg, hp = h[:, 0], h[:, 1:]
        out = {
            "type": self.type_head(hp),
            "dir": self.dir_head(hp),
            "dist": self.dist_head(hp),
            "atk": self.atk_out(self.atk_in(o["atk"]) + hp.unsqueeze(2)).squeeze(-1),
            "fire": self.fire_out(self.fire_in(o["fire"]) + hp.unsqueeze(2)).squeeze(-1),
            "value": torch.tanh(self.value(hg).squeeze(-1)),
        }
        return out

    # -------------------------------------------------------------- 행동
    @torch.no_grad()
    def act(self, o, greedy=False):
        out = self.forward(o)
        acts = {}
        for name, key, mask in (("type", "type", o["type_mask"]),
                                ("move_dir", "dir", None), ("move_dist", "dist", None),
                                ("atk", "atk", o["atk_mask"]), ("fire", "fire", o["fire_mask"])):
            lg = out[key]
            if mask is not None:
                # 후보가 하나도 없으면 첫 칸만 열어둔다 (선택되지 않으므로 상관없음)
                m = mask.clone()
                m[..., 0] |= ~m.any(-1)
                lg = _masked_logits(lg, m)
            acts[name] = lg.argmax(-1) if greedy else torch.distributions.Categorical(
                logits=lg).sample()
        logp, _ = self.evaluate_out(out, o, acts)
        return acts, logp, out["value"]

    def evaluate_out(self, out, o, a):
        """말별 선택 행동 로그확률 (B, P) 과 평균 엔트로피 (B,).
        PPO 비율은 말마다 따로 계산한다 (진영 전체 곱으로 하면 말 수십 개라 비율이 폭주)."""
        pm = o["ctrl_mask"]   # 명령을 줄 수 있는 말만
        lp_t, ent_t, p_t = _cat_logp_ent(out["type"], o["type_mask"], a["type"])
        ones6 = torch.ones_like(out["dir"], dtype=torch.bool)
        ones4 = torch.ones_like(out["dist"], dtype=torch.bool)
        lp_d, ent_d, _ = _cat_logp_ent(out["dir"], ones6, a["move_dir"])
        lp_s, ent_s, _ = _cat_logp_ent(out["dist"], ones4, a["move_dist"])
        am = o["atk_mask"].clone()
        am[..., 0] |= ~am.any(-1)
        fm = o["fire_mask"].clone()
        fm[..., 0] |= ~fm.any(-1)
        lp_a, ent_a, _ = _cat_logp_ent(out["atk"], am, a["atk"])
        lp_f, ent_f, _ = _cat_logp_ent(out["fire"], fm, a["fire"])
        t = a["type"]
        lp = lp_t + (t == MOVE) * (lp_d + lp_s) + (t == ATTACK) * lp_a + (t == FIRE) * lp_f
        ent = ent_t + p_t[..., MOVE] * (ent_d + ent_s) + p_t[..., ATTACK] * ent_a \
            + p_t[..., FIRE] * ent_f
        pmf = pm.float()
        return lp * pmf, (ent * pmf).sum(1) / pmf.sum(1).clamp(min=1)

    def evaluate(self, o, a):
        out = self.forward(o)
        lp, ent = self.evaluate_out(out, o, a)
        return lp, ent, out["value"]


def to_torch(batch):
    t = {}
    for k, v in batch.items():
        t[k] = torch.from_numpy(v) if not isinstance(v, torch.Tensor) else v
        if t[k].dtype == torch.float16:
            t[k] = t[k].float()
    return t
