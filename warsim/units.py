"""부대 종류와 말(piece).

전투력은 무기 효과 지수 방식(WEI/WUV, 미 의회예산처 1988)으로 계산한 소대당 값.
    전차 94, 브래들리 71, 재블린 73, 개인화기 4, 포병 99
"""

from dataclasses import dataclass, field

TANK, INF, BRADLEY, DISMOUNT, RECON, ARTY, SUPPLY, BDE_HQ, BN_HQ, SIGNAL, FSC, MAINT, MED \
    = range(13)
KIND_NAMES = ["tank", "inf", "bradley", "dismount", "recon", "arty", "supply",
              "bde_hq", "bn_hq", "signal", "fsc", "maint", "med"]
HQ_KINDS = (BDE_HQ, BN_HQ, SIGNAL)
# SUPPLY = 여단지원대대 배급중대, FSC = 전방지원중대(대대마다), MAINT = 정비중대, MED = 의무중대
CARGO_KINDS = (SUPPLY, FSC)              # 물자를 싣고 다니는 말
NO_SPLIT = (ARTY, BDE_HQ, BN_HQ, SIGNAL, FSC, MAINT, MED)

# 전차중대: M1 14대 x 94 x 1.34 = 1763.6 -> 소대 587.9                     [근거]
# 보병중대: 브래들리 14 x 71 = 994, 재블린 9 x 73 + 개인화기 134 x 4 = 1193  [근거]
#           -> 소대: 차량 331.3 + 하차 397.7 = 729.0
# 정찰중대: M3 12대 가정 x 71 = 852 -> 소대 284                           [개념: 차량 수]
# 포병포대: M109A7 6문 x 99 = 594                                         [근거]
# 보급소대·전방지원중대·정비중대·의무중대: 자체 방호력 30                [개념]
# 본부 (MCoE SM 3-90, 2013 장비표)                                       [근거]
#   여단본부: M2A3 2 x 71 + 지휘장갑차(M1068, 장갑수송차 가중치 30) 3 x 30 = 232
#   대대본부(제병협동대대 본부중대): M1A2 1 x 94 x 1.34 + M2A3 1 x 71
#            + M3A3 3 x 71 + 지휘장갑차 6 x 30 = 590.0
#            기병·포병·지원대대 본부도 같은 값으로 둔다                  [개념]
#   여단 통신중대(여단공병대대 소속): 험비 20대, 장갑·주무장 없음 -> 30   [개념: 전투력]
_BDE_HQ = 2 * 71 + 3 * 30
_BN_HQ = 94 * 1.34 + 71 + 3 * 71 + 6 * 30
_TANK_PLT = 14 * 94 * 1.34 / 3
_IFV_PLT = 14 * 71 / 3
_DISMOUNT_PLT = (9 * 73 + 134 * 4) / 3


@dataclass(frozen=True)
class KindSpec:
    power: float          # 소대(포대)당 전투력
    quantum: float        # 피해 단위 (차량 1대, 분대 1개 등)
    vehicles: float       # 소대당 차량 수
    fuel_cap: float       # 차량 1대 연료 (갤런)
    gph_move: float       # 이동 시 시간당 연료
    gph_idle: float       # 대기 시 시간당 연료
    foot: bool = False
    can_attack: bool = True
    maneuver: bool = True  # 전멸 판정·목표 점유에 포함
    entrench: bool = True  # 진지 효과 대상


KIND_SPECS = {
    # M1: 505갤런, 야지 60gph, 대기 10gph [근거 GlobalSecurity]
    TANK: KindSpec(_TANK_PLT, 14 * 94 * 1.34 / 14, 14 / 3, 505, 60, 10),
    # M2: 험지 18gph, 대기 1.4gph [근거 army.mil], 연료 175갤런 [개념]
    INF: KindSpec(_IFV_PLT + _DISMOUNT_PLT, 71, 14 / 3, 175, 18, 1.4),
    BRADLEY: KindSpec(_IFV_PLT, 71, 14 / 3, 175, 18, 1.4),
    DISMOUNT: KindSpec(_DISMOUNT_PLT, _DISMOUNT_PLT / 3, 0, 0, 0, 0, foot=True),
    RECON: KindSpec(4 * 71, 71, 4, 175, 18, 1.4),
    # M109A7: 브래들리 차체 공유 -> 연료값 동일 가정 [개념]
    ARTY: KindSpec(6 * 99, 99, 6, 175, 18, 1.4, can_attack=False, maneuver=False, entrench=False),
    # 보급 트럭 8대, 155갤런, 8/1 gph [개념]
    SUPPLY: KindSpec(30, 10, 8, 155, 8, 1, can_attack=False, maneuver=False, entrench=False),
    # 본부 차량 연료는 브래들리 값 [개념]
    BDE_HQ: KindSpec(_BDE_HQ, 30, 5, 175, 18, 1.4, maneuver=False, entrench=False),
    BN_HQ: KindSpec(_BN_HQ, 30, 11, 175, 18, 1.4, maneuver=False, entrench=False),
    # 험비: 25갤런, 이동 4gph, 대기 1gph [개념]
    SIGNAL: KindSpec(30, 10, 20, 25, 4, 1, can_attack=False, maneuver=False, entrench=False),
    # 지원 중대: 트럭 8대 (보급소대 값과 같게) [개념]
    FSC: KindSpec(30, 10, 8, 155, 8, 1, can_attack=False, maneuver=False, entrench=False),
    MAINT: KindSpec(30, 10, 8, 155, 8, 1, can_attack=False, maneuver=False, entrench=False),
    MED: KindSpec(30, 10, 8, 155, 8, 1, can_attack=False, maneuver=False, entrench=False),
}
SUPPORT_KINDS = (SUPPLY, FSC, MAINT, MED)

SHARED = ("power", "start_power", "food", "ammo", "fuel", "cargo_food", "cargo_ammo",
          "cargo_fuel", "cap_food", "cap_ammo", "cap_fuel")  # 분리 시 나누고 합류 시 더하는 값

DISMOUNT_SHARE = _DISMOUNT_PLT / (_IFV_PLT + _DISMOUNT_PLT)


@dataclass
class Piece:
    pid: int
    side: int
    kind: int
    hex: tuple
    platoons: int
    power: float
    start_power: float
    food: float = 0.0         # 소대-일
    ammo: float = 0.0         # 소대 기본휴대량
    fuel: float = 0.0         # 갤런
    cargo_food: float = 0.0   # 보급부대 적재
    cargo_ammo: float = 0.0
    cargo_fuel: float = 0.0
    cap_food: float = 0.0     # 적재 한도 (배급중대·전방지원중대)
    cap_ammo: float = 0.0
    cap_fuel: float = 0.0
    stationary: int = 0       # 연속 정지 턴 수
    starving: int = 0         # 식량 바닥 후 경과 턴 수
    moved: bool = False       # 이번 턴 이동 여부
    bn: int = -1              # 소속 대대 번호 (-1: 여단 직할)
    repairs: list = field(default_factory=list)  # 수리 대기 [전투력, "소"|"중", 남은 턴]

    @property
    def spec(self):
        return KIND_SPECS[self.kind]

    def food_cap(self, cfg):
        return self.platoons * cfg.food_start_days

    def ammo_cap(self):
        return float(self.platoons)

    def armed(self):
        return self.spec.can_attack or self.kind == ARTY

    def fill_cargo(self):
        self.cargo_food, self.cargo_ammo, self.cargo_fuel = self.cap_food, self.cap_ammo, self.cap_fuel

    def fuel_cap(self):
        return self.spec.vehicles * self.platoons * self.spec.fuel_cap

    def scale(self, frac):
        """전투력·물자를 비율대로 나눈 몫을 돌려주고, 자신은 나머지를 갖는다."""
        out = {}
        for k in SHARED:
            v = getattr(self, k)
            out[k] = v * frac
            setattr(self, k, v - out[k])
        return out
