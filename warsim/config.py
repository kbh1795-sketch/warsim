"""게임 규칙의 모든 수치.

각 값 옆 주석의 표시:
    [근거]  미군 자료에서 가져온 값 (출처는 docs/rules.md)
    [개념]  근거가 없어 정한 모델링 가정 -> 민감도 분석 대상
    [유도]  다른 값에서 계산한 값
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    # ---------------- 지도 ----------------
    hex_km: float = 1.0                  # [근거] ST 100-3 전차중대 거점 폭 1,000m
    width: int = 30                      # [개념] 칸 수 (좌우)
    length: int = 100                    # [개념] 칸 수 (앞뒤). 양측 배치 구역 사이가 포병 사거리보다 멀도록
    deploy_depth: int = 10               # [개념] 각자 배치 구역 깊이 (칸)
    mountain_fraction: float = 0.3       # [개념] 산지 비율
    objective_depth: int = 5             # [개념] 목표 지점: 자기 뒤쪽 끝에서 몇 칸 앞
    objective_hold_turns: int = 48       # [개념] 목표 점유 유지 시간 (48턴 = 24시간)

    # ---------------- 시간 ----------------
    turn_hours: float = 0.5              # [유도] 도보 2.4km/h 로 1칸 = 약 25분
    max_turns: int = 336                 # [개념] 7일

    # ---------------- 이동 (한 턴 최대) ----------------
    vehicle_mp: float = 12.0             # [근거] FM 34-130 GO 24km/h x 0.5h / 1km
    vehicle_mountain_cost: float = 1.5   # [근거] SLOW-GO 16km/h -> 산지 8칸
    foot_mp: float = 1.0                 # [근거] FM 21-18 주간 야지 2.4km/h
    foot_mountain_cost: float = 1.0      # [개념] 도보는 산지 감속 없음

    # ---------------- 시야 (칸) ----------------
    vision_recon: int = 4                # [근거] 기갑학교 1989 정찰병 4~5km
    vision_default: int = 2              # [개념] 전차·보병·브래들리
    vision_rear: int = 1                 # [개념] 포병·보급
    dismount_detect: int = 1             # [개념] 하차 보병은 이 거리 안에서만 발견

    # ---------------- 사격 ----------------
    direct_fire_range: int = 1           # [근거] ST 100-3 이상적 교전 거리 1,500m -> 옆 칸
    artillery_range: int = 22            # [근거] M109A7 표준탄 22km

    # ---------------- 전투 ----------------
    attrition_coeff: float = 0.05        # [개념] 받는 피해 = 계수 x 상대 유효 전투력 (턴당)
    entrench_turns: int = 1              # [개념] 이 턴 수 이상 정지하면 진지
    entrench_mult: float = 1.3           # [근거] Dupuy 급편방어
    mountain_mult: float = 1.5           # [근거] Dupuy 험한 지형 방어
    dismount_mountain_mult: float = 1.0  # [개념] 하차 보병 산지 추가 배수 (민감도 분석)
    breakpoint: float = 0.3              # [근거] C-WAM: 남은 전투력 30% 미만이면 제거

    # ---------------- 보급 ----------------
    food_start_days: float = 3.0         # [개념]
    starve_turns: int = 96               # [개념] 식량 바닥 후 2일(96턴)에 걸쳐 0으로
    ammo_per_fire: float = 0.1           # [개념] 한 턴 사격에 기본휴대량의 10%
    supply_food_cap: float = 20.0        # [개념] 보급소대 1개 적재: 소대-일
    supply_ammo_cap: float = 10.0        # [개념] 보급소대 1개 적재: 소대 기본휴대량
    supply_fuel_cap: float = 20000.0     # [개념] 보급소대 1개 적재: 갤런

    # ---------------- 통신 (FM 11-32 SINCGARS) ----------------
    comm_unit_range: int = 8             # [근거] 무전기 고출력 8km: 중대·소대 <-> 자기 대대본부
    comm_bn_range: int = 35              # [근거] 차량 증폭기 35km: 대대본부 <-> 여단본부·통신중대
    comm_los: bool = True                # [근거] 가시선 필요 (사이에 산지 칸이면 끊김, 끝이 산지면 통함)

    # ---------------- 승리 ----------------
    power_defeat: float = 0.3            # [근거] C-WAM 30%: 전체 전투력이 시작의 30% 이하면 패배

    # ---------------- 편제 (양측 동일, 중대 = 소대 3개) ----------------
    n_tank_companies: int = 5            # [근거] FM 3-96 기갑여단
    n_inf_companies: int = 4             # [근거]
    n_recon_companies: int = 3           # [근거]
    n_batteries: int = 3                 # [근거]
    n_supply_companies: int = 1          # [개념]
    # 대대: 보병형 제병협동(보병2·전차1), 전차형 제병협동 x2(전차2·보병1), 기병(정찰3),
    #       포병(포대3), 지원(보급). 각 대대본부 1, 여단본부 1, 여단 통신중대 1
    #       [근거] battleorder.org ABCT 2016, MCoE SM 3-90
    platoons_per_company: int = 3        # [근거] SM 3-90

    # 소대당 값은 units.py 의 KIND_SPECS 에 있음
    extra: dict = field(default_factory=dict)
