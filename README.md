# warsim

여단급 지상전 셀프플레이 시뮬레이션. 규칙과 근거는 `docs/rules.md`, 모든 수치는 `warsim/config.py`.

## 설치
```
poetry install
```

## 실행
```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 poetry run pytest -q tests   # 규칙 테스트 (ROS pytest 플러그인 차단)
poetry run python scripts/run_random.py  # 무작위 플레이어 대전, 속도 측정
poetry run python scripts/replay.py      # 한 판을 두고 재생 화면을 브라우저로 엶 (replays/game_0.html)
poetry run python scripts/replay.py --game 7   # 다른 판 (같은 번호면 같은 판이 재현됨)
poetry run python scripts/train.py --updates 400 --out runs/run0   # 셀프플레이 강화학습 (CPU)
poetry run python scripts/replay.py --blue runs/run0/latest.pt     # 학습 신경망(청) 대 무작위(홍) 재생
```

## 구조
- `warsim/config.py`  모든 수치 ([근거] / [개념] / [유도] 표시)
- `warsim/hexgrid.py` 1km 육각형 격자, 거리, 이동 범위
- `warsim/units.py`   병종별 전투력(무기 효과 지수), 연료, 말(piece)
- `warsim/game.py`    턴 처리 엔진
- `warsim/agents.py`  확인용 무작위 플레이어
- `warsim/replay.py`  한 판 기록 → 재생 파일(HTML) 저장
- `warsim/rl/encode.py`  관측(안개 적용 지도 평면 + 내 말 목록)과 말별 행동 변환
- `warsim/rl/model.py`  신경망 하나(CNN + 트랜스포머)가 모든 말 지휘, 가치 출력 -1~+1
- `warsim/rl/deploy.py`  배치 유형 5가지 + 밴딧(톰슨 샘플링)
- `warsim/rl/train.py`  PPO 셀프플레이, 챔피언 저장(모든 챔피언에게 승률 50% 초과)
- `warsim/viewer.html` 재생 화면 (브라우저에서 열림, 인터넷 불필요)

## 재생 화면 조작
- 스페이스: 재생/정지, ←/→: 한 턴씩
- 1배/2배/3배: 지도 확대
- 전체 / 청 시야 / 홍 시야: 안개 적용 화면 (그 진영이 실제로 아는 것만 표시)
- 칸을 누르면 그 칸 부대의 전투력·식량·탄약·연료가 오른쪽에 나옴
