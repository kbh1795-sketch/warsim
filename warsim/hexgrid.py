"""육각형 격자 (odd-r 오프셋 좌표).

칸은 (col, row). row 0 이 진영 0 의 뒤쪽 끝, row length-1 이 진영 1 의 뒤쪽 끝.
거리 계산은 큐브 좌표로 바꿔서 한다.
"""

import heapq

import numpy as np

FLAT, MOUNTAIN, RIVER, CROSSING, MINE = 0, 1, 2, 3, 4
# RIVER: 지나갈 수 없음. CROSSING: 강 위 도하 지점(평지처럼 지나감).
# MINE: 지뢰지대. 들어가면 그 칸에서 멈추고 피해를 입는다.

_CUBE_DIRS = [(1, -1, 0), (1, 0, -1), (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1)]


_CUBE_CACHE = {}


def to_cube(col, row):
    key = (col, row)
    c = _CUBE_CACHE.get(key)
    if c is None:
        x = col - (row - (row & 1)) // 2
        c = (x, -x - row, row)
        _CUBE_CACHE[key] = c
    return c


def to_offset(x, y, z):
    col = x + (z - (z & 1)) // 2
    return col, z


def distance(a, b):
    ax, ay, az = to_cube(*a)
    bx, by, bz = to_cube(*b)
    return max(abs(ax - bx), abs(ay - by), abs(az - bz))


def _cube_round(x, y, z):
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return rx, ry, rz


def line(a, b):
    """a 에서 b 까지 직선이 지나는 칸들 (양 끝 포함)."""
    n = distance(a, b)
    if n == 0:
        return [a]
    ax, ay, az = to_cube(*a)
    bx, by, bz = to_cube(*b)
    e = 1e-6  # 경계에 걸릴 때 한쪽으로 정하기 위한 작은 값
    out = []
    for i in range(n + 1):
        t = i / n
        c = _cube_round(ax + (bx - ax) * t + e, ay + (by - ay) * t + 2 * e, az + (bz - az) * t - 3 * e)
        out.append(to_offset(*c))
    return out


class HexMap:
    def __init__(self, width, length, terrain):
        self.width = width
        self.length = length
        self.terrain = terrain  # shape (length, width), FLAT / MOUNTAIN / RIVER / CROSSING / MINE
        self._reach_cache = {}
        self._los_cache = {}
        self._nbr = {}
        for r in range(length):
            for c in range(width):
                x, y, z = to_cube(c, r)
                out = []
                for dx, dy, dz in _CUBE_DIRS:
                    nc, nr = to_offset(x + dx, y + dy, z + dz)
                    if 0 <= nc < width and 0 <= nr < length:
                        out.append((nc, nr))
                self._nbr[(c, r)] = out

    @staticmethod
    def _sprinkle(t, tmp, cells, kind, target, rng):
        """kind 를 덩어리로 cells 안에 target 칸 뿌린다 (지도 앞쪽 절반만; 뒤쪽은 점대칭)."""
        count = 0
        order = sorted(cells)
        while count < target:
            c, r = order[rng.integers(len(order))]
            frontier = [(c, r)]
            for _ in range(rng.integers(4, 12)):
                if not frontier or count >= target:
                    break
                cc, rr = frontier.pop(rng.integers(len(frontier)))
                if (cc, rr) not in cells or t[rr, cc] != FLAT:
                    continue
                t[rr, cc] = kind
                count += 1
                frontier.extend(tmp.neighbors((cc, rr)))

    @classmethod
    def generate(cls, width, length, mountain_fraction, rng, river_crossings=0,
                 mine_fraction=0.0, deploy_depth=0):
        """산지(·강·지뢰지대)를 만들고, 양측 공평하도록 지도 중심 기준 점대칭으로 만든다.

        강: 지도 한가운데 두 줄(가로로 지도 전체), 도하 지점 river_crossings 쌍(점대칭이라 2배).
        지뢰지대: 양측 배치 구역 밖, 강 앞뒤 지역에 mine_fraction 만큼 덩어리로.
        """
        t = np.zeros((length, width), dtype=np.int8)
        target = int(mountain_fraction * width * length / 2)
        half = [(c, r) for r in range((length + 1) // 2) for c in range(width)]
        count = 0
        tmp = cls(width, length, t)
        while count < target:
            c, r = half[rng.integers(len(half))]
            frontier = [(c, r)]
            for _ in range(rng.integers(4, 12)):
                if not frontier or count >= target:
                    break
                cc, rr = frontier.pop(rng.integers(len(frontier)))
                if rr >= (length + 1) // 2 or t[rr, cc] == MOUNTAIN:
                    continue
                t[rr, cc] = MOUNTAIN
                count += 1
                frontier.extend(tmp.neighbors((cc, rr)))
        mid = length // 2
        if river_crossings > 0:
            t[mid - 1, :] = RIVER                 # 두 줄은 서로 점대칭 짝
            t[mid, :] = RIVER
            cols = rng.choice(width, size=river_crossings, replace=False)
            for c in cols:
                t[mid - 1, c] = t[mid, c] = CROSSING
                t[mid - 1, width - 1 - c] = t[mid, width - 1 - c] = CROSSING
        if mine_fraction > 0:
            band = {(c, r) for r in range(deploy_depth, mid - 1) for c in range(width)
                    if t[r, c] == FLAT}
            cls._sprinkle(t, tmp, band, MINE, int(mine_fraction * len(band)), rng)
        front = mid - 1 if river_crossings > 0 else (length + 1) // 2
        for r in range(front):
            for c in range(width):
                if t[r, c] in (MOUNTAIN, MINE):
                    t[length - 1 - r, width - 1 - c] = t[r, c]
        return cls(width, length, t)

    def in_bounds(self, h):
        return 0 <= h[0] < self.width and 0 <= h[1] < self.length

    def neighbors(self, h):
        return self._nbr[h]

    def is_mountain(self, h):
        return self.terrain[h[1], h[0]] == MOUNTAIN

    def is_mine(self, h):
        return self.terrain[h[1], h[0]] == MINE

    def passable(self, h):
        return self.terrain[h[1], h[0]] != RIVER

    def line_of_sight(self, a, b):
        """통신 가시선. 사이 칸에 산지가 있으면 막힘. 단 양 끝 중 하나가 산지(고지)면 통함."""
        if self.is_mountain(a) or self.is_mountain(b):
            return True
        key = (a, b) if a <= b else (b, a)
        v = self._los_cache.get(key)
        if v is None:
            v = not any(self.is_mountain(h) for h in line(a, b)[1:-1])
            self._los_cache[key] = v
        return v

    def within(self, h, radius):
        """h 에서 radius 칸 이내의 모든 칸."""
        x, y, z = to_cube(*h)
        out = []
        for dx in range(-radius, radius + 1):
            for dy in range(max(-radius, -dx - radius), min(radius, -dx + radius) + 1):
                dz = -dx - dy
                c, r = to_offset(x + dx, y + dy, z + dz)
                if 0 <= c < self.width and 0 <= r < self.length:
                    out.append((c, r))
        return out

    def reachable(self, start, mp, mountain_cost):
        """이동력 mp 안에서 갈 수 있는 칸과 비용. 평지 1, 산지 mountain_cost.
        강은 지나갈 수 없고(도하 지점만), 지뢰지대에 들어가면 거기서 멈춘다.

        다른 말이 이동을 막지 않으므로 결과는 지형에만 달려 있어 한 번 계산하면 재사용한다.
        지형을 바꾸면 clear_cache() 를 불러야 한다.
        """
        key = (start, mp, mountain_cost)
        hit = self._reach_cache.get(key)
        if hit is not None:
            return hit
        terrain = self.terrain
        best = {start: 0.0}
        heap = [(0.0, start)]
        while heap:
            d, h = heapq.heappop(heap)
            if d > best[h]:
                continue
            if h != start and terrain[h[1], h[0]] == MINE:
                continue  # 지뢰지대에서는 더 나아가지 못함
            for n in self._nbr[h]:
                tn = terrain[n[1], n[0]]
                if tn == RIVER:
                    continue
                nd = d + (mountain_cost if tn == MOUNTAIN else 1.0)
                if nd <= mp + 1e-9 and nd < best.get(n, 1e18):
                    best[n] = nd
                    heapq.heappush(heap, (nd, n))
        self._reach_cache[key] = best
        return best

    def clear_cache(self):
        self._reach_cache.clear()
