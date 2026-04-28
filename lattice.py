import numpy as np
import numba


@numba.njit
def getDistances(coords, PBC, sys_sizes):
    n_local = coords.shape[0]
    distances = np.zeros((n_local, n_local, 3))
    for i in range(n_local):
        for j in range(n_local):
            dx = coords[j, 0] - coords[i, 0]
            dy = coords[j, 1] - coords[i, 1]
            if PBC:
                if dx > sys_sizes[0] / 2:
                    dx -= sys_sizes[0]
                elif dx < -sys_sizes[0] / 2:
                    dx += sys_sizes[0]
                if dy > sys_sizes[1] / 2:
                    dy -= sys_sizes[1]
                elif dy < -sys_sizes[1] / 2:
                    dy += sys_sizes[1]
            distances[i, j, 0] = dx
            distances[i, j, 1] = dy
    return distances

@numba.njit
def getDistancesP(p, coords, PBC, sys_sizes):
    '''Находит расстояние между точкой p и остальными точками coords.
    
    Параметры:
    ----------
    p : ndarray, shape (2,)
        Координаты точки-источника [x, y].
    coords : ndarray, shape (n, 2)
        Массив координат точек, до которых вычисляются расстояния.
    PBC : bool
        Флаг периодических граничных условий.
    sys_sizes : ndarray, shape (2,)
        Размеры системы [Lx, Ly].
    
    Возвращает:
    -----------
    distances : ndarray, shape (n, 2)
        Массив векторов расстояний [dx, dy] от точки p до каждой точки в `coords`.
        При включённых PBC применяется правило минимального образа.
    '''
    n = coords.shape[0]
    distances = np.zeros((n, 2))
    
    for j in range(n):
        dx = coords[j, 0] - p[0]
        dy = coords[j, 1] - p[1]
        
        if PBC:
            # Применение правила минимального образа по оси X
            if dx > sys_sizes[0]/2:
                dx -= sys_sizes[0]
            elif dx < -sys_sizes[0]/2 :
                dx += sys_sizes[0]
            
            # Применение правила минимального образа по оси Y
            if dy > sys_sizes[1]/2:
                dy -= sys_sizes[1]
            elif dy < -sys_sizes[1]/2:
                dy += sys_sizes[1]
        
        distances[j, 0] = dx
        distances[j, 1] = dy
    
    return distances

@numba.njit
def E_sys(coords, magnetic_moments, sys_sizes, PBC, r_=0.0):
    n_local = coords.shape[0]
    E = np.zeros((n_local, n_local))
    k = 1.
    r = getDistances(coords, PBC, sys_sizes)
    for i in range(n_local):
        for j in range(i+1, n_local):
            r_ij = r[i, j]
            m_i = magnetic_moments[i]
            m_j = magnetic_moments[j]
            r_ij_mod = np.sqrt((r_ij**2).sum())
            if (r_ == 0.0) or (r_ij_mod < r_):
                A = np.dot(m_i, m_j)/(r_ij_mod**3)
                B = np.dot(m_i, r_ij) * np.dot(m_j, r_ij) / (r_ij_mod**5)
                E[j, i] = E[i, j] = (A - 3*B) * k
    return E


class Vertex:
    def __init__(self, coords, parts_array, lattice):
        self.parts_in_v = parts_array
        self.n_parts = len(parts_array)
        self.coords = coords
        self.lattice = lattice
        self.types = lattice.VERTEX_TYPES.get(self.n_parts, None)

    def calc_energy(self):
        return self.lattice.pairwise_E[self.parts_in_v, :][:, self.parts_in_v].sum() / 2
    
    def get_type(self):
        if self.types is None:
            raise ValueError(f"Unknown vertex type with {self.n_parts} parts")
        energy = np.round(self.calc_energy(), 4)
        vertex_type = self.types.get(energy, None)
        if vertex_type is None:
            raise ValueError(f"Vertex type not found for energy {energy}")
        return vertex_type

    def __repr__(self):
        return f"Vertex({self.parts_in_v})"


class Lattice:
    def __init__(self):
        self.coords = np.empty((0, 3), int)
        self.magnetic_moments = np.empty((0, 3), int)
        self._base_magnetic_moments = np.empty((0, 3), int)
        self.sizes = None
        self.state = None
        self.N = None
        self.PBC = True
        self.VERTEX_TYPES = None

    def set_PBC(self, PBC):
        if isinstance(PBC, (bool, np.bool_)):
            self.PBC = bool(PBC)
            self.pairwise_r = getDistances(self.coords, self.PBC, self.sizes)
            self.recompute_energy()
        elif isinstance(PBC, (int, np.integer)) and PBC in (0, 1):
            self.PBC = bool(PBC)
            self.pairwise_r = getDistances(self.coords, self.PBC, self.sizes)
            self.recompute_energy()
        else:
            raise ValueError("PBC must be a boolean (True/False) or 0/1")
        return self
    
    def set_r(self, r):
        self.r = r
        self.pairwise_r = getDistances(self.coords, self.PBC, self.sizes)
        self.recompute_energy()

    def set_state(self, state):
        if state.shape != (self.N,):
            raise ValueError(f"state must have shape ({self.N},)")
        div = self.state * state
        trans_matrix = div[:, None] @ div[None, :]
        self.state = state.copy()
        self.magnetic_moments = self._base_magnetic_moments * self.state[:, None]
        self.pairwise_E *= trans_matrix
        self.E = self.pairwise_E.sum()/2


    def recompute_energy(self):
        self.pairwise_E = E_sys(self.coords, self.magnetic_moments, self.sizes, self.PBC, self.r)
        self.E = self.pairwise_E.sum()/2
    

    def swap(self, spin_index):
        self.state[spin_index] *= -1
        self.magnetic_moments[spin_index] *= -1
        self.pairwise_E[spin_index, :] *= -1
        self.pairwise_E[:, spin_index] *= -1
        self.E = self.pairwise_E.sum()/2

    def brute_force(self, energies, states):
        total_states = 1 << self.N
        if total_states > 1e7:
            raise ValueError("Too many states for brute force")
        prev_gray = 0
        states[0] = self.state.copy()
        energies[0] = self.E
        for i in range(1, total_states):
            gray = i ^ (i >> 1)
            diff = gray ^ prev_gray
            mask = diff & -diff
            bit = 0
            while mask > 1:
                mask >>= 1
                bit += 1
            self.swap(bit)
            prev_gray = gray
            states[i] = self.state
            energies[i] = self.E

    def __str__(self):
        return f"Lattice(N={self.N}, PBC={self.PBC})"
    
    def __repr__(self):
        return f"Lattice(N={self.N}, PBC={self.PBC})"

class ApameaLattice(Lattice):
    def __init__(self, n, m):
        super().__init__()
        self.VERTEX_TYPES = dict([(4, dict([(20.9706, "IV"), (-4.0, "II"), (0.0, "III"), (-12.9706, "I")])),
                (3, dict([(10.4853, "C"), (-6.4853, "A"), (-2.0, "B")])),
                (2, dict([(4.2426, r"\beta"), (2.0, "b"), (-2.0, "a"), (-4.2426, r"\alpha")]))])
        self.make_lattice(n, m)

    def make_lattice(self, n, m):
        coords_blocks = []
        moment_blocks = []
        coords = np.array([
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [2.0, 0.5, 0.0],
                [3.5, 0.0, 0.0],
                [3.0, 0.5, 0.0],
                [0.5, 1.0, 0.0],
                [1.5, 1.0, 0.0],
                [1.0, 1.5, 0.0],
                [2.5, 1.0, 0.0],
                [3.0, 1.5, 0.0],
                [0.5, 2.0, 0.0],
                [1.0, 2.5, 0.0],
                [3.5, 2.0, 0.0],
                [3.0, 2.5, 0.0],
                [0.0, 3.5, 0.0],
                [1.5, 3.0, 0.0],
                [1.0, 3.5, 0.0],
                [2.5, 3.0, 0.0],
                [2.0, 3.5, 0.0],
                [3.5, 3.0, 0.0]], dtype=np.float64)
    
        magnetic_moments_even = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0]], dtype=np.float64)

        magnetic_moments_odd = np.array([
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0]], dtype=np.float64)

        for i in range(n):
            for j in range(m):
                if (i+j)%2 == 1:
                    coords_blocks.append(coords + np.array([4 * i, 4 * j, 0.0]))
                    moment_blocks.append(magnetic_moments_even)
                else:
                    coords_blocks.append(coords + np.array([4 * i, 4 * j, 0.0]))
                    moment_blocks.append(magnetic_moments_odd)
        self.coords = np.vstack(coords_blocks)
        self.magnetic_moments = np.vstack(moment_blocks)
        # self.magnetic_moments *= np.dot(self.magnetic_moments, np.array([1, 1, 0]))[:, None]
        self._base_magnetic_moments = self.magnetic_moments.copy()
        self.sizes = np.array([4*n, 4*m, 0])
        self.N = self.coords.shape[0]
        self.PBC = True
        self.r = 0.0
        self.state = np.ones(self.N, int)
        self.recompute_energy()
        self.vertexes = self.calc_vertexes()

    def __str__(self):
        return f"ApameaLattice(N={self.N}, PBC={self.PBC})"
    
    def __repr__(self):
        return f"ApameaLattice(N={self.N}, PBC={self.PBC})"

    def calc_vertexes(self) -> list:
        vertexes = np.array((self.coords + self.magnetic_moments/2, self.coords - self.magnetic_moments/2)).reshape(-1, 3)
        vertexes[vertexes[:, 0] >= self.sizes[0], 0] -= self.sizes[0]
        vertexes[vertexes[:, 1] >= self.sizes[1], 1] -= self.sizes[1]
        vertexes[vertexes[:, 2] >= self.sizes[2], 2] -= self.sizes[2]
        vertexes = np.unique(vertexes, axis=0)
        tmp = list()


        for i, v in enumerate(vertexes):
            r = getDistancesP(v, self.coords, self.PBC, self.sizes)
            parts_in_v = np.where((r**2).sum(axis=1) < 1.1)[0]
            tmp.append(Vertex(v, parts_in_v, self))

        return tmp
    
    def calc_vertexes_stats(self):
        a = np.array([v.get_type() for v in self.vertexes])
        types, counts = np.unique(a, return_counts=True)
        return dict(zip(types, counts))

class CyrrhusLattice(Lattice):
    def __init__(self, n, m):
        super().__init__()
        self.VERTEX_TYPES = dict([(4, dict([(20.9706, "IV"), (-4.0, "II"), (0.0, "III"), (-12.9706, "I")])),
                (3, dict([(10.4853, "C"), (-6.4853, "A"), (-2.0, "B")])),
                (2, dict([(4.2426, r"\beta"), (2.0, "b"), (-2.0, "a"), (-4.2426, r"\alpha")]))])
        self.make_lattice(n, m)

    def make_lattice(self, n, m):
        x = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8], dtype=np.float64) / 2
        y = np.array([1, 2, 4, 8, 3, 5, 7, 2, 4, 6, 1, 3, 5, 7, 2, 4, 6, 1, 3, 5, 0, 4, 6, 7], dtype=np.float64) / 2
        mx = np.array([0, 1, -1, -1, 0, 0, 0, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, 0, 0, 0, 1, 1, -1, 0], dtype=np.float64)
        my = np.array([1, 0, 0, 0, 1, -1, 1, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, -1, 1, -1, 0, 0, 0, -1], dtype=np.float64)
        mask = np.array([1., 1., -1., 1., -1., -1., 1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., 1., -1., -1., 1., -1., 1., 1.], dtype=np.float64)

        coords_blocks = []
        moment_blocks = []

        for i in range(n):
            for j in range(m):
                coords_blocks.append(
                    np.column_stack((
                        x + 4 * i,
                        y + 4 * j,
                        np.zeros_like(x),
                    ))
                )

                tile_mask = mask if (i + j) % 2 == 1 else 1.0
                moment_blocks.append(
                    np.column_stack((
                        mx * tile_mask,
                        my * tile_mask,
                        np.zeros_like(mx),
                    ))
                )

        self.coords = np.vstack(coords_blocks)
        self.magnetic_moments = np.vstack(moment_blocks)
        self._base_magnetic_moments = self.magnetic_moments.copy()
        self.sizes = np.array([4 * n, 4 * m, 0], dtype=np.float64)
        self.N = self.coords.shape[0]
        self.PBC = True
        self.r = 0.0
        self.recompute_energy()
        self.state = np.ones(self.N, int)
        self.vertexes = self.calc_vertexes_types()

    def __str__(self):
        return f"CyrrhusLattice(N={self.N}, PBC={self.PBC})"

    def __repr__(self):
        return f"CyrrhusLattice(N={self.N}, PBC={self.PBC})"

    def calc_vertexes_types(self) -> list:
        vertexes = np.array((self.coords + self.magnetic_moments/2, self.coords - self.magnetic_moments/2)).reshape(-1, 3)
        vertexes[vertexes[:, 0] >= self.sizes[0], 0] -= self.sizes[0]
        vertexes[vertexes[:, 1] >= self.sizes[1], 1] -= self.sizes[1]
        vertexes[vertexes[:, 2] >= self.sizes[2], 2] -= self.sizes[2]
        vertexes = np.unique(vertexes, axis=0)
        tmp = list()


        for i, v in enumerate(vertexes):
            r = getDistancesP(v, self.coords, self.PBC, self.sizes)
            parts_in_v = np.where((r**2).sum(axis=1) < 1.1)[0]
            tmp.append(Vertex(v, parts_in_v, self))

        return tmp
    
    def calc_vertexes_stats(self):
        a = np.array([v.get_type() for v in self.vertexes])
        types, counts = np.unique(a, return_counts=True)
        return dict(zip(types, counts))

if __name__ == "__main__":
    l = CyrrhusLattice(10, 10)
    l.set_r(5.1)
    l.set_state(np.random.randint(0, 2, l.N)*2-1)
    a = l.calc_vertexes_stats()
    print(a)