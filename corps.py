def calc_next(self, world_E, world_B, size, dt, c, fil):

        if not self.fil:
            ex, ey, ez = world_E[int(self.x / c), int(self.y / c), int(self.z / c), :]
            bx, by, bz = world_B[int(self.x / c), int(self.y / c), int(self.z / c), :]
        else:
            ex, ey, ez = fil[int(self.x / c), int(self.y / c), int(self.z / c), :]
            bx, by, bz = 0, 0, 0

        
        
        
        print_debug(f"Coordonnées: {self.x=}; {self.y=}; {self.z=}")
        print_debug(
            f"CoordonnéesD: {int(self.x/c)=}; {int(self.y/c)=}; {int(self.z/c)=}"
        )
        print_debug(f"Multiplicateur: {c}")

        print_debug(f"Champ_e: {ex=}; {ey=}; {ez=}")

        print_debug(f"Champ_b: {bx=}; {by=}; {bz=}")

        Fx = self.charge * (ex + bx * self.vx)
        Fy = self.charge * (ey + by * self.vy)
        Fz = self.charge * (ez + bz * self.vz)

        self.ax += Fx / self.mass
        self.ay += Fy / self.mass
        self.az += Fz / self.mass

        # Vérifier si les valeurs sont NaN
        if np.isnan(self.ax) or np.isnan(self.ay) or np.isnan(self.az):
            print_debug(f"Une des accélérations calculées est NaN. {Fx=}")

        # Étape 1: Calcul des coefficients RK4
        k1x = self.charge * (ex + bx * self.vx) / self.mass
        k1y = self.charge * (ey + by * self.vy) / self.mass
        k1z = self.charge * (ez + bz * self.vz) / self.mass

        k2x = self.charge * (ex + bx * (self.vx + 0.5 * dt * k1x)) / self.mass
        k2y = self.charge * (ey + by * (self.vy + 0.5 * dt * k1y)) / self.mass
        k2z = self.charge * (ez + bz * (self.vz + 0.5 * dt * k1z)) / self.mass

        k3x = self.charge * (ex + bx * (self.vx + 0.5 * dt * k2x)) / self.mass
        k3y = self.charge * (ey + by * (self.vy + 0.5 * dt * k2y)) / self.mass
        k3z = self.charge * (ez + bz * (self.vz + 0.5 * dt * k2z)) / self.mass

        k4x = self.charge * (ex + bx * (self.vx + dt * k3x)) / self.mass
        k4y = self.charge * (ey + by * (self.vy + dt * k3y)) / self.mass
        k4z = self.charge * (ez + bz * (self.vz + dt * k3z)) / self.mass

        # Étape 2: Calcul des valeurs intermédiaires
        vx_new = self.vx + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        vy_new = self.vy + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
        vz_new = self.vz + (dt / 6.0) * (k1z + 2 * k2z + 2 * k3z + k4z)
        print_debug(f"{vx_new=}")

        # Étape 3: Calcul final
        self.x += dt * vx_new
        self.y += dt * vy_new
        self.z += dt * vz_new

        if np.isnan(vx_new) or np.isnan(vy_new) or np.isnan(vz_new):
            print_debug(f"Une des nouvelles vitesses calculées est NaN. Fx = {Fx}")

        if self.x <= 0:
            self.x = np.float64(size - c)
            vx_new = vx_new
        elif self.x >= size - c:
            self.x = np.float64(0)
            vx_new = vx_new

        if self.y <= 0:
            self.y = np.float64(size - c)
            vy_new = vy_new
        elif self.y >= size - c:
            self.y = np.float64(0)
            vy_new = vy_new

        if self.z <= 0:
            self.z = np.float64(0)
            vz_new = vz_new
        elif self.z >= size - c:
            self.z = np.float64(size - c)
            vz_new = vz_new

        self.vx = vx_new
        self.vy = vy_new
        self.vz = vz_new

        if self.fil:
            if self.vx > 0.001:
                self.vx = np.float64(0.001)
            if self.vy > 0.001:
                self.vy = np.float64(0.001)
            if self.vz > 0.001:
                self.vz = np.float64(0.001)

        print_debug(
            f"fX = {Fx} ex = {ex}, bx {bx}, bx * self.vx {bx * self.vx}, self.vx = {self.vx} ax = {self.ax}"
        )

        print_debug(f"Coordonnées: {self.x=}; {self.y=}; {self.z=}")
        print_debug(
            f"CoordonnéesD: {int(self.x/c)=}; {int(self.y/c)=}; {int(self.z/c)=}"
        )
        print_debug(f"Multiplicateur: {c}")
