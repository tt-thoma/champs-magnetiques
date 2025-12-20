class Particle:
    def __init__(self, x, y, z, charge, mass, fil=False, solen=False, vx=0.0, vy=0.0, vz=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.charge = float(charge)
        self.mass = float(mass)
        self.fil = bool(fil)
        self.solen = bool(solen)
        self.vx = float(vx)
        self.vy = float(vy)
        self.vz = float(vz)

    def calc_next(self, field_E, field_B, size, dt, cell_size, fil=None):
        # Minimal position update: move according to current velocity.
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        # Keep particle inside the box (simple clamp)
        if self.x < 0:
            self.x = 0.0
        if self.y < 0:
            self.y = 0.0
        if self.z < 0:
            self.z = 0.0
        if self.x > size:
            self.x = size
        if self.y > size:
            self.y = size
        if self.z > size:
            self.z = size
