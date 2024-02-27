import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Correction ici

# Création d'un tableau de 100 points entre -4*pi et 4*pi
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)

# Création du tableau de l'axe z entre -2 et 2
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

# Tracé du résultat en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Correction ici
ax.plot(x, y, z, label='Lignes de champ')  # Correction ici
plt.title("Lignes de champ 3D")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.tight_layout()
plt.show()
