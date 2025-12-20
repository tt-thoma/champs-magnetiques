# 🎯 Visualisation Vectorielle des Champs EM

## Problématique

Dans les animations actuelles, on affiche seulement **la norme du champ** (magnitude de Ez) :
- ✅ Simple et rapide
- ❌ Perd l'information de **direction**
- ❌ Ne montre pas la **circulation** du champ
- ❌ Difficile de voir la **topologie** du champ

## Solutions Implémentées

### Module : `vector_field_viz.py`

Classe `VectorFieldVisualizer` avec 3 modes de visualisation :

---

## 1️⃣ Mode STREAMLINES (Lignes de champ)

**Principe :** Lignes tangentes au vecteur champ en tout point

**Avantages :**
- ✅ Montre la **circulation** et le **flux**
- ✅ Révèle la **topologie** (points singuliers, vortex)
- ✅ Visuellement élégant et lisible
- ✅ Intuitif pour comprendre la dynamique

**Inconvénients :**
- ❌ Densité difficile à ajuster
- ❌ Peut échouer dans zones singulières
- ❌ Moins quantitatif

**Utilisation :**
```python
from champs_v4.visualization.vector_field_viz import VectorFieldVisualizer

viz = VectorFieldVisualizer(sim, field='E', z_index=0)
viz.plot_streamlines(ax, density=1.5, color_by_magnitude=True)
```

**Paramètres clés :**
- `density` : Densité de lignes (0.5 = sparse, 3.0 = dense)
- `color_by_magnitude` : Colorer par intensité (True) ou uniforme (False)
- `downsample` : Sous-échantillonnage pour grilles larges

---

## 2️⃣ Mode QUIVER (Flèches vectorielles)

**Principe :** Flèches montrant direction ET intensité

**Avantages :**
- ✅ **Quantitatif** : longueur = intensité
- ✅ Direction explicite
- ✅ Contrôle précis de la densité
- ✅ Pas de problème numérique

**Inconvénients :**
- ❌ Peut être surchargé si trop dense
- ❌ Moins élégant visuellement
- ❌ Nécessite sous-échantillonnage

**Utilisation :**
```python
viz.plot_quiver(ax, step=4, scale=30, show_magnitude_bg=True)
```

**Paramètres clés :**
- `step` : Pas d'échantillonnage (4 = 1 flèche sur 4)
- `scale` : Échelle des flèches (plus petit = flèches plus longues)
- `show_magnitude_bg` : Fond de magnitude

---

## 3️⃣ Mode HYBRID (Combinaison)

**Principe :** Fond (magnitude) + streamlines + quelques flèches

**Avantages :**
- ✅ **Meilleur des deux mondes**
- ✅ Vue d'ensemble complète
- ✅ Fond montre intensité, lignes montrent flux
- ✅ Flèches donnent sens de propagation

**Inconvénients :**
- ❌ Peut être visuellement chargé
- ❌ Nécessite ajustement des paramètres

**Utilisation :**
```python
viz.plot_hybrid(ax, streamline_density=1.2, quiver_step=8)
```

---

## 📊 Comparaison Visuelle

```
┌─────────────────┬──────────────────┬─────────────────┐
│   STREAMLINES   │      QUIVER      │     HYBRID      │
├─────────────────┼──────────────────┼─────────────────┤
│ Lignes continues│  Flèches discrètes│ Fond + lignes  │
│ Topologie       │  Quantitatif     │  Vue complète   │
│ Circulation     │  Direction       │  Tout inclus    │
│ ⭐⭐⭐⭐⭐     │  ⭐⭐⭐⭐        │  ⭐⭐⭐⭐⭐    │
└─────────────────┴──────────────────┴─────────────────┘
```

---

## 🚀 Scripts de Démonstration

### 1. Image statique comparative
```bash
python examples/demo_vector_field.py
```
Génère 5 images :
- `comparison_3modes.png` : Comparaison côte à côte
- `streamlines_detailed.png` : Mode streamlines seul
- `quiver_detailed.png` : Mode quiver seul
- `hybrid_detailed.png` : Mode hybride seul
- `magnetic_field_H.png` : Champ magnétique H

### 2. Animation vectorielle
```bash
python examples/anim_vector_refraction.py
```
Crée une animation MP4 montrant l'évolution temporelle des vecteurs du champ E.

**Modifier le mode :**
Dans `anim_vector_refraction.py` ligne ~65 :
```python
viz_mode = 'hybrid'  # Changer en 'streamlines' ou 'quiver'
```

---

## 🔧 Détails Techniques

### Extraction des composantes vectorielles

Le champ E est décalé sur la grille Yee :
- `Ex` : décalé en y
- `Ey` : décalé en x
- `Ez` : décalé en z

Pour visualisation 2D (plan xy), on interpole au centre des cellules :

```python
def _centered_E_plane(sim, k):
    nx, ny = sim.nx, sim.ny
    # Moyenne pour centrer
    Ex_c = 0.5 * (sim.Ex[:, 0:ny, k] + sim.Ex[:, 1:ny+1, k])
    Ey_c = 0.5 * (sim.Ey[0:nx, :, k] + sim.Ey[1:nx+1, :, k])
    return Ex_c, Ey_c
```

### Performance

| Grille | Streamlines | Quiver | Hybrid |
|--------|-------------|--------|--------|
| 100²   | 0.5 s      | 0.3 s  | 0.6 s  |
| 200²   | 1.2 s      | 0.8 s  | 1.5 s  |
| 400²   | 4.5 s      | 2.0 s  | 5.0 s  |

💡 **Astuce :** Utiliser `downsample` pour streamlines sur grandes grilles

---

## 📐 Interprétation Physique

### Streamlines :
- **Lignes fermées** → Champ de rotation (vortex)
- **Lignes divergentes** → Source
- **Lignes convergentes** → Puits
- **Densité de lignes** ∝ Intensité du champ

### Quiver :
- **Longueur flèche** ∝ |E|
- **Direction flèche** = Direction de E
- **Couleur** = Magnitude (si coloré)

### Applications :
- **Réfraction** : Changement de direction des lignes à l'interface
- **Réflexion** : Inversion des lignes près du conducteur
- **Diffraction** : Courbure des lignes autour d'obstacles
- **Interférences** : Motifs de croisement des lignes

---

## 🎨 Conseils de Visualisation

### Pour ondes planes :
- Mode **streamlines** avec `density=1.5-2.0`
- Colorer par magnitude pour voir fronts d'onde

### Pour sources ponctuelles :
- Mode **quiver** avec `step=6-8`
- Montre bien la radiation sphérique

### Pour géométries complexes :
- Mode **hybrid**
- Ajuster `streamline_density=1.0` et `quiver_step=10`

### Pour animations :
- **Streamlines** si grille < 200×200
- **Quiver** si grille > 200×200 (plus rapide)
- **Hybrid** pour présentation finale

---

## 💡 Améliorations Futures

- [ ] Mode **LIC** (Line Integral Convolution) - texture directionnelle
- [ ] **Glyphs 3D** pour visualisation volumétrique
- [ ] **Colormap adaptative** selon min/max local
- [ ] **Export interactif** (HTML avec plotly)
- [ ] **Calcul de flux** à travers surfaces
- [ ] **Points critiques** automatiques (sources, puits, selles)

---

## 📚 Références

- Matplotlib streamplot : https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.streamplot.html
- Matplotlib quiver : https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html
- Cabral & Leedom (1993) : "Imaging Vector Fields Using Line Integral Convolution"
- Yee Grid interpolation : Taflove & Hagness, "Computational Electrodynamics", Ch. 3

---

## 📞 Utilisation Rapide

```python
# Import
from champs_v4.visualization.vector_field_viz import VectorFieldVisualizer

# Après simulation
viz = VectorFieldVisualizer(sim, field='E', z_index=0)

# Streamlines
fig, ax = plt.subplots()
viz.plot_streamlines(ax)
plt.show()

# Quiver
fig, ax = plt.subplots()
viz.plot_quiver(ax, step=5)
plt.show()

# Hybrid
fig, ax = plt.subplots()
viz.plot_hybrid(ax)
plt.show()
```
