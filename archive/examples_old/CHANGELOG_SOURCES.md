# Mise à jour : Sources Impulsionnelles pour toutes les Animations

## 🔄 Changements appliqués

Toutes les animations ont été mises à jour pour utiliser des **impulsions gaussiennes** au lieu de sources sinusoïdales continues.

---

## 📝 Fichiers modifiés

### Animations originales (magnitude scalaire)

1. ✅ **anim_01_dielectric_refraction.py** - Réfraction air/verre
2. ✅ **anim_02_metal_reflection.py** - Réflexion sur métal
3. ✅ **anim_03_lossy_medium.py** - Atténuation
4. ✅ **anim_04_dielectric_cavity.py** - Cavité résonante
5. ✅ **anim_05_layered_materials.py** - Multicouche

### Animations vectorielles (déjà créées avec impulsions)

1. ✅ **anim_01_vector_refraction.py**
2. ✅ **anim_02_vector_metal.py**
3. ✅ **anim_03_vector_lossy.py**
4. ✅ **anim_04_vector_cavity.py**
5. ✅ **anim_05_vector_multilayer.py**

---

## 🔬 Différences techniques

### AVANT (Source continue)
```python
# Ancienne méthode - enveloppe large
t0 = 100 * dt
width = 40 * dt

envelope = np.exp(-0.5 * ((t - t0) / width) ** 2)
source_value = envelope * np.sin(omega * t)
```

**Problèmes** :
- ❌ Plusieurs cycles d'oscillation
- ❌ Onde étendue dans l'espace
- ❌ Difficile de distinguer réflexion/transmission
- ❌ Interférences complexes masquent les phénomènes

### APRÈS (Impulsion gaussienne)
```python
# Nouvelle méthode - paquet compact
t0 = 80 * dt
spread = 25 * dt

pulse = np.exp(-((t - t0) / spread)**2) * np.sin(omega * t)
source_value = pulse
```

**Avantages** :
- ✅ Paquet d'onde compact et localisé
- ✅ Propagation CLAIRE et VISIBLE
- ✅ Réflexion/réfraction bien distinctes
- ✅ Meilleure pédagogie

---

## 📊 Paramètres mis à jour

| Animation | Fréquence (avant) | Fréquence (après) | Raison |
|-----------|-------------------|-------------------|---------|
| **1. Réfraction** | 10 GHz | 8 GHz | Meilleure visibilité |
| **2. Métal** | 5 GHz | 5 GHz | *(inchangée)* |
| **3. Atténuation** | 2 GHz | 6 GHz | Atténuation plus visible |
| **4. Cavité** | ~calculée~ | ~calculée~ | *(inchangée)* |
| **5. Multicouche** | 15 GHz | 10 GHz | Réduction d'interférences |

---

## 🎯 Paramètres d'impulsion

### Configuration typique
```python
t0 = 80 * dt       # Centre de l'impulsion (40-120 * dt)
spread = 25 * dt   # Largeur (15-40 * dt selon durée souhaitée)

# Formule impulsion
pulse = np.exp(-((t - t0) / spread)**2) * np.sin(omega * t)
```

### Règles de dimensionnement

**Centre d'impulsion (t0)** :
- Doit laisser le temps à l'onde de se former
- Typique : 40-120 pas de temps
- Plus grand = démarrage plus tardif

**Largeur (spread)** :
- Contrôle la durée de l'impulsion
- Plus petit = impulsion plus courte
- Plus grand = paquet plus étalé
- Typique : 15-40 pas de temps

**Fréquence (freq)** :
- Détermine la longueur d'onde
- Doit être adaptée à la taille de la grille
- λ = c / freq
- Besoin de plusieurs cellules par λ

---

## 🚀 Utilisation

### Animations originales (Ez scalaire)
```bash
python examples/anim_01_dielectric_refraction.py
python examples/anim_02_metal_reflection.py
python examples/anim_03_lossy_medium.py
python examples/anim_04_dielectric_cavity.py
python examples/anim_05_layered_materials.py
```

### Animations vectorielles (vecteurs H)
```bash
python examples/anim_01_vector_refraction.py
python examples/anim_02_vector_metal.py
python examples/anim_03_vector_lossy.py
python examples/anim_04_vector_cavity.py
python examples/anim_05_vector_multilayer.py
```

### Script de comparaison
```bash
python examples/demo_source_comparison.py
```
Génère des visualisations montrant la différence entre source continue et impulsion.

---

## 📁 Résultats

### Animations originales
```
champs_v4/results/
├── anim_01_dielectric/  (Ez scalaire)
├── anim_02_metal/
├── anim_03_lossy/
├── anim_04_cavity/
└── anim_05_multilayer/
```

### Animations vectorielles
```
champs_v4/results/
├── anim_01_vectors/  (vecteurs H)
├── anim_02_vectors/
├── anim_03_vectors/
├── anim_04_vectors/
└── anim_05_vectors/
```

### Comparaison
```
champs_v4/results/source_comparison/
├── continuous_vs_pulse.png     (propagation comparative)
└── signal_comparison.png       (signaux temporels)
```

---

## 🎨 Visualisations disponibles

### Pour chaque animation :

**Version originale** :
- Champ Ez (scalaire)
- Colormap RdBu_r, seismic, plasma, etc.
- Magnitude du champ électrique

**Version vectorielle** :
- 2 vues simultanées
- Vecteurs normalisés (toutes directions visibles)
- Streamlines ou vue hybride
- Champ magnétique H dans le plan

---

## 🔍 Validation

Testez la comparaison :
```bash
python examples/demo_source_comparison.py
```

Cela génère :
1. **5 snapshots** montrant la propagation continue vs impulsion
2. **Graphes temporels** des deux types de signaux
3. **Analyse comparative** des avantages

---

## ✨ Avantages de la mise à jour

### Pour l'enseignement :
- ✅ Propagation clairement visible
- ✅ Phénomènes physiques distincts
- ✅ Facilite la compréhension

### Pour l'analyse :
- ✅ Séparation temporelle des événements
- ✅ Identification des réflexions multiples
- ✅ Mesure des temps de propagation

### Pour la visualisation :
- ✅ Moins d'interférences parasites
- ✅ Champ plus propre
- ✅ Meilleurs snapshots

---

## 📚 Rappel physique

### Mode TM (Transverse Magnetic)
- **Ez** : perpendiculaire au plan (visualisé en scalaire)
- **Hx, Hy** : dans le plan (visualisés en vecteurs)

### Impulsion gaussienne
- Paquet d'onde localisé dans l'espace et le temps
- Contenu spectral large (transformée de Fourier)
- Idéal pour observer la propagation

---

*Documentation mise à jour - Décembre 2025*
