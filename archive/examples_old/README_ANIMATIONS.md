# 🎬 Animations de Matériaux Électromagnétiques

Ce dossier contient 5 animations démontrant la propagation d'ondes EM dans différents matériaux.

## 📋 Liste des Animations

### 1️⃣ Réfraction Diélectrique
**Fichier :** `anim_01_dielectric_refraction.py`
- **Matériaux :** Interface air (εr=1.0) → verre (εr=2.25)
- **Phénomène :** Loi de Snell-Descartes, changement de vitesse de propagation
- **Fréquence :** 10 GHz
- **Durée :** ~800 pas de temps, 200 frames

### 2️⃣ Réflexion Métallique
**Fichier :** `anim_02_metal_reflection.py`
- **Matériaux :** Air + plaque de cuivre (σ=5.8×10⁷ S/m)
- **Phénomène :** Réflexion quasi-totale sur conducteur parfait
- **Fréquence :** 5 GHz
- **Application :** Blindage EM, miroirs RF

### 3️⃣ Milieu avec Pertes
**Fichier :** `anim_03_lossy_medium.py`
- **Matériaux :** Air → eau salée (εr=80, σ=4 S/m)
- **Phénomène :** Atténuation exponentielle par absorption
- **Fréquence :** 2 GHz
- **Application :** Communications sous-marines, imagerie médicale

### 4️⃣ Cavité Résonante
**Fichier :** `anim_04_dielectric_cavity.py`
- **Matériaux :** Cavité céramique (εr=10) avec murs (εr=20)
- **Phénomène :** Modes de résonance, ondes stationnaires
- **Application :** Filtres RF, oscillateurs

### 5️⃣ Structure Multicouche
**Fichier :** `anim_05_layered_materials.py`
- **Matériaux :** 5 couches alternées (verre, plastique, céramique, téflon, résine)
- **Phénomène :** Interférences constructives/destructives
- **Fréquence :** 15 GHz
- **Application :** Revêtements antireflets, filtres optiques

## 🚀 Utilisation

### Exécuter une animation spécifique :
```powershell
cd champs-magnetiques
python examples/anim_01_dielectric_refraction.py
```

### Exécuter toutes les animations :
```powershell
python examples/run_all_animations.py
```

## 📂 Sorties

Les résultats sont sauvegardés dans :
```
champs_v4/results/
├── anim_01_dielectric/
│   ├── frames/          # Frames PNG individuelles
│   └── refraction_animation.mp4
├── anim_02_metal/
│   └── metal_reflection.mp4
├── anim_03_lossy/
│   └── lossy_medium.mp4
├── anim_04_cavity/
│   └── cavity_resonance.mp4
└── anim_05_multilayer/
    └── multilayer.mp4
```

## ⚙️ Prérequis

### Obligatoires :
- Python 3.8+
- numpy
- matplotlib

### Optionnel (pour MP4) :
- FFmpeg (pour créer les vidéos)
  - Windows : Télécharger depuis https://ffmpeg.org/download.html
  - Ajouter au PATH système

### Installation :
```powershell
pip install numpy matplotlib
```

## 🔧 Configuration

Vous pouvez modifier les paramètres dans chaque script :
- **Résolution** : `nx, ny, nz` (taille de la grille)
- **Précision spatiale** : `dx` (taille des cellules)
- **Fréquence** : `freq` (fréquence de la source)
- **Durée** : `nsteps` (nombre de pas de temps)
- **Qualité vidéo** : `frame_interval` (intervalle entre frames)

## ⚡ Performance

| Animation | Grille | Pas de temps | Temps estimé |
|-----------|--------|--------------|--------------|
| Réfraction | 200×200×1 | 800 | ~2-5 min |
| Métal | 180×180×1 | 1000 | ~2-5 min |
| Pertes | 220×220×1 | 1200 | ~3-6 min |
| Cavité | 160×160×1 | 1500 | ~2-5 min |
| Multicouche | 240×200×1 | 1400 | ~3-6 min |

**Total (toutes) :** ~15-30 minutes

💡 **Astuce :** Installez Numba pour accélérer les calculs :
```powershell
pip install numba
```
Gain de performance : 10-100× plus rapide !

## 📊 Interprétation

Les animations montrent :
- **Couleurs** : Intensité du champ électrique Ez
- **Échelle** : Rouge (positif) → Bleu (négatif)
- **Interfaces** : Marquées par des lignes (jaune, cyan, etc.)
- **Propagation** : Direction et vitesse des ondes

## 🐛 Dépannage

### Erreur "ModuleNotFoundError"
```powershell
pip install numpy matplotlib
```

### FFmpeg non trouvé
Les frames PNG sont quand même sauvegardées dans `results/anim_XX/frames/`

### Simulation trop lente
- Réduire `nsteps` (moins de pas de temps)
- Réduire `nx, ny` (grille plus petite)
- Augmenter `frame_interval` (moins de frames)
- Installer Numba

### NaN dans les résultats
- Vérifier condition CFL : `dt < dx / (c0 * sqrt(3))`
- Réduire `dt`
- Vérifier valeurs de σ et εr (doivent être > 0)

## 📚 Références

- FDTD : Taflove & Hagness, "Computational Electrodynamics"
- Yee Algorithm : IEEE Trans. Antennas Propagat., 1966
- Documentation : `PROJECT_DOCUMENTATION.md`
