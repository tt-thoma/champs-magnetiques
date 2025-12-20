# Animations Vectorielles des Champs Électromagnétiques

## 🎬 Vue d'ensemble

5 animations montrant les **vecteurs du champ magnétique** dans différentes configurations physiques.

Chaque animation génère :
- **Frames PNG** : Images haute résolution de chaque pas
- **Vidéo MP4** : Animation complète (nécessite ffmpeg)
- **2 vues simultanées** : Vecteurs normalisés + Lignes de champ (ou vue hybride)

---

## 📁 Animations disponibles

### 1. Réfraction (Interface Air/Verre)
**Fichier** : `anim_01_vector_refraction.py`

**Physique** :
- Onde se propage de gauche à droite
- Interface verticale : air (n=1.0) → verre (n=1.5)
- **Réfraction** : changement de direction selon loi de Snell
- **Réflexion partielle** : une partie de l'onde revient

**Résultats** : `champs_v4/results/anim_01_vectors/`

---

### 2. Réflexion sur Métal
**Fichier** : `anim_02_vector_metal.py`

**Physique** :
- Onde sphérique depuis source ponctuelle
- Plaque de cuivre verticale (σ = 5.8×10⁷ S/m)
- **Réflexion totale** sur le conducteur
- Formation d'**onde stationnaire** (interférence)
- Champ nul à l'intérieur du métal

**Résultats** : `champs_v4/results/anim_02_vectors/`

---

### 3. Atténuation (Milieu avec Pertes)
**Fichier** : `anim_03_vector_lossy.py`

**Physique** :
- Transition air → milieu absorbant (σ = 5.0 S/m, εᵣ = 2.5)
- **Atténuation exponentielle** de l'amplitude
- Énergie convertie en chaleur (pertes Joule)
- Vecteurs de plus en plus faibles

**Résultats** : `champs_v4/results/anim_03_vectors/`

---

### 4. Cavité Résonante
**Fichier** : `anim_04_vector_cavity.py`

**Physique** :
- Cavité rectangulaire à murs conducteurs
- Intérieur : diélectrique (εᵣ = 4.0, n = 2.0)
- Excitation par impulsion centrale
- Formation de **modes résonants**
- Patterns stationnaires complexes

**Résultats** : `champs_v4/results/anim_04_vectors/`

---

### 5. Structure Multicouche
**Fichier** : `anim_05_vector_multilayer.py`

**Physique** :
- 4 couches alternées : Air → Verre → Plastique → Air
- **Réflexions multiples** aux interfaces
- **Interférences** constructives/destructives
- Transmission complexe avec patterns

**Résultats** : `champs_v4/results/anim_05_vectors/`

---

## 🚀 Utilisation

### Option 1 : Générer toutes les animations
```bash
python examples/generate_all_vector_animations.py
```
Lance les 5 animations séquentiellement avec confirmation.

### Option 2 : Générer une animation spécifique
```bash
# Animation 1 : Réfraction
python examples/anim_01_vector_refraction.py

# Animation 2 : Réflexion métal
python examples/anim_02_vector_metal.py

# Animation 3 : Atténuation
python examples/anim_03_vector_lossy.py

# Animation 4 : Cavité résonante
python examples/anim_04_vector_cavity.py

# Animation 5 : Multicouche
python examples/anim_05_vector_multilayer.py
```

---

## 📊 Modes de visualisation

### 1. Vecteurs Normalisés
- **Tous les vecteurs ont la même longueur**
- Couleur indique la magnitude originale
- ✅ Avantage : toutes les directions visibles, même en zones faibles
- 📍 Usage : analyser topologie du champ, voir directions partout

### 2. Lignes de Champ (Streamlines)
- Lignes continues suivant le flux du champ
- Couleur indique la magnitude
- ✅ Avantage : vue globale du flux d'énergie
- 📍 Usage : comprendre circulation et trajectoires

### 3. Vue Hybride
- Fond de magnitude + streamlines + quelques vecteurs
- Combinaison des informations
- ✅ Avantage : synthèse complète
- 📍 Usage : présentation générale

---

## 🔧 Configuration requise

### Obligatoire
- Python 3.8+
- numpy
- matplotlib
- Packages du projet (`champs_v4`)

### Optionnel
- **ffmpeg** : pour générer les vidéos MP4
  - Si absent : frames PNG toujours générées
  - Windows : `choco install ffmpeg` ou télécharger depuis ffmpeg.org
  - Linux : `sudo apt install ffmpeg`

---

## ⚙️ Paramètres modifiables

Dans chaque script, vous pouvez ajuster :

```python
# Résolution spatiale
dx = 0.5e-3  # Taille cellule (mm)

# Résolution temporelle
nsteps = 800  # Nombre de pas
frame_interval = 4  # Frames tous les X pas

# Source
freq = 8e9  # Fréquence (Hz)
t0 = 80 * dt  # Centre impulsion
spread = 25 * dt  # Largeur impulsion

# Visualisation
step = 6  # Densité vecteurs (plus grand = moins de vecteurs)
arrow_scale = 3.5  # Taille vecteurs normalisés
```

---

## 📈 Performances

| Animation | Frames | Durée simulation | Taille vidéo |
|-----------|--------|------------------|--------------|
| 1. Réfraction | 200 | ~2-3 min | ~5 MB |
| 2. Métal | 200 | ~2-3 min | ~5 MB |
| 3. Atténuation | 200 | ~3-4 min | ~6 MB |
| 4. Cavité | 188 | ~3-4 min | ~5 MB |
| 5. Multicouche | 200 | ~2-3 min | ~5 MB |

*Durées indicatives sur machine moderne*

---

## 🎯 Interprétation

### Ce que montrent les vecteurs

En **mode TM** (utilisé dans ces simulations) :
- **Champ électrique E** : perpendiculaire au plan (Ez)
- **Champ magnétique H** : dans le plan (Hx, Hy) ← **C'est ce qu'on visualise**

Les vecteurs représentent la **direction et amplitude** du champ H :
- **Longueur** (quiver standard) : proportionnelle à |H|
- **Couleur** : magnitude de |H|
- **Direction** : orientation du champ dans le plan

### Phénomènes observables

✅ **Propagation** : vecteurs se déplacent dans l'espace  
✅ **Réflexion** : inversion des vecteurs à l'interface  
✅ **Réfraction** : changement de direction et longueur d'onde  
✅ **Atténuation** : diminution progressive de l'amplitude  
✅ **Interférence** : patterns complexes de vecteurs opposés  
✅ **Modes** : structures stationnaires dans cavités  

---

## 🐛 Dépannage

### Problème : Pas de vidéo générée
**Solution** : Installer ffmpeg ou utiliser directement les frames PNG

### Problème : Vecteurs trop petits/grands
**Solution** : Ajuster `arrow_scale` (valeur typique : 2.0-5.0)

### Problème : Trop de vecteurs (brouillon)
**Solution** : Augmenter `step` (valeur typique : 5-8)

### Problème : Animation trop rapide/lente
**Solution** : Ajuster `frame_interval` ou framerate ffmpeg

---

## 📚 Références

- **Algorithme** : FDTD (Finite Difference Time Domain) de Yee
- **Mode** : TM (Transverse Magnetic) pour simulations 2D
- **Visualisation** : Module `vector_field_viz.py`

---

## 📝 Notes techniques

- **PML** : Perfectly Matched Layers aux bords (absorbe les ondes)
- **CFL** : Condition de stabilité respectée (dt < dx/(c√2))
- **Numba** : Accélération JIT si disponible
- **Auto-détection** : Choix automatique E vs H selon mode dominant

---

*Généré par le système de visualisation vectorielle - v2.0*
