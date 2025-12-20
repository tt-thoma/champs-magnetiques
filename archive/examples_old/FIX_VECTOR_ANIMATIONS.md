# Corrections des Animations Vectorielles

## 🔧 Problème identifié

Les animations vectorielles 1, 2, 3, 4 et 5 avaient un bug : le `VectorFieldVisualizer` était créé **une seule fois** avant la boucle de simulation, donc il ne voyait que les champs **initiaux** (vides).

## ✅ Solution appliquée

Le visualiseur doit être **recréé à chaque frame** pour capturer l'état actuel des champs électromagnétiques.

### AVANT (incorrect) :
```python
# Visualiseur créé UNE SEULE fois
viz = VectorFieldVisualizer(sim, field='auto', z_index=0)

for n in range(nsteps):
    sim.step()
    
    if n % frame_interval == 0:
        # Utilise toujours les mêmes données initiales !
        viz.plot_normalized(axes[0], ...)
```

### APRÈS (correct) :
```python
for n in range(nsteps):
    sim.step()
    
    if n % frame_interval == 0:
        # Créer visualiseur avec les données ACTUELLES
        viz = VectorFieldVisualizer(sim, field='auto', z_index=0)
        viz.plot_normalized(axes[0], ...)
```

## 📝 Fichiers corrigés

1. ✅ **anim_01_vector_refraction.py** - Réfraction
2. ✅ **anim_02_vector_metal.py** - Réflexion métal
3. ✅ **anim_03_vector_lossy.py** - Atténuation
4. ✅ **anim_04_vector_cavity.py** - Cavité résonante
5. ✅ **anim_05_vector_multilayer.py** - Multicouche

## 🎯 Résultat

Les animations vectorielles affichent maintenant correctement l'évolution temporelle des champs magnétiques !

---

*Note : Les animations originales (Ez scalaire) n'avaient pas ce problème car elles lisent directement `sim.Ez` sans passer par un objet intermédiaire.*
