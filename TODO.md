# TODO - Améliorations du Projet FDTD Yee3D

Ce fichier liste les axes d'amélioration identifiés pour le projet de simulation FDTD 3D, ainsi que des stratégies pour éviter les NaN et améliorer la stabilité.

## 1. Performance et Optimisation Computationnelle
- [ ] Intégrer Numba (@jit) sur les boucles update_E et update_H pour accélérer les calculs (gain potentiel 10-100x).
- [ ] Implémenter la parallélisation CPU avec multiprocessing (diviser la grille en sous-domaines).
- [ ] Explorer GPU avec CuPy/PyCUDA pour accélérer les simulations 3D.
- [ ] Optimiser la mémoire : utiliser float32 si précision suffisante, éviter les copies inutiles.
- [ ] Remplacer Matplotlib par VTK/Mayavi pour visualisations 3D interactives plus rapides.
- [ ] Ajouter des benchmarks de performance (timeit) pour mesurer les améliorations.

## 2. Précision et Fidélité Physique
- [ ] Implémenter un maillage adaptatif pour raffiner autour des sources sans augmenter la taille globale.
- [ ] Passer à un schéma FDTD d'ordre supérieur (ex. : ordre 4) pour réduire la dispersion numérique.
- [ ] Ajouter support pour matériaux dispersifs (Drude/Lorentz), anisotropes et non-linéaires.
- [ ] Inclure pertes dans les matériaux (conductivité finie).
- [ ] Valider la précision avec solutions analytiques ou benchmarks (CST, COMSOL).

## 3. Fonctionnalités et Extensibilité
- [ ] Ajouter des sources diversifiées : ondes planes, modulées en fréquence (chirp), sources distribuées.
- [ ] Intégrer post-traitement FFT pour analyse fréquentielle (spectres, diagrammes de rayonnement).
- [ ] Calculer la puissance rayonnée et l'impédance d'antenne.

- [ ] Développer une interface graphique (Tkinter/PyQt) pour configuration sans code.
- [ ] Réorganiser le code en classes modulaires (Source, Material, Boundary) pour faciliter l'extension.
- [ ] Ajouter des plugins pour nouveaux modules.

## 4. Stabilité et Robustesse (Éviter les NaN)
- [ ] Implémenter contrôle CFL automatique : calculer dt basé sur CFL max (1/sqrt(3) en 3D).
- [ ] Ajouter damping numérique global (E *= (1 - damping*dt)) pour absorber oscillations.
- [ ] Rendre les enveloppes obligatoires pour toutes sources (gaussienne, Tukey).
- [ ] Limiter amplitude des sources et monitorer max(E) < seuil.
- [ ] Ajouter vérifications NaN automatiques après chaque pas (assert np.isfinite).
- [ ] Utiliser float64 pour éviter underflow/overflow ; clipper les champs si nécessaire.
- [ ] Améliorer l'implémentation PML pour réduire réflexions.
- [ ] Ajouter "soft start" pour sources afin d'éviter transitoires.

## 5. Usabilité et Maintenance
- [ ] Ajouter tests unitaires avec pytest (convergence, conservation d'énergie).
- [ ] Améliorer la gestion d'erreurs et logging pour debugging.

## Stratégies Générales pour Éviter les NaN
- Toujours respecter CFL strict : dt = dx / (c * sqrt(3)) * 0.9.
- Utiliser enveloppes temporelles pour limiter la durée des sources.
- Monitorer les champs en temps réel et arrêter si NaN détecté.
- Valider matériaux (epsilon, mu > 0) et initialisations (E/H = 0).
- Pour debug : réduire grille et durée pour tests rapides.

Priorité : Commencer par la stabilité (damping, CFL) et performance (Numba) pour des simulations fiables et rapides.



Dans un futur lointain :
 fusionner le modèle avec un spice ou un bem .
 
