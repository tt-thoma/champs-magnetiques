#!/usr/bin/env python3
"""
Guide rapide des animations disponibles
"""

print("""
╔══════════════════════════════════════════════════════════════════════╗
║          ANIMATIONS ÉLECTROMAGNÉTIQUES DISPONIBLES                   ║
╚══════════════════════════════════════════════════════════════════════╝

📊 ANIMATIONS SCALAIRES (magnitude Ez)
════════════════════════════════════════════════════════════════════════

1. Réfraction (Air → Verre)
   python examples/anim_01_dielectric_refraction.py
   Loi de Snell, réflexion partielle

2. Réflexion sur Métal
   python examples/anim_02_metal_reflection.py
   Onde stationnaire, conducteur parfait

3. Atténuation (Milieu avec pertes)
   python examples/anim_03_lossy_medium.py
   Décroissance exponentielle, pertes Joule

4. Cavité Résonante
   python examples/anim_04_dielectric_cavity.py
   Modes résonants, patterns stationnaires

5. Structure Multicouche
   python examples/anim_05_layered_materials.py
   Réflexions multiples, interférences

🔷 ANIMATIONS VECTORIELLES (champ H normalisé)
════════════════════════════════════════════════════════════════════════

6. Vecteurs Réfraction
   python examples/anim_01_vector_refraction.py
   Circulation du champ, changement de direction

7. Vecteurs Métal
   python examples/anim_02_vector_metal.py
   Champ nul dans conducteur, rotation autour

8. Vecteurs Atténuation
   python examples/anim_03_vector_lossy.py
   Affaiblissement progressif des vecteurs

9. Vecteurs Cavité
   python examples/anim_04_vector_cavity.py
   Circulation complexe, vortex

10. Vecteurs Multicouche
    python examples/anim_05_vector_multilayer.py
    Topologie du champ, interférences vectorielles

🎓 DÉMOS PÉDAGOGIQUES
════════════════════════════════════════════════════════════════════════

A. Comparaison des sources (continu vs impulsion)
   python examples/demo_source_comparison.py
   Avantage des impulsions gaussiennes

B. Vecteurs normalisés (explication)
   python examples/demo_normalized_vectors.py
   4 modes de visualisation vectorielle

C. Propagation simple (base)
   python examples/demo_simple_propagation.py
   Concepts fondamentaux FDTD

🚀 LANCER TOUTES LES ANIMATIONS
════════════════════════════════════════════════════════════════════════

python examples/generate_all_10_animations.py

Génère les 10 animations (5 scalaires + 5 vectorielles)
Durée : ~20-30 minutes
Sortie : champs_v4/results/anim_XX/

════════════════════════════════════════════════════════════════════════

💡 Conseils :
   - Chaque animation crée un dossier results/anim_XX/
   - Frames PNG : results/anim_XX/frames/
   - Vidéo MP4 : results/anim_XX/anim_XX.mp4 (nécessite ffmpeg)
   - Logs détaillés affichés pendant l'exécution

📚 Documentation complète : README.md
""")
