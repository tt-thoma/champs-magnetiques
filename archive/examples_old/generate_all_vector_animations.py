"""
Script principal pour generer les 5 animations VECTORIELLES.

Lance toutes les animations avec visualisation des vecteurs du champ.
"""
import sys
import subprocess
from pathlib import Path

def run_animation(script_name, description):
    """Execute un script d'animation."""
    print()
    print("=" * 80)
    print(f"  LANCEMENT : {description}")
    print("=" * 80)
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False  # Afficher sortie en temps reel
        )
        print()
        print(f"[OK] {description} - TERMINE")
        return True
    except subprocess.CalledProcessError as e:
        print()
        print(f"[ERREUR] {description} - ECHEC")
        return False
    except KeyboardInterrupt:
        print()
        print(f"[ANNULE] {description} - Interrompu par utilisateur")
        return False


def main():
    print()
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 20 + "GENERATION DES 5 ANIMATIONS VECTORIELLES" + " " * 17 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print()
    print("Ce script va generer 5 animations montrant :")
    print("  1. Refraction (interface air/verre)")
    print("  2. Reflexion (conducteur metallique)")
    print("  3. Attenuation (milieu avec pertes)")
    print("  4. Resonance (cavite dielectrique)")
    print("  5. Interferences (structure multicouche)")
    print()
    print("Chaque animation montre les VECTEURS du champ magnetique H.")
    print()
    
    # Demander confirmation
    response = input("Continuer ? (o/N) : ").strip().lower()
    if response not in ['o', 'oui', 'y', 'yes']:
        print("Annule.")
        return
    
    # Liste des animations
    animations = [
        ("anim_01_vector_refraction.py", "Animation 1 : REFRACTION"),
        ("anim_02_vector_metal.py", "Animation 2 : REFLEXION METAL"),
        ("anim_03_vector_lossy.py", "Animation 3 : ATTENUATION"),
        ("anim_04_vector_cavity.py", "Animation 4 : CAVITE RESONANTE"),
        ("anim_05_vector_multilayer.py", "Animation 5 : MULTICOUCHE"),
    ]
    
    # Obtenir le repertoire du script
    script_dir = Path(__file__).parent
    
    # Compteurs
    success_count = 0
    fail_count = 0
    
    # Lancer chaque animation
    for i, (script_name, description) in enumerate(animations, 1):
        script_path = script_dir / script_name
        
        if not script_path.exists():
            print(f"[ERREUR] Script introuvable : {script_name}")
            fail_count += 1
            continue
        
        print()
        print(f">>> Animation {i}/5")
        
        if run_animation(str(script_path), description):
            success_count += 1
        else:
            fail_count += 1
            
            # Demander si continuer
            response = input("Continuer avec les autres animations ? (o/N) : ").strip().lower()
            if response not in ['o', 'oui', 'y', 'yes']:
                print("Arret demande par utilisateur.")
                break
    
    # Resume
    print()
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 32 + "RESUME FINAL" + " " * 34 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print()
    print(f"Animations reussies : {success_count}")
    print(f"Animations echouees : {fail_count}")
    print()
    
    if success_count > 0:
        print("Resultats disponibles dans :")
        print("  champs_v4/results/anim_01_vectors/")
        print("  champs_v4/results/anim_02_vectors/")
        print("  champs_v4/results/anim_03_vectors/")
        print("  champs_v4/results/anim_04_vectors/")
        print("  champs_v4/results/anim_05_vectors/")
        print()
        print("Chaque dossier contient :")
        print("  - frames/ : Images PNG de chaque frame")
        print("  - *.mp4 : Video complete (si ffmpeg disponible)")
    
    print()
    print("#" * 80)


if __name__ == '__main__':
    main()
