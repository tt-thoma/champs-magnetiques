"""
Script principal pour generer TOUTES les animations (10 au total).

Lance les 5 animations originales (Ez scalaire) + 5 animations vectorielles (H).
"""
import sys
import subprocess
from pathlib import Path
import time

def run_animation(script_path, description):
    """Execute un script d'animation."""
    print()
    print("=" * 80)
    print(f"  LANCEMENT : {description}")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        
        elapsed = time.time() - start_time
        print()
        print(f"[OK] {description} - TERMINE en {elapsed/60:.1f} min")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print()
        print(f"[ERREUR] {description} - ECHEC apres {elapsed/60:.1f} min")
        return False
        
    except KeyboardInterrupt:
        print()
        print(f"[ANNULE] {description} - Interrompu par utilisateur")
        raise


def main():
    print()
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 18 + "GENERATION DE TOUTES LES ANIMATIONS (10)" + " " * 20 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print()
    print("Ce script va generer 10 animations :")
    print()
    print("ANIMATIONS ORIGINALES (Ez scalaire) :")
    print("  1. Refraction air/verre")
    print("  2. Reflexion sur metal")
    print("  3. Attenuation dans milieu absorbant")
    print("  4. Resonance dans cavite")
    print("  5. Interferences multicouche")
    print()
    print("ANIMATIONS VECTORIELLES (vecteurs H) :")
    print("  6. Refraction air/verre (vecteurs)")
    print("  7. Reflexion sur metal (vecteurs)")
    print("  8. Attenuation (vecteurs)")
    print("  9. Resonance cavite (vecteurs)")
    print(" 10. Interferences multicouche (vecteurs)")
    print()
    print("Temps estime total : 20-30 minutes")
    print()
    
    # Demander confirmation
    response = input("Continuer ? (o/N) : ").strip().lower()
    if response not in ['o', 'oui', 'y', 'yes']:
        print("Annule.")
        return
    
    # Repertoire du script
    script_dir = Path(__file__).parent
    
    # Liste des animations
    animations = [
        # Animations originales (Ez scalaire)
        ("anim_01_dielectric_refraction.py", "1/10 - Refraction (Ez)"),
        ("anim_02_metal_reflection.py", "2/10 - Reflexion metal (Ez)"),
        ("anim_03_lossy_medium.py", "3/10 - Attenuation (Ez)"),
        ("anim_04_dielectric_cavity.py", "4/10 - Cavite resonante (Ez)"),
        ("anim_05_layered_materials.py", "5/10 - Multicouche (Ez)"),
        
        # Animations vectorielles (H)
        ("anim_01_vector_refraction.py", "6/10 - Refraction (vecteurs H)"),
        ("anim_02_vector_metal.py", "7/10 - Reflexion metal (vecteurs H)"),
        ("anim_03_vector_lossy.py", "8/10 - Attenuation (vecteurs H)"),
        ("anim_04_vector_cavity.py", "9/10 - Cavite resonante (vecteurs H)"),
        ("anim_05_vector_multilayer.py", "10/10 - Multicouche (vecteurs H)"),
    ]
    
    # Compteurs
    success_count = 0
    fail_count = 0
    
    start_time_total = time.time()
    
    # Lancer chaque animation
    for script_name, description in animations:
        script_path = script_dir / script_name
        
        if not script_path.exists():
            print(f"[ERREUR] Script introuvable : {script_name}")
            fail_count += 1
            continue
        
        try:
            if run_animation(script_path, description):
                success_count += 1
            else:
                fail_count += 1
                
                # Demander si continuer
                response = input("\nContinuer avec les autres animations ? (o/N) : ").strip().lower()
                if response not in ['o', 'oui', 'y', 'yes']:
                    print("Arret demande par utilisateur.")
                    break
                    
        except KeyboardInterrupt:
            print("\nInterruption par utilisateur.")
            break
    
    # Resume
    elapsed_total = time.time() - start_time_total
    
    print()
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + " " * 32 + "RESUME FINAL" + " " * 34 + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print()
    print(f"Animations reussies : {success_count}/10")
    print(f"Animations echouees : {fail_count}/10")
    print(f"Temps total : {elapsed_total/60:.1f} minutes")
    print()
    
    if success_count > 0:
        print("Resultats disponibles dans :")
        print()
        print("ANIMATIONS ORIGINALES (Ez) :")
        print("  champs_v4/results/anim_01_dielectric/")
        print("  champs_v4/results/anim_02_metal/")
        print("  champs_v4/results/anim_03_lossy/")
        print("  champs_v4/results/anim_04_cavity/")
        print("  champs_v4/results/anim_05_multilayer/")
        print()
        print("ANIMATIONS VECTORIELLES (H) :")
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
