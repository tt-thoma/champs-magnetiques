"""
Script pour exécuter toutes les animations d'exemples de matériaux.
"""
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# Import des modules d'animation
from examples import anim_01_dielectric_refraction
from examples import anim_02_metal_reflection
from examples import anim_03_lossy_medium
from examples import anim_04_dielectric_cavity
from examples import anim_05_layered_materials


def main():
    """Exécute toutes les animations de démonstration."""
    
    animations = [
        ("1 - Réfraction diélectrique", anim_01_dielectric_refraction.main),
        ("2 - Réflexion métallique", anim_02_metal_reflection.main),
        ("3 - Milieu avec pertes", anim_03_lossy_medium.main),
        ("4 - Cavité résonante", anim_04_dielectric_cavity.main),
        ("5 - Structure multicouche", anim_05_layered_materials.main),
    ]
    
    print("\n" + "=" * 70)
    print(" 🎬 GÉNÉRATION DES 5 ANIMATIONS DE MATÉRIAUX ".center(70))
    print("=" * 70 + "\n")
    
    for i, (name, func) in enumerate(animations, 1):
        print(f"\n{'#' * 70}")
        print(f"  Animation {i}/5 : {name}")
        print(f"{'#' * 70}\n")
        
        try:
            func()
            print(f"\n✓ Animation {i} terminée avec succès\n")
        except Exception as e:
            print(f"\n✗ Erreur dans animation {i} : {e}\n")
            import traceback
            traceback.print_exc()
            
            # Demander si on continue
            response = input("\nContinuer avec les animations suivantes ? (o/n) : ")
            if response.lower() != 'o':
                print("Arrêt du script.")
                return
    
    print("\n" + "=" * 70)
    print(" ✓ TOUTES LES ANIMATIONS SONT TERMINÉES ".center(70))
    print("=" * 70)
    print("\nRésumé des animations créées :")
    print("  1. Réfraction à l'interface air-verre (loi de Snell)")
    print("  2. Réflexion sur plaque métallique (conducteur)")
    print("  3. Atténuation dans eau salée (milieu avec pertes)")
    print("  4. Résonance dans cavité diélectrique")
    print("  5. Interférences dans structure multicouche")
    print("\nLes fichiers MP4 sont dans : champs_v4/results/anim_XX/")


if __name__ == '__main__':
    main()
