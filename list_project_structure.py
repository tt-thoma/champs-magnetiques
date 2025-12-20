#!/usr/bin/env python3
"""
Liste les fichiers actifs du projet champs-magnetiques
après nettoyage et réorganisation (Dec 2025)
"""

import os
from pathlib import Path

def list_project_files():
    """Liste tous les fichiers actifs du projet"""
    
    base = Path(__file__).parent
    
    print("=" * 70)
    print("PROJET CHAMPS-MAGNETIQUES - STRUCTURE ACTIVE")
    print("=" * 70)
    
    # Core modules
    print("\n📦 MODULES PRINCIPAUX (champs_v4/)")
    core_dir = base / "champs_v4"
    if core_dir.exists():
        for py_file in sorted(core_dir.rglob("*.py")):
            if "__pycache__" not in str(py_file):
                rel_path = py_file.relative_to(base)
                print(f"   {rel_path}")
    
    # Examples
    print("\n🎬 ANIMATIONS & DEMOS (examples/)")
    examples_dir = base / "examples"
    if examples_dir.exists():
        py_files = sorted(examples_dir.glob("*.py"))
        
        # Animations scalaires
        print("   Scalar (Ez magnitude):")
        for f in py_files:
            if f.name.startswith("anim_") and "vector" not in f.name:
                print(f"      • {f.name}")
        
        # Animations vectorielles
        print("   Vector (H field):")
        for f in py_files:
            if f.name.startswith("anim_") and "vector" in f.name:
                print(f"      • {f.name}")
        
        # Demos
        print("   Demos:")
        for f in py_files:
            if f.name.startswith("demo_"):
                print(f"      • {f.name}")
        
        # Launcher
        print("   Launcher:")
        for f in py_files:
            if f.name.startswith("generate_"):
                print(f"      • {f.name}")
    
    # Documentation
    print("\n📚 DOCUMENTATION")
    for md_file in sorted(base.glob("*.md")):
        print(f"   {md_file.name}")
    
    # Archive
    print("\n🗄️  ARCHIVE (archive/examples_old/)")
    archive_dir = base / "archive" / "examples_old"
    if archive_dir.exists():
        archived = list(archive_dir.glob("*.py")) + list(archive_dir.glob("*.md"))
        print(f"   {len(archived)} fichiers archivés")
        print(f"   (voir archive/examples_old/README.md)")
    
    # Stats
    print("\n" + "=" * 70)
    total_py = len(list(base.rglob("*.py")))
    active_py = total_py - len(list(archive_dir.rglob("*.py"))) if archive_dir.exists() else total_py
    print(f"TOTAL: {active_py} fichiers Python actifs / {total_py} totaux")
    print("=" * 70)

if __name__ == "__main__":
    list_project_files()
