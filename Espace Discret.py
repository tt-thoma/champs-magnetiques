import numpy as np

class Monde():
    def __init__(self,taille_cellule,taille_monde) -> None:
        self.liste_particule: list = []
        self.taille_cellule = taille_cellule
        self.taille_monde = taille_monde
        self.plan = np.zeros((taille_monde // taille_cellule, taille_monde // taille_cellule))
        
    def test(self):
        print(self.plan)
        
        
monde = Monde(2,10)
monde.test()
