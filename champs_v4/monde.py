import corps 
import constantes

import numpy as np




class Monde():
    def __init__(self,dimensions_total,dimensions_grille) -> None:
        # dimensions et grille ------! bien mettre les unitées !
        self.dimensions_total = dimensions_total
        self.dimensions_grille = dimensions_grille
        self.grille, champs_e , champs_b = np.zeros(dimensions_total//dimensions_grille,dimensions_total//dimensions_grille)
        self.nombres_casses = np.size(self.grille)
        self.liste_corps: list[corps.class_Corps] = []
        
    def ajoute_corps(self,nouveau_corps: corps.class_Corps):  
        self.liste_corps.append(nouveau_corps)
        
    def positions_discretes(self):
        positions = np.array([(corps.x, corps.y) for corps in self.liste_corps])
        positions_discretes = (positions // self.dimensions_grille) % self.grille.shape
        for i, corps in enumerate(self.liste_corps):
            corps.x, corps.y = positions_discretes[i, :]
            
