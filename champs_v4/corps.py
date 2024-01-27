
class class_Corps():
    def __init__(self,x,y,vitesse_x,vitesse_y,aceleration_x,aceleration_y,densité_charge_éléctrique,masse) -> None:
        # Position
        self.x = x
        self.y = y
        # Vitesse
        self.vitesse_x = vitesse_x
        self.vitesse_y = vitesse_y
        # Accélération
        self.aceleration_x = aceleration_x
        self.aceleration_y = aceleration_y
        # propriétés fondamentales
        self.densité_charge_éléctrique = densité_charge_éléctrique
        self.masse = masse
    
    