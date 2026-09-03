#%% Defining Config Packet
class Task2:
    class MetaData:
        OutputExt = 'Data.pkl'
    class Settings:
        class Normals:
            CameraPose = (0.0, 0.0, 0.0) 
            Radius = 500
            Max_nn = 500                 
            TileSize = 5
        class Smoothing:
            Cutoff = 0.05    # Regola il grado di lisciatura: più è basso (es. 0.02) più appiattisce, più è alto (es. 0.15) più segue i picchi.
            Order = 3        # Indica la ripidità del taglio del rumore: 3 è lo standard (si auto-riduce per evitare crash su array corti).
            PadPerc = 0.25   # Aggiunge dati speculari (10%) ai bordi prima di filtrare per evitare salti innaturali a inizio e fine onda.
            EdgeOrder = 2    # Usa una curva (2) invece di una retta (1) per calcolare in modo molto più preciso le derivate alle estremità.
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 0