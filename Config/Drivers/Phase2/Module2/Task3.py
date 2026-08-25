#%% Defining Config Packet
class Task3:
    class MetaData:
        OutputExt = 'Data.pkl'
        MeshExt = 'Mesh.ply'
    class Settings:
        class Calib:
            Dataset = 'Dataset1'
            Pair = ('Camera1', 'Camera2')
            Model = 'Model27'
        Bounds = [[5, 20]] #[64, 94]
        class LocalFilter:
            class Size:
                voxelSize = 0.5
            class Radius:
                nbPoints = 25
                radius = 5.0
            class Stats:
                nbNeighbors = 100
                stdRatio = 1.5 
        class GlobalFilter:
            class Positional:
                enabled = True
                exclusion_zones = [ 
                    [[-100, -100 , -1000], [-15, -5, 1000]],
                    [[50, -1000, - 1000],[100, 1000, 1000]],
                    [[-100, -12, 465], [0, -7.0, 1000]]
                ]
            class Size:
                voxelSize = 2.0       
            class Stats:
                enabled = True
                nbNeighbors = 30
                stdRatio = 1.5 
            class Radius:
                enabled = True
                nbPoints = 10           
                radius = 5.0
            class PCA:
                enabled = True 
                search_radius = 100      
                max_nn = 500
                threshold = 0.2    
        class Mesh:
            class NormalOrient:
                radius = 50           
                max_nn = 500       
                cameraLoc = [0.0, 0.0, 0.0]
            class Poisson:
                depth = 12            
                width = 0.0
                scale = 1.1          
                linear_fit = True    
                density_trim = 0.001  #
            class Cleaning:
                max_distance_from_source = 4.5
            class Smoothing:
                class Regularization:
                    enabled = True
                    target_triangles = 15000
                class Subdivision:
                    enabled = True
                    method = 'midpoint'       
                    iterations = 2          
                class Taubin:
                    enabled = True
                    iterations = 150       
                    lambda_filter = 0.53
                    mu = -0.54          
    class General:
        Activation = True
        Maker = True
        Destroyer = False
        Version = 2