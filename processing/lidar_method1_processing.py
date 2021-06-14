import os
import open3d as o3d
import numpy as np 
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import math
import time

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    
# Function to find Angle
# Credit: https://www.geeksforgeeks.org/angle-between-two-planes-in-3d/
def angle(a1, b1, c1, a2, b2, c2):  
      
    d = ( a1 * a2 + b1 * b2 + c1 * c2 ) 
    e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1) 
    e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2) 
    d = d / (e1 * e2) 
    A = math.degrees(math.acos(d)) 
    print("Angle is", A, "degree") 
    return A
    
# Function to find distance 
# credit: https://www.geeksforgeeks.org/distance-between-a-point-and-a-plane-in-3-d/
def shortest_distance(x1, y1, z1, a, b, c, d):  
      
    d = abs((a * x1 + b * y1 + c * z1 + d))  
    e = (math.sqrt(a * a + b * b + c * c)) 
    print("Perpendicular distance is", d/e)
    return  d/e
    
# Function to find equation of plane.
# credit: https://stackoverflow.com/a/54168183
def equation_plane(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return a,b,c,d

# Set directory to LIDAR data
directory = '/home/osteinnes/os-master/data/slow2/'

# Iteration parameters
number_of_files = len([item for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item))])
firstFileIndex = 1 # Index of first file
numFiles = number_of_files-firstFileIndex+1       # Number of files to be loaded

# Load data files from CE30-C Lidar
filelist = []
for i in range(firstFileIndex,firstFileIndex+numFiles):
    filelist.append(directory+"%s.xyzrgb" %i)


heights = []
angles = []
rail_points = []

start = time.process_time()

for file in filelist:
    
    # Load lidar file
    lidar = o3d.io.read_point_cloud(file)
    
    # Expected position assumptions
    x = np.asarray(lidar.points)[:,0]>-37
    y = np.asarray(lidar.points)[:,2]>1
    test_list = [a and b for a, b in zip(x, y)]
    res = [i for i, val in enumerate(test_list) if val]
    lidar_pc = lidar.select_by_index(res)
    
    # Contains more than 20 points
    if len(np.asarray(lidar_pc.points))>20:

        # Clustering
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                lidar_pc.cluster_dbscan(eps=10, min_points=30, print_progress=True))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        lidar_pc.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # If any cluster
        if len(np.unique(labels))>0:

            # Retrieve valid points
            valid_points = labels[labels!=-1]
            counts = np.bincount(valid_points)
            rail = np.argmax(counts)
            cluster = np.asarray(lidar_pc.points)[(labels==rail),:]

            if len(cluster)>20:

                # Create new point cloud object of cluster for processing
                rail_cluster = cluster
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(rail_cluster)

                # Find rail top
                rail = rail_cluster.astype("float64")

                # Get top x coordinate
                rail_top = rail[rail[:,0]==np.max(rail[:,0])]

                # Rotate YZ plane with 3 degrees to compensate for moutning on car
                lidar_eq = equation_plane(0,0,0,-0.5240777928304134,1,10.01372345997921,-0.5240777928304134,0,10.01372345997921)
                
                # Measure height 
                height = shortest_distance(rail_top[0,0], rail_top[0,1], rail_top[0,2], lidar_eq[0], lidar_eq[1], lidar_eq[2], lidar_eq[3])
                heights.append((57+height))
                plane_eq, rail_indexes = pcd.segment_plane(100,10,100)

                # Measure angle
                ang = angle(lidar_eq[0],lidar_eq[1],lidar_eq[2], plane_eq[0],plane_eq[1],plane_eq[2])
                angles.append(ang)
                
                rail_points.append(len(rail))

        else:
            heights.append(math.nan)
    else:
        heights.append(math.nan)

np.savetxt("lidar_heights.csv", heights, delimiter=',')
np.savetxt("lidar_angles.csv", angles, delimiter=',')
        
print(heights)
print("Angles: ", angles)

print(len(heights))

print("Average number of points describing road rail: ", np.mean(rail_points))

print("Time taken: ", time.process_time() - start, " sec")