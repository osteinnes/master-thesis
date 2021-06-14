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
inlier_rmse = []
fitness = []

# Guard rail CAD model
gt = o3d.io.read_point_cloud("/home/osteinnes/os-master/code/pointclouds/a-rail-obj-lidar.ply")

# Pre-defined measuring points
top_point = np.array(gt.points)[268]
bottom_point = np.array(gt.points)[872]
top_point = [top_point[0], top_point[1], top_point[2], 1]
bottom_point = [bottom_point[0], bottom_point[1], bottom_point[2], 1]

start = time.process_time()

for file in filelist:
    
    # Load point cloud
    lidar = o3d.io.read_point_cloud(file)
    
    # Expected area assumption
    x = np.asarray(lidar.points)[:,0]>-37
    y = np.asarray(lidar.points)[:,2]>1
    test_list = [a and b for a, b in zip(x, y)]
    res = [i for i, val in enumerate(test_list) if val]
    lidar_pc = lidar.select_by_index(res)
    
    if len(np.asarray(lidar_pc.points))>20:

        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                lidar_pc.cluster_dbscan(eps=10, min_points=30, print_progress=True))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        lidar_pc.colors = o3d.utility.Vector3dVector(colors[:, :3])

        if len(np.unique(labels))>0:

            # Retrieve valid cluster
            valid_points = labels[labels!=-1]
            counts = np.bincount(valid_points)
            rail = np.argmax(counts)
            cluster = np.asarray(lidar_pc.points)[(labels==rail),:]
            
            if len(cluster)>20:

                rail_cluster = cluster
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(rail_cluster)
                
                # Find rail top
                rail = rail_cluster.astype("float64")
                
                pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    gt, o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=100))

                pcd.estimate_normals(
                        o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))

                pcd_fpfh2 = o3d.pipelines.registration.compute_fpfh_feature(
                        pcd,
                        o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=100))

                gt.paint_uniform_color([0,0,1])

                #o3d.visualization.draw_geometries([gt, pcd])
                
                distance_threshold = 10

                fast_result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
                                    gt, pcd, pcd_fpfh, pcd_fpfh2,
                                    o3d.pipelines.registration.FastGlobalRegistrationOption(
                                        maximum_correspondence_distance=distance_threshold))

                #draw_registration_result(gt, pcd, fast_result.transformation)

                icp_result = o3d.pipelines.registration.registration_icp(
                                gt, pcd, distance_threshold, fast_result.transformation,
                                o3d.pipelines.registration.TransformationEstimationPointToPlane())

                #draw_registration_result(gt, pcd, icp_result.transformation)

                # Get top x coordinate
                rail_top = rail[rail[:,0]==np.max(rail[:,0])]
                
                # Add result metrics
                fitness.append(icp_result.fitness)
                inlier_rmse.append(icp_result.inlier_rmse)

                # Rotat YZ plane with 3 degrees to compensate for moutning on car
                lidar_eq = equation_plane(0,0,0,-0.5240777928304134,1,10.01372345997921,-0.5240777928304134,0,10.01372345997921)
                
                # Apply transformation to pre-defined top and bottom points, and calculate perpendicular distance.
                height_p = np.matmul(icp_result.transformation, top_point)
                height_b = np.matmul(icp_result.transformation, bottom_point) 
                height = shortest_distance(height_p[0], height_p[1], height_p[2], lidar_eq[0], lidar_eq[1], lidar_eq[2], lidar_eq[3])
                heights_b = shortest_distance(height_b[0], height_b[1], height_b[2], lidar_eq[0], lidar_eq[1], lidar_eq[2], lidar_eq[3])
                heights.append((57+heights_b))
                
                # Apply transformation to model
                temp = copy.deepcopy(gt)
                temp.transform(icp_result.transformation)
                
                # Plane of transformed CAD model
                plane_eq, rail_indexes = pcd.segment_plane(10,10,100)

                # Calculate angle to artificial lidar plane
                ang = angle(lidar_eq[0],lidar_eq[1],lidar_eq[2], plane_eq[0],plane_eq[1],plane_eq[2])
                angles.append(ang)
                
                rail_points.append(len(rail))

        else:
            heights.append(math.nan)
    else:
        heights.append(math.nan)
        
print("Time taken: ", np.mean(timetaken))

np.savetxt("lidar_heights.csv", heights, delimiter=',')
np.savetxt("lidar_angles.csv", angles, delimiter=',')
np.savetxt("lidar_fitness.csv", fitness, delimiter=',')
np.savetxt("lidar_inlier_rmse.csv", inlier_rmse, delimiter=',')
        
print(heights)
print("Angles: ", angles)

print(len(heights))

print("Average number of points describing road rail: ", np.mean(rail_points))

print("Time taken: ", time.process_time() - start, " sec")