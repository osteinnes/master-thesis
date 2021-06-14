import open3d as o3d
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import os

np_points = [
    [7,5,-0],
    [7,5,2],
    [7,-5,2],
    [7,-5,-0],
    [-7,5,-0],
    [-7,5,2],
    [-7,-5,2],
    [-7,-5,-0]
]

# From numpy to Open3D
points = o3d.utility.Vector3dVector(np_points)

import copy
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

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

# Path for dataset
directory = '/home/osteinnes/os-master/data/guttorm/'

# Iteration values
number_of_files = len([item for item in os.listdir(directory) if os.path.isfile(os.path.join(directory, item))])
firstFileIndex = 1 # Index of first file
numFiles = number_of_files       # Number of files to be loaded

# Load data files from CE30-C Lidar
filelist = []
for i in range(firstFileIndex,firstFileIndex+numFiles):
    filelist.append(directory+"pc_%s.ply" %i)

heights = []
angles = []
fitness = []
inlier_rmse = []

# Rail CAD model
gt = o3d.io.read_point_cloud("/home/osteinnes/os-master/code/pointclouds/a-rail-obj-pos.ply")

for file in filelist:

    # Read point cloud
    pc = o3d.io.read_point_cloud(file)
    
    # Crop based on expected area assumption
    crop = o3d.geometry.AxisAlignedBoundingBox()
    crop = crop.create_from_points(points)
    te = pc.crop(crop)
    
    # Downsample point cloud
    downsampled =  te.voxel_down_sample(voxel_size=0.02)
    downsampled2 = te.voxel_down_sample(voxel_size=0.01)

    # Statistical outlier removal
    cl, ind = downsampled.remove_statistical_outlier(nb_neighbors=10,
                                                    std_ratio=0.08)

    # Uncomment for visualization of inlier/outliers
    #display_inlier_outlier(downsampled, ind)
    
    # Timer
    start = time.process_time()

    # Perfrom clustering
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            cl.cluster_dbscan(eps=0.083, min_points=100, print_progress=False))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    cl.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    start = time.process_time()
    
    if len(np.unique(labels))>2:

        # Extract cluster
        valid_points = labels[labels!=-1]
        counts = np.bincount(valid_points)
        road = np.argmax(counts)
        cluster = np.asarray(cl.points)[(labels==road),:]
        road_surface = cluster

        # Convert the road_surface array to have type float64
        bounding_polygon = road_surface.astype("float64")

        # Create a SelectionPolygonVolume
        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = "X"
        vol.axis_max = np.max(bounding_polygon[:, 0])
        vol.axis_min = np.min(bounding_polygon[:, 0])
        bounding_polygon[:, 0] = 0
        vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)
        cropped_pcd = vol.crop_point_cloud(cl)
        bounding_box = cropped_pcd.get_axis_aligned_bounding_box()
        bounding_box.color = (1, 0, 0)

        # Clustering guard rail
        rail_valid = valid_points[valid_points!=road]
        railcounts = np.bincount(rail_valid)
        rail = np.argmax(railcounts)
        rail_cluster = np.asarray(cl.points)[(labels==rail),:]

        # Convert the corners array to have type float64
        bounding_polygon = rail_cluster.astype("float64")

        # Create a SelectionPolygonVolume
        vol = o3d.visualization.SelectionPolygonVolume()
        vol.orthogonal_axis = "X"
        vol.axis_max = np.max(bounding_polygon[:, 0])
        vol.axis_min = np.min(bounding_polygon[:, 0])
        bounding_polygon[:, 0] = 0
        vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)
        cropped_rail = vol.crop_point_cloud(downsampled2)

        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            gt, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))

        cropped_rail.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        pcd_fpfh2 = o3d.pipelines.registration.compute_fpfh_feature(
                cropped_rail,
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))

        
        distance_threshold = 0.1
        gt.paint_uniform_color([0, 0.651, 0.929])
        
        fast_result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
                            gt, cropped_rail, pcd_fpfh, pcd_fpfh2,
                            o3d.pipelines.registration.FastGlobalRegistrationOption(
                                maximum_correspondence_distance=distance_threshold))

        # Uncomment for visualization                        
        #draw_registration_result(gt, cropped_rail, fast_result.transformation)
        
        icp_result = o3d.pipelines.registration.registration_icp(
                        gt, cropped_rail, 0.1, fast_result.transformation,
                        o3d.pipelines.registration.TransformationEstimationPointToPlane())

        # Uncomment for visualization
        #draw_registration_result(gt, cropped_rail, icp_result.transformation)
        
        fitness.append(icp_result.fitness)
        inlier_rmse.append(icp_result.inlier_rmse)
        
        # If ICP fitness is over 70 percent - proceed
        if icp_result.fitness > 0.7:
        
            # Apply ICP transformation and segment plane through CAD model.
            temp = copy.deepcopy(gt)
            temp.transform(icp_result.transformation)
            plane_eq, rail_indexes = temp.segment_plane(0.1,10,100)

            # Apply transformation to pre-defined measuring points.
            top_point = np.matmul(icp_result.transformation, [0.235497, 0.256371, 0.979773 ,1])
            bottom_point= np.matmul(icp_result.transformation, [0.367633, -0.021856, 0.930564 ,1])

            # Estimate road surface plane equation
            # Segment the estimated plane, return estimated plane equation and indexes of inliers.
            road_surface_plane_eq, road_surface_indexes = cropped_pcd.segment_plane(0.01,20,100)

            # Point cloud of estimated road plane
            est_road_surface = cropped_pcd.select_by_index(road_surface_indexes)
            est_road_surface.paint_uniform_color([1,0,0])

            # Point cloud of outliers of the estimated road plane
            outliers_road_surface = cropped_pcd.select_by_index(road_surface_indexes, True)

            # Measure height relative to road surface
            a2, b2, c2, d2 = road_surface_plane_eq
            height = shortest_distance(bottom_point[0],bottom_point[1],bottom_point[2],a2,b2,c2,d2)

            # Measure angle relative to road surface
            a4, b4, c4, d4 = plane_eq
            angl = angle(road_surface_plane_eq[0],road_surface_plane_eq[1],road_surface_plane_eq[2],a4,b4,c4)
            
            heights.append(height)
            angles.append(90+(90-angl))

        else:
            heights.append(math.nan)
            angles.append(math.nan)
            print("Rail not found, fitting failed")
    else: 
        heights.append(math.nan)
        angles.append(math.nan)
        fitness.append(math.nan)
        inlier_rmse.append(math.nan)
        print("Rail not found, cluster failed")
        
np.savetxt("zed_heights.csv", heights, delimiter=',')
np.savetxt("zed_angles.csv", angles, delimiter=',')
np.savetxt("zed_fitness.csv", fitness, delimiter=',')
np.savetxt("zed_inlier_rmse.csv", inlier_rmse, delimiter=',')