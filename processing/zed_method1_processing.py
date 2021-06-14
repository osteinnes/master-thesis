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


for file in filelist:

    # Load file
    pc = o3d.io.read_point_cloud(file)
    
    # Crop based on area of interest
    crop = o3d.geometry.AxisAlignedBoundingBox()
    crop = crop.create_from_points(points)
    te = pc.crop(crop)
    
    # Downsample dense point cloud
    downsampled =  te.voxel_down_sample(voxel_size=0.02)
    cl, ind = downsampled.remove_statistical_outlier(nb_neighbors=10,
                                                    std_ratio=0.08)
    
    # Timer
    start = time.process_time()

    # Perform statistical outlier removal
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

        # Retrieve cluster
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

        # You need to specify what axis to orient the polygon to.
        # I choose the "Y" axis. I made the max value the maximum Y of
        # the polygon vertices and the min value the minimum Y of the
        # polygon vertices.
        vol.orthogonal_axis = "X"
        vol.axis_max = np.max(bounding_polygon[:, 0])
        vol.axis_min = np.min(bounding_polygon[:, 0])

        # Set all the Y values to 0 (they aren't needed since we specified what they
        # should be using just vol.axis_max and vol.axis_min).
        bounding_polygon[:, 0] = 0

        # Convert the np.array to a Vector3dVector
        vol.bounding_polygon = o3d.utility.Vector3dVector(bounding_polygon)

        # Crop the point cloud using the Vector3dVector
        cropped_rail = vol.crop_point_cloud(cl)

        # Find plane through rail cluster
        plane_eq, rail_indexes = cropped_rail.segment_plane(0.1,10,100)
        rail_segment = cropped_rail.select_by_index(rail_indexes)
        rail_segment.paint_uniform_color([1,0,0])
        rail_segment_inverted = cropped_rail.select_by_index(rail_indexes, True)
        rail_segment_inverted.paint_uniform_color([0,1,0])
        bounding_box2 = cropped_rail.get_axis_aligned_bounding_box()
        bounding_box2.color = (1, 0, 0)

        # Find rail top
        rail = rail_cluster.astype("float64")
        rail_top = rail[rail[:,1]==np.min(rail[:,1])]
        rail_bottom = rail[rail[:,1]==np.max(rail[:,1])]

        # Get BB center
        arb_points = np.asarray(bounding_box2.get_box_points())
        c = bounding_box2.get_center()

        # Estimate road surface plane equation
        # Segment the estimated plane, return estimated plane equation and indexes of inliers.
        road_surface_plane_eq, road_surface_indexes = cropped_pcd.segment_plane(0.01,20,100)

        # Point cloud of estimated road plane
        est_road_surface = cropped_pcd.select_by_index(road_surface_indexes)
        est_road_surface.paint_uniform_color([1,0,0])

        # Point cloud of outliers of the estimated road plane
        outliers_road_surface = cropped_pcd.select_by_index(road_surface_indexes, True)

        #Calculate height relative to road surface
        a2, b2, c2, d2 = road_surface_plane_eq
        t = rail_top[0]
        height = shortest_distance(t[0],t[1],t[2],a2,b2,c2,d2)

        # Calculate angle realtive to road surface
        a4, b4, c4, d4 = plane_eq
        angl = angle(road_surface_plane_eq[0],road_surface_plane_eq[1],road_surface_plane_eq[2],a4,b4,c4)
        heights.append(height)
        angles.append(angl)
    else: 
        heights.append(math.nan)
        angles.append(math.nan)
        print("Rail not found, cluster failed")
        
np.savetxt("heights.csv", heights, delimiter=',')
np.savetxt("angles.csv", angles, delimiter=',')