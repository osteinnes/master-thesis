import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

accumulated_verts = None

PLY_HEADER = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
FILENAME = "./stereo5.ply"

def write_ply():
    with open(FILENAME, 'w') as f:
        f.write(PLY_HEADER % dict(vert_num=len(accumulated_verts)))
        np.savetxt(f, accumulated_verts, '%f %f %f %d %d %d')


def append_ply_array(verts, colors):
    global accumulated_verts
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts_new = np.hstack([verts, colors])
    if accumulated_verts is not None:
        accumulated_verts = np.vstack([accumulated_verts, verts_new])
    else:
        accumulated_verts = verts_new

left = cv2.imread("C:\\Users\\olema\\Documents\\depthg\\light_left.bmp")
right = cv2.imread("C:\\Users\\olema\\Documents\\depthg\\light_right.bmp")

left_im = cv2.rotate(left, cv2.ROTATE_90_CLOCKWISE)
right_im = cv2.rotate(right, cv2.ROTATE_90_CLOCKWISE)

w = 1544
h = 2064



# Best pinhole
camera_matrix_1 = np.array([1745.080557130917, 0.0, 760.1968017918941, 0.0, 1746.7714914866085, 1086.605580833187, 0.0, 0.0, 1.0]).reshape((3,3))
dist_coeffs_1 = np.array([-0.2252285112110764, 0.1719381021226991, 0.0006972463831982788, -0.0007314044876933391, -0.07667270615073435])
R1 = np.array([0.999984325225602, 0.0004900521674318758, -0.005577557885923413, -0.00048719918442757576, 0.9999997498066026, 0.0005128583498882053, 0.005577807817801204, -0.0005101329292961465, 0.9999843137891423]).reshape((3,3))
P1 = np.array([1757.1251501760169, 0.0, 730.9820022583008, 0.0, 0.0, 1757.1251501760169, 1078.1273651123047, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape((3,4))

camera_matrix_2 = np.array([1743.9608759297628, 0.0, 728.690162670372, 0.0, 1745.1287828312136, 1049.2640575164812, 0.0, 0.0, 1.0]).reshape((3,3))
dist_coeffs_2 = np.array([-0.21415746466932212, 0.13250962753633583, 0.0006391239786728067, -0.000774165985291772, -0.031464328284912274])
R2 = np.array([0.9999235742949814, -0.0007755760889598082, 0.01233872160635034, 0.000781887335119053, 0.999999565961925, -0.0005066834877563608, -0.012338323279277592, 0.0005162922542691593, 0.9999237467031994]).reshape((3,3))
P2 = np.array([1757.1251501760169, 0.0, 730.9820022583008, -177.4204152588782, 0.0, 1757.1251501760169, 1078.1273651123047, 0.0, 0.0, 0.0, 1.0, 0.0]).reshape((3,4))

map1, map2 = cv2.initUndistortRectifyMap(camera_matrix_1, dist_coeffs_1, R1, P1, (w, h), cv2.CV_16SC2)
map3, map4 = cv2.initUndistortRectifyMap(camera_matrix_2, dist_coeffs_2, R2, P2, (w, h), cv2.CV_16SC2)
rectified1 = cv2.remap(left_im, map1, map2, cv2.INTER_LINEAR)
rectified2 = cv2.remap(right_im, map3, map4, cv2.INTER_LINEAR)


cv2.imwrite("rect_left.bmp", rectified1)
cv2.imwrite("rect_right.bmp", rectified2)

rect1 =cv2.cvtColor(rectified1, cv2.COLOR_BGR2GRAY)
rect2 = cv2.cvtColor(rectified2, cv2.COLOR_BGR2GRAY)

cv2.imwrite("gray_left.bmp", rect1)
cv2.imwrite("gray_right.bmp", rect2)
min_disp = 16*2
num_disp = 16*9

window_size = 7

stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                numDisparities=num_disp,
                                blockSize=10,
                                P1=8 * 3 * window_size ** 2,
                                P2=32 * 3 * window_size ** 2,
                                disp12MaxDiff=1,
                                uniquenessRatio=2,
                                speckleWindowSize=50,
                                speckleRange=2
                                )

# Computing disparity
disparity = stereo.compute(rect1, rect2).astype(np.float32) / 16.0

# Reconstructing point cloud
h, w = rect1.shape[:2]
f = 0.6 * w  # guess for focal length

# Best pinhole
R = np.array([0.9998386989876213, 0.001278196039342726, -0.017914860357420796, -0.0012598831259722201, 0.9999986723182127, 0.0010334681034194011, 0.017916157547023557, -0.0010107307736996018, 0.9998389818976121]).reshape(3,3)
T = np.array([-0.10096426868671587, 7.831145764110309e-05, -0.0012458652196421698])

RR1, RR2, RP1, RP2, Q, ROI1, ROI2 = cv2.stereoRectify(camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2, [w,h], R, T)
print("Q: ", Q)
points = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
colors = cv2.cvtColor(rectified1, cv2.COLOR_BGR2RGB)
mask = disparity > disparity.min()+1
out_points = points[mask]
out_colors = colors[mask]
append_ply_array(out_points, out_colors)

disparity_scaled = (disparity - min_disp) / num_disp
disparity_scaled += abs(np.amin(disparity_scaled))
disparity_scaled /= np.amax(disparity_scaled)
disparity_scaled[disparity_scaled < 0] = 0
pc =  np.array(255 * disparity_scaled, np.uint8) 

# Plot disparity map (depth map)
plt.imshow(disparity,'gray')
plt.show()

# Sace disparity
cv2.imwrite("disparity.bmp", disparity)

# Save point cloud
write_ply()
