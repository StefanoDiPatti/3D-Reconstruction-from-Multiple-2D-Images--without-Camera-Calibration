
"""
Created on Sun Apr 26 23:36:59 2020

@author: HashiramaYaburi
"""


import cv2
import numpy as np
from math import radians
from math import sqrt
import PIL.ExifTags
import PIL.Image
from mpmath import cot
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from skimage import img_as_ubyte
import open3d as o3d
from open3d import *


def Kuwahara(original, winsize):
    image = original.copy()
    # make sure original is a numpy array
    #image = original.astype(np.float64)
    plt.imshow(image, interpolation='nearest')
    plt.show()
    # make sure window size is correct
    if winsize % 2 != 1:
        raise Exception(
            "Invalid winsize %s: winsize must follow formula: w = 4*n+1." % winsize)

    # Build subwindows
    tmpAvgKerRow = np.hstack(
        (np.ones((1, (winsize-1)//2+1)), np.zeros((1, (winsize-1)//2))))
    tmpPadder = np.zeros((1, winsize), dtype=int)
    tmpavgker = np.tile(tmpAvgKerRow, ((winsize-1)//2 + 1, 1))
    tmpavgker = np.vstack((tmpavgker, np.tile(tmpPadder, ((winsize-1)//2, 1))))
    tmpavgker = tmpavgker/np.sum(tmpavgker)

    # tmpavgker is a 'north-west' subwindow (marked as 'a' above)
    # we build a vector of convolution kernels for computing average and
    # variance
    avgker = np.empty((4, winsize, winsize))  # make an empty vector of arrays
    avgker[0] = tmpavgker			# North-west (a)
    avgker[1] = np.fliplr(tmpavgker)  # North-east (b)
    avgker[3] = np.flipud(tmpavgker)  # South-west (c)
    avgker[2] = np.fliplr(avgker[3])  # South-east (d)

    # Create a pixel-by-pixel square of the image
    squaredImg = image**2

    # preallocate these arrays to make it apparently %15 faster
    avgs = np.zeros([4, image.shape[0], image.shape[1]])
    stddevs = np.zeros([4, image.shape[0], image.shape[1]])

    # Calculation of averages and variances on subwindows
    for k in range(4):
        avgs[k] = convolve2d(image, avgker[k], mode='same')
        stddevs[k] = convolve2d(squaredImg, avgker[k], mode='same')
        stddevs[k] = stddevs[k] - avgs[k]**2

    # Choice of index with minimum variance
    # returns index of subwindow with smallest variance
    indices = np.argmin(stddevs, axis=0)

    # print(indices)

    # Building the filtered image (with nested for loops)
    filtered = np.zeros(original.shape)
    for row in range(original.shape[0]):
        for col in range(original.shape[1]):
            filtered[row, col] = avgs[indices[row, col], row, col]

    return filtered

# Function to create point cloud file


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
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

    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


# Function to create point cloud file
def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(int)
    y = np.linspace(0, height - 1, height).astype(int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


# Function to create point cloud file
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
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
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

#== Parameters =======================================================================
BLUR = 21
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,1.0) # In BGR format
def edge_detection(_img):
    #-- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(_img, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
        
    #-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Previously, for a previous version of cv2, this line was: 
    #  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Thanks to notes from commenters, I've updated the code but left this note
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]
        
    #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))
        
        
    #-- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask
        
    #-- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
    _img         = _img.astype('float32') / 255.0                 #  for easy blending
        
    masked = (mask_stack[:,:,0] * _img) + (((1-mask_stack) * MASK_COLOR)[:,:,0]) # Blend
    masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

    #cv2.imshow('_img', masked)                                   # Display
    #cv2.waitKey()
    plt.imshow(_img, interpolation='nearest')
    plt.show()
    
    return masked

window_size = 15
i = 1
count = 1
while i < 7:
    j=i+1
    while j < 7:
        imgL = cv2.imread("./images_example/%d.jpg" % i)  # load left and right images
        imgR = cv2.imread("./images_example/%d.jpg" % j)
        h, w, c = imgL.shape
        dim = (w//4, h//4)
        #dim = (w,h)
        # resize image
        resized = cv2.resize(imgL, dim, interpolation=cv2.INTER_AREA)
        colored_resized = cv2.resize(imgL, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        filtered = Kuwahara(gray, 3)
        plt.imshow(filtered, interpolation='nearest')
        plt.show()
    
        img = cv2.normalize(src=filtered, dst=None, alpha=0,
                            beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        #img = edge_detection(img)
        orb = cv2.ORB_create(nfeatures=100)
        # Find keypoints and descriptors directly
        keypoints, descriptors = orb.detectAndCompute(img, None)
        img_2 = cv2.drawKeypoints(img, keypoints, None)
        plt.imshow(img_2), plt.show()
    
        # resize image
        resized = cv2.resize(imgR, dim, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        filtered = Kuwahara(gray, 3)
        plt.imshow(filtered, interpolation='nearest')
        plt.show()
    
        img_3 = cv2.normalize(src=filtered, dst=None, alpha=0,
                              beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        #img_3 = edge_detection(img_3)
        # Find keypoints and descriptors directly
        keypoints2, descriptors2 = orb.detectAndCompute(img_3, None)
        img_4 = cv2.drawKeypoints(img_3, keypoints2, None)
        plt.imshow(img_4),plt.show()
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(descriptors,descriptors2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 10 matches.
        img_5 = cv2.drawMatches(img_2,keypoints,img_4,keypoints2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.imshow(img_5),plt.show()
    
        img_points1 = []
        img_points2 = []
    
        # Recuperamos las coordenadas de los puntos en correspondencias:
        for match in matches:
            img_points1.append(keypoints[match.queryIdx].pt)
            img_points2.append(keypoints2[match.trainIdx].pt)
    
        img_points1 = np.array(img_points1, dtype=np.int32)
        img_points2 = np.array(img_points2, dtype=np.int32)
    
        F, mask = cv2.findFundamentalMat(img_points1, img_points2, cv2.RANSAC)
        print("Fundamental Matrix")
        print(F)
    
        # eliminate outliers
        img_points1 = img_points1[mask.ravel() == 1]
        img_points2 = img_points2[mask.ravel() == 1]
    
        E, mask = cv2.findEssentialMat(
            img_points1, img_points2, F, method=cv2.RANSAC, prob=0.999, threshold=3.0)
    
        # POSE RECOVERY #
        points, R, t, mask = cv2.recoverPose(E, img_points1, img_points2)
        #anglesBetweenImages = rotationMatrixToEulerAngles(R)
    
        print("Essential matrix")
        print(E)
    
        H, status = cv2.findHomography(img_points1, img_points2, cv2.RANSAC, 5.0)
        H2, status2 = cv2.findHomography(img_points2, img_points1, cv2.RANSAC, 5.0)
    
        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(img_points2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = drawlines(img, img_3, lines1, img_points1, img_points2)
    
        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(img_points1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = drawlines(img_3, img, lines2, img_points2, img_points1)
    
        plt.subplot(121), plt.imshow(img5)
        plt.subplot(122), plt.imshow(img3)
        plt.show()
    
        points1r = img_points1.reshape((img_points1.shape[0] * 2, 1))
        points2r = img_points2.reshape((img_points2.shape[0] * 2, 1))
    
        retBool, rectmat1, rectmat2 = cv2.stereoRectifyUncalibrated(
            points1r, points2r, F, dim, H, H2)
    
        dst11 = cv2.warpPerspective(img, rectmat1, dim)
        dst22 = cv2.warpPerspective(img_3, rectmat2, dim)
        
        
        lines1 = cv2.computeCorrespondEpilines(img_points2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = drawlines(dst11, dst22, lines1, img_points1, img_points2)
    
        lines2 = cv2.computeCorrespondEpilines(img_points1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = drawlines(dst22, dst11, lines2, img_points2, img_points1)
    
        plt.figure(figsize=(16, 8))
        plt.subplot(121), plt.imshow(img5)
        plt.subplot(122), plt.imshow(img3)
        plt.show()
    
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=3,
            numDisparities=24,
            blockSize=18,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=30,
            # speckleWindowSize=0,
            # speckleRange=2,
            # preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(
            matcher_left=left_matcher)
        wls_filter.setLambda(80000)
        wls_filter.setSigmaColor(1.2)
    
        disparity_left = left_matcher.compute(dst11, dst22)
        disparity_right = right_matcher.compute(dst22, dst11)
        disparity_left = np.int16(disparity_left)
        disparity_right = np.int16(disparity_right)
        filteredImg = wls_filter.filter(disparity_left, img, None, disparity_right)
    
        depth_map = cv2.normalize(
            src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        depth_map = np.uint8(depth_map)
        # Invert image. Optional depending on stereo pair
        #depth_map = cv2.bitwise_not(depth_map)
        plt.imshow(depth_map, 'gray')
        plt.show()
        '''
        focal_lenght = 77.8
    
        # Perspective transformation matrix
        # This transformation matrix is from the openCV documentation, didn't seem to work for me.
        Q = np.float32([[1, 0, 0, -(w//4)/2.0],
                        [0, 1, 0, -(h//4)/2.0],
                        [0, 0, 0, focal_lenght],
                        [0, 0, -1, 0]])
    
        # This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision.
        # Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
        Q2 = np.float32([[1, 0, 0, 0],
                         [0, -1, 0, 0],
                         # Focal length multiplication obtained experimentally.
                         [0, 0, focal_lenght*0.05, 0],
                         [0, 0, 0, 1]])
        
        
        depth_map = np.invert(depth_map)
        plt.imshow(depth_map, 'gray')
        plt.show()
    
        # Reproject points into 3D
        points_3D = cv2.reprojectImageTo3D(depth_map, Q2)
        # Get color points
        colors = cv2.cvtColor(colored_resized, cv2.COLOR_BGR2RGB)
    
        # Get rid of points with value 0 (i.e no depth)
        mask_map = depth_map > depth_map.min()
        
    
        # Mask colors and points.
        output_points = points_3D[mask_map]
        print("3D points")
        print(output_points)
        output_points = output_points.reshape(-1, 3)
        output_colors = colors[mask_map]
        
        # Define name for output file
        output_file = 'cloud_bin_%d.ply' % count
    
        # Generate point cloud
        print("\n Creating the output file... \n")
        create_output(output_points, output_colors, output_file)'''
        
        '''
        background = cv2.imread("background.png",0)  # load background
        # resize image
        background = cv2.resize(background, dim, interpolation=cv2.INTER_AREA)
        depth_map = background - depth_map'''
        '''
        depth_map = np.invert(depth_map)
        plt.imshow(depth_map, 'gray')
        plt.show()'''
        
        depth_map = edge_detection(depth_map)
        plt.imshow(depth_map, 'gray')
        plt.show()
        
        
        focal_lenght = 77.8
        Cu = (w//4)/2
        Cv = (h//4)/2
        
        K = np.float32([[focal_lenght, 0.0,Cu],
                      [0.0, focal_lenght, Cv],
                      [0.0,0.0,1.0]])
        
        
        
        K_inv = np.linalg.inv(K)
        
        # Get pixel coordinates
        pixel_coords = pixel_coord_np(w//4, h//4)  # [3, npoints]
        
        # Apply back-projection: K_inv @ pixels * depth
        cam_coords = K_inv[:3, :3] @ pixel_coords * depth_map.flatten()
        
        cam_coords = cam_coords[:, np.where(cam_coords[2] <= 80)[0]]
        
        
        
        # Visualize
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cam_coords.T[:, :3])
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.io.write_point_cloud("cloud_bin_%d.pcd"%count, pcd)
        
        # Generate point cloud
        print("Creating the output file")
        print(count)
        print(pcd)
        
    
        count += 1
        j+=1
    i += 1


voxel_size = 0.5
max_correspondence_distance_coarse = voxel_size *15
max_correspondence_distance_fine = voxel_size * 1.5


def load_point_clouds(voxel_size=0.0):
    pcds = []
    for i in range(1,count):
        pcd = o3d.io.read_point_cloud("cloud_bin_%d.pcd" % i)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
        cl, ind = pcd_down.remove_radius_outlier(nb_points=16, radius=0.05)
        inlier_cloud = pcd_down.select_by_index(ind, invert=True)
        pcds.append(pcd_down)
    return pcds


def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target,
                                                        max_correspondence_distance_coarse, np.identity(4))
    print(evaluation)
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=True))
    return pose_graph


o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
pcds_down = load_point_clouds(voxel_size)
o3d.visualization.draw_geometries(pcds_down)


print("Full registration ...")
pose_graph = full_registration(pcds_down,
                               max_correspondence_distance_coarse,
                               max_correspondence_distance_fine)

print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
o3d.pipelines.registration.global_optimization(
    pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)

print("Transform points and display")
for point_id in range(len(pcds_down)):
    #print(pose_graph.nodes[point_id].pose)
    pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
o3d.visualization.draw_geometries(pcds_down)


print("Make a combined point cloud")
#pcds = load_point_clouds(voxel_size)
pcd_combined = o3d.geometry.PointCloud()
for point_id in range(len(pcds_down)):
    #pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    pcd_combined += pcds_down[point_id]
pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
#o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)
o3d.visualization.draw_geometries([pcd_combined_down])


poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_combined_down, depth=11, width=0, scale=2.1, linear_fit=True)[0]
bbox = pcd_combined_down.get_axis_aligned_bounding_box()
p_mesh_crop = poisson_mesh.crop(bbox)
dec_mesh = p_mesh_crop.simplify_quadric_decimation(10000)
dec_mesh.remove_degenerate_triangles()
dec_mesh.remove_duplicated_triangles()
dec_mesh.remove_duplicated_vertices()
dec_mesh.remove_non_manifold_edges()
dec_mesh.filter_sharpen()
o3d.io.write_triangle_mesh("p_mesh_c.ply", dec_mesh)
o3d.visualization.draw_geometries([dec_mesh])
vox_mesh = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_combined_down, 10.5)
o3d.visualization.draw_geometries([vox_mesh])
#o3d.io.write_triangle_mesh("vox_mesh.ply", vox_mesh)