import cv2
import glob
import os
import numpy as np

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        # Load all images from the folder
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(all_images)} Images for stitching in {path}')

        # Initialize list for homographies
        homography_matrix_list = []

        # Read the first image to initialize the panorama
        stitched_image = cv2.imread(all_images[0])
        h1, w1 = stitched_image.shape[:2]

        # Create a larger canvas for stitching images to avoid overflow
        canvas_width = 2 * w1
        canvas_height = 2 * h1
        panorama = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

        # Place the first image in the center of the canvas
        x_offset = canvas_width // 4
        y_offset = canvas_height // 4
        panorama[y_offset:y_offset+h1, x_offset:x_offset+w1] = stitched_image

        # Homography reference point
        previous_homography = np.eye(3)

        # Iterate over all subsequent images
        for i in range(1, len(all_images)):
            img1 = stitched_image  # current panorama
            img2 = cv2.imread(all_images[i])  # next image to warp

            # (1) Detect features and compute descriptors (e.g., using SIFT)
            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
            keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

            # (2) Match descriptors between two images
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)

            # (3) Apply ratio test to keep only good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # (4) Extract the matched keypoints
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            # Debugging output to check shapes
            print(f"src_pts shape: {src_pts.shape}, dst_pts shape: {dst_pts.shape}")

            # (5) Manually compute homography matrix
            homography_matrix = self.compute_homography(src_pts, dst_pts)
            homography_matrix_list.append(homography_matrix)

            # (6) Warp img2 to img1's perspective and blend them
            previous_homography = np.dot(previous_homography, homography_matrix)

            # Calculate the size of the warped image
            warped_img2 = cv2.warpPerspective(img2, previous_homography, (canvas_width, canvas_height))

            # Blend the images (can be enhanced later with blending techniques)
            panorama = cv2.addWeighted(panorama, 1, warped_img2, 1, 0)

        return panorama, homography_matrix_list

    def compute_homography(self, src_pts, dst_pts):
        """ Computes the homography matrix using DLT (Direct Linear Transformation) """
        # Ensure src_pts and dst_pts are (N, 2) arrays
        src_pts = np.asarray(src_pts).reshape(-1, 2)
        dst_pts = np.asarray(dst_pts).reshape(-1, 2)
        num_points = src_pts.shape[0]
        
        A = []
        for i in range(num_points):
            x, y = src_pts[i]
            xp, yp = dst_pts[i]

            # Two rows for each correspondence in matrix A
            A.append([-xp, -yp, -1, 0, 0, 0, x * xp, x * yp, x])
            A.append([0, 0, 0, -xp, -yp, -1, y * xp, y * yp, y])

        # Convert A to a numpy array and perform SVD
        A = np.array(A)
        U, S, Vh = np.linalg.svd(A)
        L = Vh[-1, :] / Vh[-1, -1]  # The solution is the last row of Vh, normalized

        # Reshape L into the homography matrix H
        H = L.reshape(3, 3)
        return H


        # Two rows for each correspondence in matrix A
        A.append([-xp, -yp, -1, 0, 0, 0, x * xp, x * yp, x])
        A.append([0, 0, 0, -xp, -yp, -1, y * xp, y * yp, y])    

        # Convert A to a numpy array and perform SVD
        A = np.array(A)
        U, S, Vh = np.linalg.svd(A)
        L = Vh[-1, :] / Vh[-1, -1]  # The solution is the last row of Vh, normalized

        # Reshape L into the homography matrix H
        H = L.reshape(3, 3)
        return H


