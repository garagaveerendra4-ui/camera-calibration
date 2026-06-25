import glob
import numpy as np
import cv2
#from save_calibration import save_calibration

np.set_printoptions(suppress=True, precision=4) 

square_size=55
# Checkerboard size
CHECKERBOARD = (9,6)

# Stop criteria
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

# Real world points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(
    -1, 2
)
objp=objp * square_size

# Arrays
objpoints = []
imgpoints = []

# Load checkerboard images
images = glob.glob("/home/user1/calibration validation/images/*.jpg")
if len(images) == 0:
    raise SystemExit(
        "No images found in the 'images' folder. Add .jpg images and retry."
    )

image_size = None

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: couldn't read {fname}, skipping.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]

    # Find corners (use common flags to improve detection)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD,flags)

    if ret:
        objpoints.append(objp.copy())

        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

if len(objpoints) == 0:
    raise SystemExit(
        "No checkerboard corners were found. Check images or adjust CHECKERBOARD."
    )


# Calibration
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

print("camera matrix=",cameraMatrix)
print("distortion coefficieng=",distCoeffs)
# Reprojection Error

total_reprojection_error = 0

for i in range(len(objpoints)):

    imgpoints2, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        cameraMatrix,
        distCoeffs
    )

    image_reprojection_error = (
        cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        / len(imgpoints2)
    )

    total_reprojection_error += image_reprojection_error

average_reprojection_error = (
    total_reprojection_error / len(objpoints)
)

print(
    "\nTotal Reprojection Error:",
    average_reprojection_error
)