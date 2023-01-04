import numpy as np
import cv2


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only


def color_thresh(img, rgb_thresh=(160,160,160),types=0):
    color_select =np.zeros_like(img[:,:,0])
    if types == 0:
        above_thresh = (img[:,:,0]>rgb_thresh[0])&(img[:,:,1]>rgb_thresh[1])&(img[:,:,2]>rgb_thresh[2])  # for navigable
    elif types == 1:
        above_thresh = (img[:,:,0]>rgb_thresh[0])&(img[:,:,1]>rgb_thresh[1])&(img[:,:,2]<rgb_thresh[2])#for rocks
    else:
        above_thresh = (img[:,:,0]>0)&(img[:,:,1]>0)&(img[:,:,2]>0)#for mask
    color_select[above_thresh] = 1
    return color_select


# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1] / 2).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel ** 2 + y_pixel ** 2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated


def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos #to get the translated rover centric coordinates, we have to first divide by scale
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world


# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))  # keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:, :, 0]), M, (img.shape[1], img.shape[0]))
    return warped, mask

def perspect_transform2(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))  # keep same size as input image
    return warped
def impose_range(xpix, ypix, range=80):
    dist = np.sqrt(xpix ** 2 + ypix ** 2)
    return xpix[dist < range], ypix[dist < range]
# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO:
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform


    dst_size = 5 #used a destination size of 5, for better fidelity, this does mess with collisions, need optimal number for phase 2
    bottom_offset = 6 #offset of the warped image from the bottom
    scale = 2 * dst_size  # used a scale of destination size*2, the scale does help in improving fidelit, however at the cost of collisions, this is the size to be converted from when setting to world map
    image = Rover.img #get the rover image to do transformations on
    source = np.float32([[14, 140], [301, 140], [200, 96], [118, 96]]) #hardcoded numbers for phase 1, phase 2 this has to be automated
    destination = np.float32([[image.shape[1] / 2 - dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - bottom_offset],
                              [image.shape[1] / 2 + dst_size, image.shape[0] - bottom_offset - 2 * dst_size],
                              [image.shape[1] / 2 - dst_size, image.shape[0] - bottom_offset - 2 * dst_size] ])

    # 2) Apply perspective transform

    warped, mask = perspect_transform(image, source, destination)
    Rover.warped = warped
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples

    threshed = color_thresh(warped)
    #obstacleMap = np.absolute(np.float32(threshed) - 1) * color_thresh(warped, types=2)#had two approaches to get the mask, first is the method of just multiplying any nonzero positive value by 1 manually
    obstacleMap = np.absolute(np.float32(threshed) - 1) * mask # 2nd is by using numpy non-zero method
    lower_yellow = np.array([24 - 5, 100, 100])
    upper_yellow = np.array([24 + 5, 255, 255])
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Threshold the HSV image to get only upper_yellow colors
    rockMap = cv2.inRange(hsv, lower_yellow, upper_yellow)
    rockMap = perspect_transform2(rockMap, source, destination)




    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    # Example: Rover.vision_image[:,:,0] =f obstacle color-thresholded binary image
    #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
    #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary imagepitch
    #if((Rover.pitch <1 and Rover.pitch > 359) and (Rover.roll < 1 and Rover.roll>359)):

    Rover.vision_image[:, :, 2] = threshed * 255 #setting the blue channel to be navigable terrain
    #Rover.navigable_thresh_image[:,:,2]= threshed*255 # seperating the navigable for debugging
    Rover.vision_image[:, :, 1] = rockMap * 255#setting the green channel to be rocks
    #Rover.rock_thresh_image[:,:,1] = rockMap * 255 #seperating the rock for debugging
    Rover.vision_image[:, :, 0] = obstacleMap * 255 #setting the red channel to be obstacles, this will be the most dominant in the vision image
    #Rover.obstacle_thresh_image[:,:,0] = obstacleMap * 255 #seperating the obstacles for debugging
    idx = np.nonzero(Rover.vision_image)
    Rover.vision_image[idx] = 255




    # 5) Convert map image pixel values to rover-centric coords

    xpix, ypix = rover_coords(threshed)
    rock_x, rock_y = rover_coords(rockMap)  # set the coordinates of the rock
    obstaclexpix, obstacleypix = rover_coords(obstacleMap)
    xpix, ypix = impose_range(xpix, ypix)
    obstaclexpix, obstacleypix = impose_range(obstaclexpix, obstacleypix)


    # 6) Convert rover-centric pixel values to world coordinates

    world_size = Rover.worldmap.shape[0]
    x_world, y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
    obstacle_x_world, obstacle_y_world = pix_to_world(obstaclexpix, obstacleypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale) #setting obstacles in world map
    rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)# transform it to world coordinates




    # 7) Update Rover worldmap (to be displayed on right side of screen)
    # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
    #          Rover.worldmap[rock_y_world, rock_x_world, 1] +S= 1
    #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    #rock_xcen = rock_x_world[rock_idx]  # set the centers
    #rock_ycen = rock_y_world[rock_idx]
    if (Rover.pitch < 0.4 or Rover.pitch > 359.6) and (Rover.roll < 0.4 or Rover.roll > 359.6):
        Rover.worldmap[y_world, x_world, 2] += 1 #set that we found navigable terrain in the world, color is dominant in world map
        Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1  #set that we found obstacles in map, just a bit lighter
        Rover.worldmap[rock_y_world, rock_x_world, 1] = 255  # set the green channel on the worldmap with the location of the rock
    Rover.worldmap = np.clip(Rover.worldmap, 0, 255)
    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
    # Rover.nav_dists = rover_centric_pixel_distances
    # Rover.nav_angles = rover_centric_angles

    dist, angles = to_polar_coords(xpix, ypix)
    rock_dist, rock_ang = to_polar_coords(rock_x, rock_y)
    Rover.nav_angles = angles #set the rover angles
    Rover.nav_dists = dist #set the rover distances, this can be seen in the pipeline, and how it works
    Rover.samples_angle = rock_ang
    Rover.samples_dist = rock_dist
    if Rover.start_pos is None:
        Rover.start_pos = (Rover.pos[0], Rover.pos[1])
        print('STARTING POSITION IS: ', Rover.start_pos)

    return Rover