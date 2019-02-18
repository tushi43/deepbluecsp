import math

# take values angle_of_camera and vertical_distance from database
aov = 64.59985901018617
#62.99986399027697
angle_of_camera1 = aov/2
angle_of_camera2 = aov/2
vertical_distance = 119.38 # cm
# print("tan(x) in angle = ", math.tan(angle_of_camera/2))
# print("tan(x) in radians = ", math.tan(math.radians(angle_of_camera/2)))
distanceX = math.tan(math.radians(angle_of_camera1)) * vertical_distance
distanceY = math.tan(math.radians(angle_of_camera2)) * vertical_distance
total_distance = distanceY + distanceX
print("Total Distance = ", total_distance,"cm")

# take value from database
length_pixel = 4128
per_pixel = total_distance / length_pixel
print("measure per pixel =",per_pixel,"cm")

# main code for depth

depth_pixels = 9
depth_measure = depth_pixels * per_pixel

print("Depth of pothole =",depth_measure,"cm")



