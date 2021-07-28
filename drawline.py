import cv2


##length 150
def get_endpoint(theta, phi, center_x, center_y, length=300):
    endpoint_x = -1.0 * length * math.cos(theta) * math.sin(phi) + center_x
    endpoint_y = -1.0 * length * math.sin(theta) + center_y
    return endpoint_x, endpoint_y

# 그림 그리는거


label_txt = open("dataset/integrated_label(validation).txt" , "r")
labels = label_txt.readlines()
head_batch_label, gaze_batch_label = loadLabel_gazetest(labels,[filename])

##end 포인트 가져오는것
image = cv2.imread(os.path.join(args.im_path, image_file_name))
output_image = np.copy(color_img)

cv2.arrowedLine(output_image, (int(center_x), int(center_y)), (int(GT_endpoint_x), int(GT_endpoint_y)), (0, 255, 0), 2)
cv2.arrowedLine(output_image, (int(center_x), int(center_y)), (int(endpoint_x), int(endpoint_y)), (255, 0, 0), 2)

cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0]+str(people_count) + '_headpose.jpg'), output_image)

