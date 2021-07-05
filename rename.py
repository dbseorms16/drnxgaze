import cv2
import os


dataset_path = "../rt_gene/rt_gene_model_training/rt_gene_dataset"
# left_patch_path  ="inpainted/left"
# right_patch_path  ="inpainted/right"
face_image_path = "inpainted/face"
file_type = '.jpg'
result_path = "./dataset/benchmark/face_test/HR"
os.makedirs(result_path, exist_ok=True)

for i in range(14,17):
    sub_path = os.path.join(dataset_path, f's{i:0>3}_glasses')

    # left_file_path = os.path.join(sub_path, left_patch_path)
    # right_file_path = os.path.join(sub_path, right_patch_path)
    face_file_path = os.path.join(sub_path, face_image_path)

    # left_file_list = os.listdir(left_file_path)
    # right_file_list = os.listdir(right_file_path)
    face_file_list = os.listdir(face_file_path)

    for num in range(len(face_file_list)):
        # left_file_name = os.path.join(left_file_path, left_file_list[num])
        # right_file_name = os.path.join(right_file_path, right_file_list[num])
        face_file_name = os.path.join(face_file_path, face_file_list[num])
        


        # left = cv2.imread(left_file_name)
        # right = cv2.imread(right_file_name)
        face = cv2.imread(face_file_name)

        # cv2.imwrite(os.path.join(result_path, f's{i:0>3}_'+left_file_list[num]), left)
        # cv2.imwrite(os.path.join(result_path, f's{i:0>3}_'+right_file_list[num]), right)
        print(face_file_list[num])
        cv2.imwrite(os.path.join(result_path, f's{i:0>3}_'+str(int(face_file_list[num].split('_')[1]))+file_type), face)
        # # os.rename(file_name, os.path.join(result_path,split[0]+file_type))
        print(face_file_list[num] + ' >> ' + '  Completed!')
