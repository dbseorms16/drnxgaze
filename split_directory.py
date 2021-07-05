import cv2
import os


root ='upsample_result'
dir_name = 'x4'
target_dir = os.path.join(root,dir_name)
file_list = os.listdir(target_dir)
result_base_path = "../rt_gene/rt_gene_model_training/rt_gene_dataset"

for i in range(len(file_list)):

    target_file_name = os.path.join(target_dir, file_list[i])
    split_file_name = file_list[i].split("_")

    move_dir = os.path.join(result_base_path, f'{split_file_name[0]}_glasses/inpainted/{split_file_name[1]}_LR')
    os.makedirs(move_dir, exist_ok=True)
    move_file_name = os.path.join(move_dir, file_list[i])

    x =cv2.imread(target_file_name)
    cv2.imwrite(move_file_name, x)
    print(file_list[i] + ' >> ' +move_file_name +'  Completed!')
