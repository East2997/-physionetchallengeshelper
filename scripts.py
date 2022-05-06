# -*- ecoding: utf-8 -*-
# @ProjectName: python-classifier-2022-master
# @ModuleName: scripts
# @Function: 
# @Author: Eliysiumar
# @Time: 2022/4/30 17:30

# 一些小脚本
import random
import re, os
import shutil
from helper_code import find_patient_files


# 因为没有测试集，所以需要切分训练集
def split_train_test(
        raw_data_folder,
        target_train_folder = None,
        target_test_folder=None,
        test_proportion=0.15,  # 测试集比例
        is_move=True  # 是否移动测试集至测试集目录
):
    patient_files = find_patient_files(raw_data_folder)
    num_patient_files = len(patient_files)
    test_files = random.choices(patient_files, k=int(num_patient_files * test_proportion))
    test_idx_list = []
    for test_file in test_files:
        match = re.match("^.*?(\d*).txt$", test_file)
        test_idx = int(match.group(1))
        test_idx_list.append(test_idx)
    if is_move:
        if target_test_folder is None or target_train_folder is None:
            raise Exception("No target floder!")
        else:
            file_list = os.listdir(raw_data_folder)
            for f in sorted(file_list):
                match = re.match("^(\d*)(.*?)$", f)
                file_idx = int(match.group(1))
                if file_idx in test_idx_list:
                    shutil.copyfile(os.path.join(raw_data_folder, f), os.path.join(target_test_folder, f))
                else:
                    shutil.copyfile(os.path.join(raw_data_folder, f), os.path.join(target_train_folder, f))


if __name__ == '__main__':
    split_train_test("..\\data\\training_data", target_train_folder="..\\data\\split_train_data", target_test_folder="..\\data\\split_test_data")
