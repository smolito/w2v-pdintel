import fnmatch
import os
from get_project_root import root_path
import jiwer


project_root = root_path(ignore_cwd=False)
data_root = project_root + r"\PD_intelligibilityData"

print("data root:", data_root)

reference = "hello world"
hypothesis = "hello wool de be"

error = jiwer.wer(reference, hypothesis)

print(error)

"""
for directory in os.listdir(data_root):
    print("IM IN:" + directory)
    for folder in os.listdir(os.path.join(data_root, directory)):
        print(folder)
        for file in os.listdir(os.path.join(data_root, directory, folder)):
            if fnmatch.fnmatch(file, 'B*.wav'):
                print("^^^ here found: " + file)
            if fnmatch.fnmatch(file, 'PR1*.wav'):
                print("^^^ here found: " + file)

"""