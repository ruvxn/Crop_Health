import os

test_path = 'dataset/test'

if not os.path.exists(test_path):
    print("Error: Test directory does not exist!")
else:
    num_files = sum([len(files) for _, _, files in os.walk(test_path)])
    print(f"Test dataset contains {num_files} images.")
