import os
from pathlib import Path
import subprocess
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('type', choices=['training', 'validation', 'testing'])
parser.add_argument('--out-dir', default='./dataset')
parser.add_argument('--no_segments', type=int, default = 0, help="Number of downloaded segments")
args = parser.parse_args()
print("Python Version is {}.{}.{}".format(sys.version_info[0], sys.version_info[1], sys.version_info[2]))
if sys.version_info[1] <= 6:
    from subprocess import PIPE
    p1 = subprocess.run("gsutil ls -r gs://waymo_open_dataset_v_1_2_0_individual_files/{type}".format(type=args.type), shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True)
else:
    p1 = subprocess.run("gsutil ls -r gs://waymo_open_dataset_v_1_2_0_individual_files/{type}".format(type=args.type), shell=True, capture_output=True, text=True)
if p1.returncode != 0:
    raise Exception("gsutil error, check that gsutil is installed and configured correctly according to README file !", p1.stderr)
download_list = p1.stdout.split()

print("Creating/accessing dataset folder...")

data_path = os.path.join(os.path.dirname( __file__ ), '..', args.out_dir, args.type)
# create the main dataset and data split directories if not found
Path(data_path).mkdir(parents=True, exist_ok=True)

no_downloads = args.no_segments+1 if args.no_segments != 0 else len(download_list)
for i in range(1, no_downloads):
    print("Downloading segment {}".format(i))
    p2 = subprocess.run(" ".join(["gsutil cp", download_list[i], data_path]), shell=True)
      
