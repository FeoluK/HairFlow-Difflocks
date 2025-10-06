#!/usr/bin/env python3

#given the difflocks dataset downloaded from mpi uncompress everything

# ./uncompress_data.py --dataset_zipped_path <FOLDER_WITH_ALL_THE_ZIP_FILES> --out_path <DATASET_PATH>



import sys
import os
import argparse
import subprocess
from os import listdir
from os.path import isfile, join


def main():

    #argparse
    parser = argparse.ArgumentParser(description='Uncompress dataset')
    parser.add_argument('--dataset_zipped_path', required=True, help='Path to the difflocks folder containing all zip files')
    parser.add_argument('--out_path', required=True, type=str, help='Where to output the difflocks dataset')
    args = parser.parse_args()


    in_path=args.dataset_zipped_path
    onlyfiles = [f for f in listdir(in_path) if isfile(join(in_path, f))]
    for file_name in onlyfiles:
        filepath=os.path.join(in_path,file_name)
        cmd=["7zz","x", "-y", filepath, "-o"+args.out_path]
        # print("cmd",cmd)
        subprocess.run(cmd, capture_output=False)
        print("filename", file_name)
    

if __name__ == '__main__':
    main()
    
