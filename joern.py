import subprocess
import os
import shutil
import pandas as pd
from tqdm import tqdm

# path to joern-parse
JOERNPATH = "./joern-cli"
root_dir = "./dataset/"

# location if source code dirs
source_dir = "./data/PROMISE/src"

def parse_source_code_to_dot(file_path, f, out_dir_pdg='/parsed/dot/pdg', out_dir_cpg='/parsed/dot/cpg'):
    root_path = "./dataset/"
    try:
        os.makedirs(root_path+out_dir_pdg)
        os.makedirs(root_path+out_dir_cpg)
    except:
        pass
    out_dir_cpg = root_path + out_dir_cpg

    # parse source code into cpg
    print("parsing source code into cpg...")
    shell_str = "sh" + JOERNPATH + "./joern-parse" + file_path
    subprocess.call(shell_str, shell=True)
    print("exporting cpg from cpg root...")
    