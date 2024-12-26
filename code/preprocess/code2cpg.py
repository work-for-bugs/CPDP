import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# 项目路径和输出路径
project_path = "../../data/PROMISE/apache-ant-1.6.0/src/main/org/apache/tools" 
output_path = "../../dataset/PROMISE/apache-ant-1.6.0-joern-parse-cpgs" # bin 和 zip 都可以
joern_cli_path = "../../joern-cli"

# 确保输出路径存在
os.makedirs(output_path, exist_ok=True)

# 获取所有 Java 文件
java_files = []
for root, _, files in os.walk(project_path):
    for file in files:
        if file.endswith(".java"):
            java_files.append(os.path.join(root, file))

# 添加进度条，使用 tqdm 包裹 java_files 列表
for java_file in tqdm(java_files, desc="Processing Java Files"):
    # 只保留文件路径的最后两级目录和文件名
    relative_path = os.path.relpath(java_file, project_path)  # 从项目根目录开始的相对路径
    file_name = os.path.basename(relative_path)
    # 构造输出文件名
    cpg_file = os.path.join(output_path, f"{file_name}.cpg.bin.zip")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(cpg_file), exist_ok=True)

    # 构建 javasrc2cpg 命令
    command = [
        "D:\\CPDP\\joern-cli\\joern-parse.bat", # 这里要用绝对路径
        java_file,
        "--output",
        cpg_file
    ]

    try:
        # 执行命令
        subprocess.run(command, check=True)
        print(f"Processing file: {java_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error processing file: {java_file}")
        print(e)


print("All files processed!")

