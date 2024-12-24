import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# 项目路径和输出路径
project_path = "../data/PROMISE/apache-ant-1.6.0"
output_path = "../dataset/PROMISE/apache-ant-1.6.0-zipTemp"
joern_cli_path = "../joern-cli"
graphml_output_path = "../dataset/PROMISE/apache-ant-1.6.0-graphml"

# 确保输出路径存在
os.makedirs(output_path, exist_ok=True)
os.makedirs(graphml_output_path, exist_ok=True)

# # 获取所有 Java 文件
# java_files = []
# for root, _, files in os.walk(project_path):
#     for file in files:
#         if file.endswith(".java"):
#             java_files.append(os.path.join(root, file))

# # 添加进度条，使用 tqdm 包裹 java_files 列表
# for java_file in tqdm(java_files, desc="Processing Java Files"):
#      # 构造输出文件名
#     sanitized_file_name = java_file.replace("/", "_").replace("\\", "_").replace(":", "_")
#     cpg_file = os.path.join(output_path, f"{sanitized_file_name}.cpg.bin.zip")

#     # 确保输出目录存在
#     os.makedirs(os.path.dirname(cpg_file), exist_ok=True)

#     # 构建 javasrc2cpg 命令
#     command = [
#         "D:\\CPDP\\joern-cli\\javasrc2cpg.bat", 
#         java_file,
#         "--output",
#         cpg_file
#     ]

#     try:
#         # 执行命令
#         subprocess.run(command, check=True)
#         print(f"Processing file: {java_file}")

#     except subprocess.CalledProcessError as e:
#         print(f"Error processing file: {java_file}")
#         print(e)


# 执行命令将 CPG 转换为 GraphML
for cpg_file in os.listdir(output_path):
    if cpg_file.endswith(".cpg.bin.zip"):
        cpg_path = os.path.join(output_path, cpg_file)
        graphml_file = os.path.join(graphml_output_path, cpg_file.replace(".cpg.bin.zip", ".graphml"))
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(graphml_file), exist_ok=True)

        command = [
            "D:\\CPDP\\joern-cli\\joern-export.bat", 
            "--repr=cpg14",
            # "--format=graphml",
            cpg_path,
            "--out",
            graphml_file,
        ]
        
        try:
            subprocess.run(command, check=True)
            print(f"Exported GraphML for {cpg_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error exporting GraphML for {cpg_file}: {e}")

print("All files processed!")

