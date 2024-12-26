import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# 项目路径和输出路径
project_path = "../../data/PROMISE/apache-ant-1.6.0"
output_path = "../../dataset/PROMISE/apache-ant-1.6.0-joern-parse-cpgs"
joern_cli_path = "../../joern-cli"
graphml_output_path = "../../dataset/PROMISE/apache-ant-1.6.0-graphml"
csv_output_path = "../../dataset/PROMISE/apache-ant-1.6.0-csv"
dot_output_path = "../../dataset/PROMISE/apache-ant-1.6.0-joern-parse-dot"
ast_output_path = "../../dataset/PROMISE/apache-ant-1.6.0-ast"

# 确保输出路径存在
os.makedirs(output_path, exist_ok=True)
os.makedirs(graphml_output_path, exist_ok=True)

def to_graphml():
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


def to_csv():
    # 执行命令将 CPG 转换为 csv
    for cpg_file in os.listdir(output_path):
        if cpg_file.endswith(".cpg.bin.zip"):
            cpg_path = os.path.join(output_path, cpg_file)
            csv_file = os.path.join(csv_output_path, cpg_file.replace(".cpg.bin.zip", ".csv"))
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)

            command = [
                "D:\\CPDP\\joern-cli\\joern-export.bat", 
                "--repr=all",
                "--format=neo4jcsv",
                cpg_path,
                "--out",
                csv_file,
            ]
            
            try:
                subprocess.run(command, check=True)
                print(f"Exported csv for {cpg_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error exporting csv for {cpg_file}: {e}")


def to_dot(output_path, ast_output_path):
    # 执行命令将 CPG 转换为 dot
    # 获取所有 .cpg.bin.zip 文件
    cpg_files = [f for f in os.listdir(output_path) if f.endswith(".cpg.bin.zip")]

    # 用 tqdm 包装文件列表，显示进度条
    for cpg_file in tqdm(cpg_files, desc="Exporting DOT files", unit="file"):
        if cpg_file.endswith(".cpg.bin.zip"):
            cpg_path = os.path.join(output_path, cpg_file)
            dot_file = os.path.join(dot_output_path, cpg_file.replace(".cpg.bin.zip", ""))
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(dot_file), exist_ok=True)

            command = [
                "D:\\CPDP\\joern-cli\\joern-export.bat", 
                "--repr=cpg14",
                cpg_path,
                "--out",
                dot_file,
            ]
            try:
                subprocess.run(command, check=True)
                print(f"Exported dot for {cpg_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error exporting dot for {cpg_file}: {e}")

to_dot(output_path, dot_output_path)
print("All files processed!")

