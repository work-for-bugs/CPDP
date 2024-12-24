# 将每个代码文件处理成cpg
```
cd ./code
python code2cpg.py
```

# 将生成cpg导出
```
python cpg_Export.py
```

# 写一个bash，用上述方式处理所有项目

# joern的输出形式是？如何把joern的输出构建成图？
# 需要什么样的数据？每个方法都有标签？

# 需要导出到neo4j吗？

# word2vec 和 处理成图结构 的先后顺序？

# 将cpg处理成图结构，以供图卷积

# 将代码转成向量

# 输入到GCN-输入格式？

# 模型训练

# 怎么测试？




进入joern-cli目录
```
./joern 
```

使用 importCode 命令导入 Java 项目到 Joern
```
importCode("../data/PROMISE/apache-ant-1.6.0")
```

将 CPG 导出并可视化



step1：生成CPG
生成整个项目的CPG，会生成一个包含整个项目的CPG文件
```
javasrc2cpg --output /path/to/project.cpg.bin.zip /path/to/java/project
```

step2：在Joern中加载CPG
```
importCpg("/path/to/project.cpg.bin.zip")
```

step3：按文件提取CPG子图
获取所有文件
```
val files = cpg.file.name.l
```
为每个文件导出CPG
```
files.foreach { fileName =>
  val sanitizedFileName = fileName.replace("/", "_").replace("\\", "_")
  val outputPath = s"./output/${sanitizedFileName}.graphml"
  cpg.file.name(fileName).toGraphML(outputPath)
}
```

