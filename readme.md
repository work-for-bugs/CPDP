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

