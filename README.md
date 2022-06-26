**项目简介**

采用双目视觉，测量被摄物体的尺寸，如高、宽等（也可以用来测距，但需要对代码稍作改动）。

**项目文件说明**

- imgs: 双目相机采集的棋盘格（每个格子边长1厘米）图片
- m.py: 直接`python m.py`执行就好，需要准备python3+opencv的环境
- melonL.jpg/melonR.jpg: 被摄物体的双目图像

**使用演示**

1. 使用`python m.py`命令，会出现depth、rectify两个窗口
2. 调整depth窗口的参数，这里默认不动就可以
3. 在rectify竖线右侧左目图像部分，摁下鼠标选择起始点，松开鼠标选择终止点
4. 控制台窗口会显示起始点和终止点的坐标和距离

![](https://github.com/suqinglee/binocular-vision-measurement/blob/master/imgs/result.jpg)