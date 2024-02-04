---
runme:
  id: 01HNMA6NVTEZ7492KKJF4M78GP
  version: v2.2
---

# 目标检测改述

目标检测在CV中指的是`检测图中的某种物体的位置` + `标注出所检测到的物体的类别`.

## Anchors Base
![](notes_images/Anchor_base目标检测.png)
Anchor指的是`锚点`, Anchor base 就是通过锚点为基础构建不同大小的框, 然后对框里的东西进行检测&分类.
- Anchor可以在框的左上角, 也可以在框的中心

1. 为了使检测的成功率更高, 通常需要生成很多不同尺寸的框(人为设定固定的尺寸), 将框映射到图片(或者feature map)上, 然后再对框进行筛选, 筛选出框住物体面积最大的框出来.

2. 为了使框覆盖的更好而训练网络对 anchor box 进行调整(网络进行微调)
  - **本质上网络预测的是 anchor box 的偏移量(用于修正人为设定的框)**

Anchor base 的缺点:
- 生成的框太多, `计算量大`
- 在训练阶段, 与 ground truth 计算IoU(Intersection Over Union, 交并比)时, 通常需要`人为地进行正负样本的区分`(如: IoU > 0.7 为正样本, IoU < 0.3 为负样本, IoU处于0.3到0.7之前直接忽略), 这样会导致大量的负样本的不平衡, 从而导致`训练不平衡`.


## Anchor Free
最初的目的是解决 Anchor base 方法计算量大的问题而设计, 核心思想是: 不去生成大量框(只生成3个框, 然后回归出最准确的框), 而是直接预物体的中心点位置, 并预测框的高宽.
- 经典网络有 centerNet 和 cornerNet
  - centerNet 是预测中心点和高宽
  - cornerNet 则是预测框的左上角和右下角的坐标




# 回顾v1~v7
演变时间线
![](notes_images/YOLO_v1-v7_演变时间线.png)


## YOLO v1 -- Anchor Free
提出时间: 2015年.
注意: **YOLO v1 是 Anchor Free 的网络**, 它把Anchor的预测当做`回归`问题来处理.

对于网络结构部分:
![](notes_images/YOLOv1网络.png)
需注意:
- 由于输入层和最后的 FC(full connection) 层是固定的, 使得`输入的图片大小被固定`, 写死无法改变. 
  - 如果想要输入更大的图片, 那么整个网络都需要调整并重新训练

- 网络最后得到的 $7*7*30$ 指的是: 
  - 整个图片被分成了 $7*7$ 个网格(一共`49个anchor点`), 每个网格沿着通道方向可以取出一个**30维**的向量, 这30个数表示的是这个网格在 预测两个框, 每个框有 5 个坐标 $(X,Y,H,W,C)$ + 20 个类别的概率 + 1 个置信度
    - 坐标值里的 C 是confidence, 用来表征当前这个框里的东西是**前景还是背景**

对于Loss部分:
![](notes_images/YOLOv1_loss.png)
- 使用的loss主要是 softmax + cross entropy + MSE(mean square error)
  - 上面黄绿色的是第一部分 loss, 用来代`表框的定位loss`
  - 下面蓝色部分则是第二步 loss, 用来代表`类别的loss`
    - 类别分成两部分: 包含物体的类别和不包含物体的类别(2分类) + 物体所属的具体类别(多分类)


优缺点:
```txt
优点:
  1. 速度快, 达到 45fps
  2. end to end, 结构简单
  3. 可以检测多个物体

缺点:
  1. 定位不准, 定位精度不高
  2. 对小物体的检测效果不好
  3. 每个cell只能生成2个框, 并且只能检测1个具体类别.
      这是由最后输出的分类头决定的, 它的特征向量只能输出20个类别只中的1个类.
```

## YOLO v2 -- Anchor Base
提出时间: 2016年

改进目标: 解决 YOLOv1 定位精度差, 对小目标检测不友好的问题.

网络结构:
![](notes_images/YOLOv2_网络结构.png)
- 改进了backbone, 提出 `DarkNet19`, 速度没有变慢, 但是特征提取能力比V1更强

- 用**卷积层替代了 FC(Fully Connection) 层**, 使得**输入图片大小不再固定**(只要`输入图片尺寸是32的整数倍都可以`)

- 设计了 Neck 层: 在YOLOv1中, 它是直接一路卷积下来的, 没有进行多尺度的特征融合(v2对原图 1/16 的 feature map 和 1/32的 feature map 进行融合). 融合之后, 使用的是单一尺度接入任务的分类头.

- 在 Neck 层里使用了 `CBL模块`: 卷积和激活函数之间引入 `BN(Batch Normalization) 层`, 做的操作是: feature map每个元素减去均值, 再除以方差, 均值和方差是由当前的feature map计算出来的; 之后再对归一化后的feature map进行 平移和缩放(每个BN层包含两个可学习的参数 $\gamma$ 和 $\beta$ ).
    ![](notes_images/BN层使用的公式.png)
    - 加 BN 层可以使得网络收敛更快, 训练的时候梯度更稳定(因为调整了分布之后才丢进激活函数, 这样就不容易梯度爆炸以及梯度消失).
    - 参考博客: 
      - https://zhuanlan.zhihu.com/p/379721314
      - https://blog.csdn.net/weixin_42080490/article/details/108849715

- 分类头的输出中, 通道数是 3*(1+C+4), 3表示预测3个框; C在这里表示物体的具体类别, 也还是20类; 1 是confidence, 表示前景还是背景; 4则是框的坐标(Cx, Cy, H, W).
  - 这三个框(称之为`先验框`)的尺寸是从训练数据集得到的: 首先对数据集的 ground truth 框进行 `k-means` 聚类出`3个类`, 然后取出`这3个聚类中心的尺寸作为 anchor box 的尺寸`.

YOLOv2相比于v1的改进:
![](notes_images/YOLOv2相比V1的改进点.png)
```txt
1. Batch Normalization：
  提升模型收敛速度, 起到一定正则化效果，降低模型的过拟合

2. High Resolution Classifier：
  使用448*448的分类头

3. Convolutional With Anchor Boxes：
  用卷积替换全连接

4. New Backbone Network: 
  Darknet-19, 特征提取能力比 v1 强

5. Dimension Clusters：
  先验框生成策略, 对训练数据集的ground truth框进行k-means聚类得到3个先验框
  这么做可以提高召回率(higher recall)

6. Direct location prediction

7. Fine-Grained Features：
  多尺度融合, Neck层对 1/32 和 1/16 的 feature map 进行融合, 相比 v1 增加了细粒度的特征.

8. Multi-Scale Training：
  去掉FC后，可多尺度输入(将同一张图进行1.5和0.5倍的缩放之后, 也丢进网络进行训练, 相当于一种数据增强), 这样可以模拟出包含小目标的图片.
```

## YOLO v3 -- Anchor Base
提出时间: 2018年

![](notes_images/YOLOv3_网络结构.png)
- backbone使用 DarkNet-53
- Neck部分: 特征融合之后, 分别使用3个不同的尺度(下采样到原图 1/8, 1/16, 1/32 的大小)接入不同的预测头.
  - 对于这三个尺寸所接入的预测头: 每个预测头都分别预测3个框, 所以最终效果是针对图片的每一个Anchor, 一共会预测9个框.
    - 这么做的效果是: 在原图的同一个区域内(同一个区域指的是同一个Anchor, 这里的Anchor表示框的中心位置), 在这个区域里生成了更小的框和更大的框(也就是预测了不同的宽高), 这样可以更好地适应不同尺寸的物体. 比如图中$76*76*225$的tensor, 它主要负责预测**小物体**(整个图切出更多的格子, 检测粒度更细), 用于预测小目标的分支也被称为`P3`分支; 然后 $38*38*225$ 的tensor, 它主要负责预测**中等**大小的物体, 也被称为`p4`分支; 最后 $19*19*225$ 的tensor, 它主要负责预测**大物体**, 也被分成为`p5`分支.

- 网络使用的模块:
  - CBL: Convolution + Batch Normalization + Leaky ReLU, 和 V2 一样
  - ResUnit:  对CBL构建更大的模块, 并加入的残差连接
  - ResX: 在ResUnit前面再拼接一个CBL模块, 名字里的x表示有x个ResUnit, 如Res2表示有两个ResUnit

- 输出通道数255: 这个数字由 $3*(1+C+4)$ 计算而来; 3表示预测3个框; C在这里表示物体的具体类别, 由于训练使用的是**COCO数据集, 所以有80个类别**; 1 是confidence, 表示前景还是背景; 4则是框的坐标(Cx, Cy, H, W).

## YOLO v4 -- Anchor Base
提出时间: 2020年.

![](notes_images/YOLOv4_网络结构.png)
- SPP + PANet: SPP是空间金字塔池化, PANet是特征金字塔融合
  - SPP: 用来解决输入图片尺寸不固定的问题, 通过金字塔池化, 将不同尺寸的特征图融合到一起 (使用不同大小的池化核来获得高宽不同但通道数相同的feature)
    ![](notes_images/SPP模块图解.png)
      - 参考博客: https://blog.csdn.net/weixin_38145317/article/details/106471322
  - PANet: 用来解决多尺度特征融合的问题, 通过特征金字塔融合, 将不同尺度的特征图融合到一起

    ```txt
    在YOLOv3中, 进行特征融合的时候只有"从下到上"的特征融合, 但在YOLOv4中, 还进行了"从上到下"的特征融合, 这样可以更好地适应不同尺寸的物体.
    ```

- 模块:
  ```txt
  CBL: 
    Convolution + Batch Normalization + Leaky ReLU
  CBM: 
    Convolution + Batch Normalization + Mish
    Mish也是一种激活函数
  ResUnit: 
    用CBL和残差连接构建的模块, 和 YOLOv3 一样
  CSPn: 
    利用CBM, CBL 和 ResUnit构建的模块, 并加入了残差连接
    n表示有n个CBM模块
  ```

- 加入新的数据增强方式
  - Mixup, Cutmix, Mosaic
    - Mixup: 将两张图片按照一定的比例混合在一起, 使得网络更加robust
    - Cutmix: 将两张图片按照一定的比例混合在一起, 并且将其中一张图片的一部分挖掉, 然后将另一张图片的一部分填充进去
    - Mosaic: 将四张图片按照一定的比例混合在一起, 使得网络更加robust
  - 参考博客: https://blog.csdn.net/u011984148/article/details/107572526


## YOLO v5 -- Anchor Base
提出时间: 2020年. (与V4同年)
![](notes_images/YOLOv5_网络结构.png)

Backbone: 
- 使用Fucus模块, 进行下采样
- 后面分支分别得到原图的 1/8, 1/16, 1/32 的 feature map, 然后分别接入Neck模块的不同层

Neck:
- 就是简单的模块拼凑+特征融合

Head:
- 1/8, 1/16, 1/32 的 feature map 分别接入不同的预测头, 每个预测头都预测3个框, 所以最终效果是针对图片的每一个Anchor, 一共会预测9个框.

模块:
- CBL, 与v3, v4一样
- CSP模块:
  - 有两种, 构造方式与 V4 不同. 一种有残差, 一种没有残差.
- Focus:
  - 通过切片的方式, 将输入的输入tensor切片成四份(**偶数行+偶数列, 偶数行+奇数列, 奇数行+偶数列, 奇数行+奇数列**), 然后将这四份按顺序concate到通道上, 此时通道扩大4倍, 宽高各减少2倍, 这样可以减少计算量, 但是不会丢失信息. (直观地看, 实现了两倍下采样)
  ![](notes_images/YOLOv5_Focus模块细节.png)
  - 参考博客: https://blog.csdn.net/qq_39056987/article/details/112712817


### YOLOv5 的 v6.0版本
![](notes_images/YOLOv5的v6.0版本.png)

```txt
2021年10月12日，yolov5 发布了 V6.0 版本，改进：
  - 引入了全新的更小( Nano )的模型 P5(YOLOv5n) 和 P6(YOLOv5n6)，总参数减少了 75%，从 7.5M 缩小到了 1.9M，非常适合于移动端或者是 CPU 的环境

  - 用 Conv(k=6, s=2, p=2) 代替 Focus 层，主要是为了方便模型导出

  - 使用 SPPF 代替 SPP 层

  - 减少 P3 主干层 C3

  - 将 SPPF 放在主干的后面

  - 在最后一个 C3 主干层中重新引入shortcut 

  - 更新超参数
```

## YOLOX -- Anchor Free
提出时间: 2021年.

![YOLOX](notes_images/YOLOX_网络结构.png)
Backbone: 和 YOLOv5 差不多, 都有Focus模块, CBL模块, CSP模块

Neck: "从上到下" 与 "从下到上" 的特征融合, 中间再将输入concatenate (当前的目标检测都用这种方式)

Head(YOLOX不一样的地方)
- YOLO v1~v5, 预测头都是将3个任务(区分前景背景+定位+物体分类)合并在一个head上, YOLOX则将它们分开了(解耦)
- 注意在输出时, **有些边缘计算的框架可能支持不是那么好**, 此时在工程上会将网络止步于decouple head 部分, 后面的 concatenate 和 reshape 当做 `后处理`来做.


![](notes_images/YOLOX_backbone细节.png)
- 不同大小的网络主要通过调整backbone来实现: `Backbone`里**CSP模块中的BottleNeck的重复次数**可以控制网络的大小. (网络结构的左下角, CSP3模块里粉色的BottleNeck的数量)
- 输出通道数分别对应P3(小目标), P4(中等目标), P5(大目标)的预测头输出通道数


## YOLO v6 -- Anchor Free
美团提出的目标检测算法, 2022年.

![](notes_images/YOLOv7_网络结构.png)
美团官方解读博客: https://mp.weixin.qq.com/s/RrQCP4pTSwpTmSgvly9evg

模块:
- CB: CB表示`Conv+BN`, 使用不同的大小卷积核的CB模块组成, 底下标注的数字为**卷积核大小**
- Rep-I: 卷积核大小为(1,2) 和 (3,2), 卷积之后相加, 再接一个ReLU
- Rep-II: 卷积核大小为(1,1) 和 (3,1),  卷积之后相加, 再接一个ReLU
- Rep-III: 卷积核大小为(1,1) 和 (3,1) 并加上一个BN,  之后相加, 再接一个ReLU
- RVB1_x: RVB表示`Rep VGG Block`, 由不同的Rep子模块构成.
- SPPF: 先进行卷积(CBL模块), 然后按照图片中的形式**连续进行池化**, 最后再进行卷积, 用来提取不同尺度的特征
  - SPP是没有进行卷积的, 而是**分别**进行`直接的池化`并concatenate.

Neck部分:
![](notes_images/YOLOv6_neck部分.png)
- 整体上逻辑与 v4, v5 一样, 都是自顶向下的特征融合与自底向上的特征融合都做, 中间层进行了连接; 但是这里不同的是没有使用CSP模块, 而是使用 RepBlock.

Head部分:
![](notes_images/YOLOv6_head部分.png)
- 与YOLOX很像, 都是解耦头.
- head分成3个任务: Cls预测当前区域是前景或者背景; Reg预测框的位置; Obj预测物体的类别. 

小结:
```txt
设计了更高效的 Backbone 和 Neck ：
  受到硬件感知神经网络设计思想的启发，基于 RepVGG style 设计了可重参数化、更高效的骨干网络 EfficientRep Backbone 和 Rep-PAN Neck。

优化设计了更简洁有效的 Efficient Decoupled Head，在维持精度的同时，进一步降低了一般解耦头带来的额外延时开销。

在训练策略上:
  采用 Anchor-free 范式，同时辅以 SimOTA 标签分配策略以及 SIoU 边界框回归损失来进一步提高检测精度。
```






## YOLO v7
提出时间: 2022年.


