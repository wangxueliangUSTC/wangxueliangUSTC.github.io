---
layout: article
title: 《机器学习实战》学习总结（五）——Logistic回归
aside:
  toc: true
tags: ['机器学习','统计学习方法']
---

<!--more-->

> ###### *这篇文章转载自[我在知乎的文章](https://zhuanlan.zhihu.com/p/30347326)*

## **摘要**

1. Logistic回归分类

2. 梯度下降法

3. 代码实现与解释

## **Logistic回归**

逻辑斯特回归（logistic regression）是一种非常经典的分类方法。其用到的分类函数一般为Sigmoid函数，其函数形式为：

![img](https://pic1.zhimg.com/v2-25fb8958f26c8798bef5eca2527cdebc_b.jpg)

其图形表示如下：

![img](https://pic4.zhimg.com/v2-876cb2f9e1bc0229245fb3f58ca0451b_b.jpg)

从图中我们可以看到，当z=0时，函数值为0.5。随着z值的增加，对应的函数值将逼近于1；随着z值的减小，函数值将逼近于0。

因此为了实现logistic回归，对于样本，我们可以在每个特征上乘以一个回归系数，然后将所有的结果值相加，将总和带入sigmoid函数中，进而得到一个范围在0~1之间的数值。大于0.5的数据被分到1类，小于0.5的即被分到0类。

### **二分类问题**

对于Logistic回归中的二分类问题，当我们给定数据样本x时，其被分到1类和0类的条件概率分别为：

![img](https://pic4.zhimg.com/v2-41401bfe615c050acee78c6adf8ea32f_b.jpg)

其中，

![img](https://pic2.zhimg.com/v2-a5c43bfe15efe302ef87623c91b76fe9_b.jpg)

那么上式就可改写为：

![img](https://pic3.zhimg.com/v2-ef5a4b1fb6ff23fcfde79b95cfb1aba6_b.jpg)

我们现在需要做的，就是怎么来确定模型中的参数w呢？

### **模型参数估计**

在Logistic回归参数学习中，对于给定的训练数据集T=｛(x1,y1),(x2,y2),...(xN,yN)｝，我们用极大似然估计法估计模型参数w，从而得到logistic回归模型。

![img](https://pic4.zhimg.com/v2-57be4b250fed08675b90425ceb0ed777_b.jpg)

当我们对L(w)求得极大值，也就得到了参数w的估计值。

这样，问题就变成了以对数似然估计为目标函数的最优化问题。而我们解决这个问题我们一般采用梯度下降法和拟牛顿法。

### **多分类问题**

当然，上述的logistic模型同样可以推广到多分类问题。设Y的取值为｛1，2，3,...K｝，那么回归模型即为：

![img](https://pic4.zhimg.com/v2-122a97354cf204bfdeb85ebbce2d2c0b_b.jpg)

## **梯度下降法**

梯度下降法可以解决上面对于参数w的优化问题。

梯度下降法是一种迭代算法，通过选取适当的初始值，不断迭代，对参数值不断更新，进行目标函数的极小化，直到收敛。由于负梯度方向是函数值减小最快的方向，所以在迭代的每一步，我们向负梯度方向更新参数值，从而减小函数值。

梯度算法的迭代公式为：

![img](https://pic1.zhimg.com/v2-39f455b3cf0732c6bb53f19bff1364f0_b.jpg)

其中：

![img](https://pic4.zhimg.com/v2-c5fbd11dd0b741cfeb9621add2fb7a83_b.jpg)

*同理，梯度上升法，就是向梯度方向移动，以求得函数的极大值。*

### **随机梯度下降法**

在实现梯度下降法时，我们发现在进行梯度下降法的回归系数更新时需要遍历整个数据集，如果样本或者特征过多的话，这种方法计算复杂度太高。

有一种改进的方法，就是一次我们只随机用一个样本点来更新回归系数，那么该方法就叫做随机梯度下降法。

这两种算法在下面的代码部分都有实现，读者可以自行参考。

*求解回归系数w的方法除了上面提到的梯度下降法，还有拟牛顿法，想要详细了解的同学可以见《统计学习方法》附录B。*

上面一部分，我们系统地从分类函数讲到怎么确定优化目标函数，再到怎么解优化目标函数。

至此我们就完成了Logistic回归原理部分的学习。下面是代码部分。

## **代码实现**

1.回归梯度上升优化算法

程序清单：

```python
from numpy import *
# 打开文本文件函数
def loadDataSet():
    dataMat=[];labelMat=[]
    # 打开文本文件
    fr=open(r'C:\\Users\Administrator\Desktop\MliA\MLiA_SourceCode
\machinelearninginaction\Ch05\testSet.txt')
    # 逐行读取
    for line in fr.readlines():
        # 对文本进行处理,处理为一个列表
        lineArr=line.strip().split()
        # 加到dataMat中，并把第一个值设为1.0
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        # 求得类别标签
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

# 利用sigmoid函数进行计算
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

# 梯度上升算法
# dataMatIn里存放的是特征，但是第一列都为1.0，实际上为100*3的矩阵
# classLabels里存放的是类别标签，1*100的行向量
def gradAscent(dataMatIn,classLabels):
    # 转换为Numpy矩阵类型
    dataMatrix=mat(dataMatIn)
    # 转化为矩阵类型并求转置
    labelMat=mat(classLabels).transpose()
    # 求得矩阵大小
    m,n=shape(dataMatrix)
    # alpha是目标移动的步长
    alpha=0.01
    # 设置迭代次数
    maxCycle=500
    # 权重初始化为1
    weights=ones((n,1))
    for k in range(maxCycle):
        # 注意，这里h是一个m*1的列向量
        h=sigmoid(dataMatrix*weights)
        # 求得误差
        error=(labelMat-h)
        # 更新权重
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights
```

2.画出数据集最佳拟合直线

程序清单：

```python
# 作图
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    # 将dataMat类型变成数组
    dataArr=array(dataMat)
    # 得到数据的样本数
    n=shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    # 将样本分成两类，放到列表中
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    # 两个种类用不同的颜色表示
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    # 标注X轴的范围与步长
    x=arange(-3.0,3.0,0.1)
    # 表示出分界线的方程
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    # 坐标名称
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()
```

3.随机梯度上升算法

程序清单:

```python
# 随机梯度上升算法
# 这里的程序与梯度上升相差不大，唯一的区别是这里的h、error都是值，不是矩阵
def stocGraAscent0(dataMatrix,classLabels):
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights
```

4.改进的随机梯度上升算法

程序清单：

```python
# 改进的随即梯度上升算法
def stocGraAscent1(dataMatrix,classLabels,numIter=150):
    m,n=shape(dataMatrix)
    alpha=0.01
    weights=ones(n)
    for j in range(numIter):
        dataIndex=list(range(m))
        for i in range(m):
            # 每次调整alpha
            alpha=4/(1.0+j+i)+0.01
            # 随机选取样本来更新回归系数
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex]
            # 删除已经使用过的样本
            del(dataIndex[randIndex])
    return weights
```

5.用Logistic回归进行分类

程序清单：

```python
#sigmoid()分类函数
def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    # 值如果大于0.5，归为1.0类
    if(prob>0.5):return 1.0
    else:return 0.0


def colicTest():
    # 打开训练集合
    frTrain=open(r'C:\\Users\Administrator\Desktop\MliA'
                '\MLiA_SourceCode\machinelearninginaction\Ch05\
                  horseColicTraining.txt')
    # 打开测试集合
    frTest=open(r'C:\\Users\Administrator\Desktop\MliA'
                '\MLiA_SourceCode\machinelearninginaction\Ch05\horseColicTest.txt')
    # 初始化训练集和标签的列表
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        # 对训练集的数据格式化处理
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            # 将每一行的特征数据放到lineArr中
            lineArr.append(float(currLine[i]))
        # 再将lineArr作为列表放到trainingSet中
        trainingSet.append(lineArr)
        # 将标签放到trainingLabels中
        trainingLabels.append(float(currLine[21]))
    # 得到训练集的权重
    trainWeights=stocGraAscent1(array(trainingSet),trainingLabels,500)
    errorCount=0;numTestVec=0.0
    # 对测试集进行测试
    for line in frTest.readlines():
        # 计算测试集的个数
        numTestVec+=1.0
        # 对测试数据进行格式化处理
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        # 如果学习出来的结果和真实结果不一致，则错误数加一
        if(int(classifyVector(array(lineArr),trainWeights))!=int(currLine[21])):
            errorCount+=1
        # 计算错误率
        errorRate=(float(errorCount)/numTestVec)
        print('the error rate of this test is %f'%errorRate)
        return errorRate
# 调用colicTest()函数多次，计算错误率的平均值
def multiTest():
    numTests=10
    errorSum=0.0
    for k in range(numTests):
        errorSum+=colicTest()
    print('after %d iterations the arrange error rate is:%f'
            % (numTests,errorSum/float(numTests)))
```

至此，我们完成了Logistic回归分类的原理和代码部分的学习。个人建议推导部分最好自己手动推导一遍，会有效加深自己的理解。下一节学习支持向量机。

## **声明**

最后，所有资料均本人自学整理所得，如有错误，欢迎指正，有什么建议也欢迎交流，让我们共同进步！转载请注明作者与出处。

以上原理部分主要来自于《机器学习》—周志华，《统计学习方法》—李航，《机器学习实战》—Peter Harrington。代码部分主要来自于《机器学习实战》，文中代码用Python3实现，这是机器学习主流语言，本人也会尽力对代码做出较为详尽的注释。