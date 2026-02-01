setwd("C:\\Users\\HNNHSyc645\\Desktop\\这儿黎明静悄悄.wonder\\2026美赛实战保奖班")

#################################lasso回归筛选特征变量#################################
#################################lasso回归筛选特征变量#################################
install.packages("mlbench")

library(glmnet)
library(mlbench)             # 加载mlbench包
library(dplyr)

data=read.csv("2023 年 1 月全国城市空气质量报告 3版.csv")
head(data)

mydata=data.frame(select(data,PM2.5,PM10,SO2,NO2,CO,O3,综合指数AQI))
colSums(is.na(mydata))   #再次总结NA数量

#x,y输入
y<-mydata$综合指数AQI
x=as.matrix(data.frame(select(mydata,PM2.5,PM10,SO2,NO2,CO,O3)))

###family参数的选取 family= c("gaussian", "binomial", "poisson", "multinomial", "cox", "mgaussian"),
#"gaussian"   （高斯分布）：因变量为连续型数值变量，定量变量
#"binomial"   （二项分布）：因变量是二分类变量， 0 和 1 ，定类变量
#"poisson"    （泊松分布）：因变量是非负整数的计数类型数据
#"multinomial"（多项分布）：因变量是多分类变量，0，1，3，4，6等等 

modelxy=glmnet(x,y,family="gaussian",nlambda=100, alpha=1)
modelxy

#绘制回归系数路径图
plot(modelxy,xvar="lambda", label=TRUE)  #随着log（lambda）增加，系数值逐步迫近为0，可以将不重要的特征系数压缩为 0。

#绘制十折交叉验证
set.seed(999)
ccccvfit=cv.glmnet(x,y,family = "gaussian") #不同log（lambda）值下的均方误差MSE
plot(ccccvfit)


#输出lambda.min和lambda.1se
ccccvfit$lambda.min #求出最小值                 
ccccvfit$lambda.1se #求出最小值一个标准误的λ值    


#求出系数压迫后的最新系数
coef1<-coef(ccccvfit,s = "lambda.min")
coef2<-coef(ccccvfit,s = "lambda.1se")

coef1
coef2
#################################结束结束############################################
#################################结束结束############################################




################################################### 随机森林#####################################
################################################### 随机森林#####################################
################################################### 随机森林#####################################
install.packages("randomForest")
install.packages("caret")

#导包
library(randomForest)
library(caret)
library(pROC)
library(dplyr)

# 第一步：导入数据
data=read.csv("心脏病诊断数据集.csv")
mydata=data.frame(select(data,年龄,性别,胸痛类型,静息血压,血浆类固醇含量,空腹血糖是否超标,静息心电图结果,最高心率,运动型心绞痛,运动引起的ST抑制,最大运动量时ST下降,THAL,是否患有心脏病))
summary(mydata)

# 第二步：拆分数据集
dl = createDataPartition(mydata$是否患有心脏病,p = 0.8, list = F) #自己设置比例，本例为7:3
traindata= data[dl,]       #训练集
valdata = data[-dl,]       #验证集
set.seed(10)  #设置随机种子


# 第三步：用训练集训练随机森林模型
res = randomForest(as.factor(是否患有心脏病)~.,data=traindata,importance=T,na.action = na.pass,ntree=300) 

plot(res)  #绘制决策树与误差图
importance(res)   #自变量的重要性指标得分
varImpPlot(res)   #绘制重要性排序图

