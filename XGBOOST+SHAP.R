setwd("C:\\Users\\HNNHSyc645\\Desktop\\这儿黎明静悄悄.wonder\\2026美赛实战保奖班")

# 安装包
install.packages("xgboost")
install.packages("shapviz")
install.packages("ggplot2")

# 加载库
library(xgboost)
library(ggplot2)
library(shapviz)
library(dplyr)

# 加载数据
data=read.csv("2023 年 1 月全国城市空气质量报告 3版.csv")
head(data)

X=data.frame(select(data,PM2.5,PM10,SO2,NO2,CO,O3))
head(X)

y<-as.numeric(data$综合指数AQI) 
head(y)

# 训练XGBoost模型
model <- xgboost(data=as.matrix(X), label = y, nrounds =100,objective ="reg:squarederror")

# 计算SHAP
shap <- shapviz(model, X_pred =as.matrix(X))
shap

# 计算特征重要性 - 正确方法
importance_matrix <- xgb.importance(
  feature_names = colnames(X),
  model = model
)

# 查看特征重要性
print(importance_matrix)

# 绘制特征重要性图
xgb.plot.importance(importance_matrix = importance_matrix)


#单样本的shap解释 
sv_waterfall(shap, row_id = 1)
sv_force(shap,row_id = 1)

#单样本的shap解释 
sv_waterfall(shap, row_id = 128)  
sv_force(shap,row_id = 128)


# 可视化：蜂群图
sv_importance(shap,
              kind ="bee",
              size = 1.5,
) + 
  ggtitle("                            全局样本的特征解释（蜂群图）                    
 <--抑制AQI的方向                                            提高AQI的方向-->"
  )+
  theme_bw()+ #空白背景
  scale_color_gradient(low ="#f7d13d", high ="#a52c60")



#单个变量依赖图
sv_dependence(shap,v=c("PM2.5","PM10","SO2","NO2","CO","O3"),color_var = NULL,alpha = 0.2,color = "#990000",size = 2)

#多个变量偏相关依赖图
sv_dependence(shap,v=c("PM2.5","PM10","SO2","NO2","CO","O3"),alpha = 0.2)

































