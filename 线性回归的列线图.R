setwd("C:\\Users\\HNNHSyc645\\Desktop\\这儿黎明静悄悄.wonder\\2026美赛实战保奖班")

# 1. 加载所需的程序包
# 如果尚未安装，请先通过 install.packages("包名") 进行安装
install.packages("skimr")
install.packages("regplot")

library(caret)    # 用于数据拆分
library(skimr)    # 用于数据概览
library(regplot)  # 用于绘制列线图
library(ggplot2)  # 用于数据可视化

# 2. 加载您的数据
# 请将 "2023 年 1 月全国城市空气质量报告 3版.csv" 替换为您的实际文件路径
# 或者，您可以运行这行代码，然后手动选择文件
air_quality <- read.csv("2023 年 1 月全国城市空气质量报告 3版.csv")

# 3. 查看数据基本情况
skimr::skim(air_quality)

# 4. 拆分数据集
# 目的是将数据分为训练集（用于建模）和测试集（用于评估模型性能）
set.seed(42)  # 设置随机种子以保证结果可复现
train_indices <- createDataPartition(
  y = air_quality$综合指数AQI, # 目标变量
  p = 0.75,                   # 训练集占比75%
  list = FALSE
)
train_data <- air_quality[train_indices, ]
test_data <- air_quality[-train_indices, ]

# 5. 构建线性回归模型公式
# 我们将使用 "PM2.5", "PM10", "SO2", "NO2", "CO", "O3" 作为预测变量
predictor_vars <- c("PM2.5", "PM10", "SO2", "NO2", "CO", "O3")
formula_reg <- as.formula(
  paste("综合指数AQI ~", paste(predictor_vars, collapse = " + "))
)
print(formula_reg) # 打印并检查公式


# 6. 训练模型
set.seed(42)
fit_lm <- lm(formula_reg, data = train_data)

# 7. 查看模型摘要
# summary() 提供了模型的详细信息，如系数、R²值等
summary(fit_lm)

# 8. 绘制列线图
# 列线图（Nomogram）可以直观地展示每个变量对预测结果的贡献
regplot(
  reg = fit_lm,
  title = "空气质量综合指数预测模型",
  subticks = TRUE
)

# 9. 模型性能评估
# 9.1 在训练集上进行预测
train_pred <- predict(fit_lm, newdata = train_data)
# 计算训练集的误差指标（RMSE, R-squared, MAE）
defaultSummary(data.frame(obs = train_data$综合指数AQI, pred = train_pred))

# 9.2 在测试集上进行预测
test_pred <- predict(fit_lm, newdata = test_data)
# 计算测试集的误差指标
defaultSummary(data.frame(obs = test_data$综合指数AQI, pred = test_pred))

# 10. 结果可视化
# 10.1 训练集预测结果可视化
plot(x = train_data$综合指数AQI,
     y = train_pred,
     las = 1,
     xlab = "实际值 (Actual)",
     ylab = "预测值 (Prediction)",
     main = "实际值与预测值比较",
     sub = "训练集")
abline(a = 0, b = 1, col = "red", lwd = 2.5, lty = "dashed")

# 10.2 测试集预测结果可视化
plot(x = test_data$综合指数AQI,
     y = test_pred,
     las = 1,
     xlab = "实际值 (Actual)",
     ylab = "预测值 (Prediction)",
     main = "实际值与预测值比较",
     sub = "测试集")
abline(a = 0, b = 1, col = "red", lwd = 2.5, lty = "dashed")

# 10.3 使用ggplot2进行更美观的集中展示
pred_result <- data.frame(
  obs = c(train_data$综合指数AQI, test_data$综合指数AQI),
  pred = c(train_pred, test_pred),
  group = c(rep("训练集", length(train_pred)), rep("测试集", length(test_pred)))
)

ggplot(pred_result, aes(x = obs, y = pred, fill = group, colour = group)) +
  geom_point(shape = 21, size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", linewidth = 1) +
  labs(title = "模型预测性能",
       x = "实际综合指数AQI",
       y = "预测综合指数AQI",
       fill = NULL, colour = NULL) +
  theme_bw(base_size = 14) +
  theme(legend.position = "bottom")





































































