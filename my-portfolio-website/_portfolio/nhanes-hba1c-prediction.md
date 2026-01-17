---
title: "基于常规体检指标的 HbA1c 风险预测模型"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/nhanes-hba1c-prediction
date: 2024-01-15
excerpt: "利用 LASSO 特征选择与集成学习，基于常规体检指标构建高精度 HbA1c 风险预测模型，AUC 达 0.87"
header:
  teaser: /images/portfolio/nhanes-hba1c-prediction/model_roc_curves.png
tags:
  - 机器学习
  - 医疗数据分析
  - 特征选择
  - 模型可解释性
tech_stack:
  - name: Python
  - name: Scikit-learn
  - name: XGBoost
  - name: SHAP
  - name: Pandas
  - name: Matplotlib
  - name: Seaborn
---

## 项目背景

糖化血红蛋白（HbA1c）是糖尿病诊断和管理的关键指标。传统的 HbA1c 检测需要静脉采血，成本较高且不便。本项目旨在利用美国国家健康与营养调查（NHANES）数据，基于常规体检指标（如 BMI、血压、腰围等）构建机器学习模型，预测个体 HbA1c 异常风险，为基层医疗提供低成本筛查工具。

## 数据与方法

### 研究人群特征
研究人群的基本特征如表1所示，包括训练集和验证集的人口统计学、体格测量和生化指标分布：

![人口学资料 Table 1](/images/portfolio/nhanes-hba1c-prediction/table1_demographic_characteristics.png)
*表1：研究人群基线特征（训练集 vs 验证集）*

**关键发现：**
1. **样本规模**：总样本量超过10,000人，具有统计学意义
2. **数据完整性**：关键变量缺失率低于5%，数据质量良好
3. **分组平衡**：训练集和验证集在人口学特征上分布均衡
4. **风险分布**：高HbA1c风险比例约30%，与人群流行病学数据一致
5. **特征分布**：连续变量（如BMI、血压）符合正态或近似正态分布

### 数据来源与预处理
项目整合了 NHANES 四个调查周期（2011-2012、2013-2014、2015-2016、2017-2018）的多个数据模块，包括人口学、体检、实验室和问卷数据。通过数据清洗、缺失值处理和特征工程，构建了包含50+个候选特征的数据集。

### 特征选择方法
采用 LASSO（Least Absolute Shrinkage and Selection Operator）回归进行特征选择，通过交叉验证确定最优正则化参数，从众多候选特征中筛选出最具预测价值的指标。

### 模型构建与评估
构建并比较了五种机器学习模型：
1. 逻辑回归（Logistic Regression）
2. 随机森林（Random Forest）
3. 梯度提升（Gradient Boosting）
4. 支持向量机（SVM）
5. 多层感知器（MLP）

使用13个评估指标进行综合性能评估，包括 AUC、准确率、召回率、F1分数等。

## 核心实现

### 1. LASSO 特征选择
```python
# LASSO 特征选择
from sklearn.linear_model import LassoCV

# 初始化 LASSO 交叉验证模型
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)

# 拟合模型
lasso_cv.fit(X_train_scaled, y_train)

# 获取选择的特征
selected_features = X_train.columns[lasso_cv.coef_ != 0]
print(f"LASSO 选择的特征数量: {len(selected_features)}")
print(f"选择的特征: {selected_features.tolist()}")