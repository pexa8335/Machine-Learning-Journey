---
tags:
  - LinearRegression
  - GradientDescent
  - Math
  - ML
Date: 2025-09-28
Relevant:
  - "[[Lecture content/02 - Linear Regression & Gradient Descent/Linear Regression.|Linear Regression.]]"
  - "[[Gradient Descent.]]"
  - "[[3.1 - Linear Regression.]]"
---
# 1. Recall.

## 1.1 Model.
Ta nhớ lại, trong bài [[Gradient Descent.]] đã giới thiệu 1 model mới là:
$$
f_{w}(x) = w_{0} + w_{1}x
$$
Với $w = (w_{0}, w_{1})$ là bộ tham số cần tối ưu hóa của model.

## 1.2 Loss function.

Hàm mất mát MSE để evaluate model's performance sẽ có thay đổi:
$$
L(w) = \frac{1}{N}\sum_{i=1}^N(t^{(i)} - wx^{(i)})^2
$$
$$
= \frac{1}{N}\sum_{i=1}^N(t^{(i)} - (w_{0} + w_{1}x^{(i)}))^2
$$
## 1.3 Optimization.

Và tất nhiên, mục đích của chúng ta là tìm bộ tham số $w = (w_{0}, w_{1})$ sao cho khi model dùng bộ tham số này thì **Loss function** đạt giá trị nhỏ nhất - tức là, khi dùng bộ tham số này $w = (w_{0}, w_{1})$ đạt độ chính xác cao nhất.

$$\hat{w}_0, \hat{w}_1 = \underset{w_0, w_1}{\arg \min} L(w_0, w_1)$$
Tối ưu hàm $L(w_{0}, w_{1})$ theo 2 tham số $(w_{0}, w_{1})$ để tìm ra bộ $\hat{w_{0}}, \hat{w_{1}}$ tốt nhất.

Để giải quyết bài toán tối ưu hóa trên, ta có thể tính đạo hàm của $L(w)$ theo 2 biến $(w_{0}, w_{1})$ và tìm nghiệm của $\frac{\partial L(w)}{\partial w_{0}} = 0$ và $\frac{\partial L(w)}{\partial w_{1}} = 0$.

