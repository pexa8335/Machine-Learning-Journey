---
tags:
  - LinearRegression
  - Math
  - ML
  - "#GradientDescent"
Date: 2025-09-24
Code:
Relevant:
  - "[[Lecture content/02 - Linear Regression & Gradient Descent/Linear Regression.|Linear Regression.]]"
---
>[!quote] Note
>This should be learned together with the code file.


## Motivation.

**Bài toán đơn giản thì có công thức giải, nhưng bài toán phức tạp thì không**

- Với linear regression 1 tham số → giải bằng tay được. [[Lecture content/02 - Linear Regression & Gradient Descent/Linear Regression.|Linear Regression.]]
- Với linear regression nhiều tham số → vẫn có công thức (normal equation).
- Nhưng khi chuyển sang deep learning (mạng nơ-ron nhiều lớp, nhiều triệu tham số) → không còn công thức giải đóng, vì hàm loss cực kỳ phức tạp, không thể giải phương trình đạo hàm = 0.
⇒ Lúc này, **gradient descent là cách duy nhất khả thi** để tìm nghiệm xấp xỉ.

## Trường hợp phức tạp (multiple parameters)

Nhưng với **Linear Regression thực tế**, model thường có dạng:  
**y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ**

Khi đó:
- Có n+1 tham số cần tối ưu: w₀, w₁, w₂, ..., wₙ
- Phải tính đạo hàm riêng theo từng tham số: ∂L/∂w₀ = 0, ∂L/∂w₁ = 0, ...
- Điều này tạo ra **hệ phương trình tuyến tính** với n+1 ẩn số
- Giải hệ này bằng đại số ma trận dẫn đến Normal Equation: **θ = (X^T X)⁻¹ X^T y**

Tạm thời chưa đi sâu vào khái niệm **Normal Equation**.

---
# Gradient Descent algorithm.

Input gồm: 
- Hàm $f(x)$ cần cực tiểu hóa.
- Initialized value $x_{0}$ tại t = 0.
- Learning rate $\alpha > 0$.

while [end condition not satisfied] do:
- $g_{t}$ = $f'(x_{t})$
- $x_{t+1} = x_{t} - \alpha g_{t}$
- t = t + 1

From each iteration $t$, calculate đạo hàm tại $f(x_{t})$ và cập nhật giá trị x tại iteration tiếp theo ngược chiều đạo hàm.

>[!question]
>Tại sao phải cập nhật giá trị $x_{t+1}$ ngược chiều đạo hàm?
Bạn đang nêu đúng ý tưởng chung, nhưng có một chút nhầm lẫn nhỏ 👇

- f′(xt)>0f'(x_t) > 0: đồ thị **đang dốc lên** khi đi sang phải → muốn giảm hàm số thì phải đi **sang trái** (ngược chiều đạo hàm).
- f′(xt)<0f'(x_t) < 0: đồ thị **đang dốc xuống** khi đi sang phải → muốn giảm hàm số thì phải đi **sang phải** (ngược chiều đạo hàm).
 
---
## Dataset.

Given $D = \{(x^{(i)}, t^{(i)}) \}_{i=1}^N$:

> $x^{(i)}$ are the input features for the $i^{th}$ data point.
> $t^{(i)}$ is the target value for the $i^{th}$ data point.

Trong ví dụ này, target values will contain noises $\varepsilon$ nhỏ.
$$
\varepsilon^{(i)} \sim N(0, \sigma^2)
$$
$$
t^{(i)} = f(x^{(i)}) + \varepsilon^{(i)}
$$
với $f(x) = 3x+2$ is an unknown target function.

**Lưu ý**: hàm $f$ này được sử dụng để minh họa, thực tế, không thể biết $f$ cũng như epsilon cũng như phân phối của noises epsilon.

## Model.

Thực nghiệm với model đơn giản:
$$
y = f_{w}(x) = w_{0}x
$$
với $w_{0}$ is the only one parameter of model.

## Loss function.

Using MSE loss function to evaluate model performance.
$$
L(w) = \frac{1}{N}\sum_{i=1}^N(t^{(i)} - f_{w}(x^{(i)}))^2 
$$
## Optimization.

Using **Gradient Descent algorithm** to find the best $w_{0}$ value so that it can cực tiểu hóa loss function MSE.

For each iteration $t$, we update:
$$
w_{0}^{(t+1)} = w_{0}^{(t)} - \alpha L'(w_{0}^{(t)})
$$
t đại diện cho thời điểm/vòng lặp thứ t, $w_{0}$ ở mỗi vòng lặp sẽ bằng $w_{0}$ ở vòng lặp trước đó trừ đi $\alpha$ nhân với giá trị đạo hàm của L'(w) tại với $w_{0}$ ở thời điểm $t$.


