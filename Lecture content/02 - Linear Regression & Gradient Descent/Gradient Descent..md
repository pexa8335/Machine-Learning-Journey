---
tags:
  - LinearRegression
  - Math
  - ML
  - "#GradientDescent"
Date: 2025-09-24
Code: Gradient_Descent.ipynb
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
# 1. Gradient Descent algorithm.

**Input:** 
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

- f′(xt)>0f'(x_t) > 0: đồ thị **đang dốc lên** khi đi sang phải → muốn giảm hàm số thì phải đi **sang trái** (ngược chiều đạo hàm).
- f′(xt)<0f'(x_t) < 0: đồ thị **đang dốc xuống** khi đi sang phải → muốn giảm hàm số thì phải đi **sang phải** (ngược chiều đạo hàm).

>[!important]
>- Learning rate shouldn't be too large vì bước nhảy của nó sẽ có thể bỏ qua điểm cực trị.
>- Learning rate shouldn't be too small vì sẽ tốn nhiều thời gian training.

---
# 2. Gradient Descent in our problem.

**Input**:
- Hàm mất mát $L(w) = MSE(w_{0})$ cần minimize.
- Initialized value $w_{0}$ tại t = 0.
- Learning rate $\alpha > 0$.
while [condition not satisfied] do:
- $g_{t} = L'(w_{0})$
- $w_{0_{t+1}} = w_{0}-\alpha.g_{t}$
- t = t + 1
We want to find the value of $w_{0}$ that can minimize the loss function MSE(w).

$$
w_{0}^* = argmin_{w_{0}}L(w) = argmin_{w_{0}} \frac{1}{N} \sum_{i=1}^N(t^{(i)} - w_{0}x^{(i)})^2
$$
Đạo hàm của hàm mất mát MSE theo $w_{0}$ là:

$$
\frac{dL}{dw_{0}} = \frac{1}{N}\sum_{i=1}^N 2(w_{0}x^{(i)}-t^{(i)})x^{(i)}
$$
---
## 2.1 Dataset.

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

## 2.2 Model.

Thực nghiệm với model đơn giản:
$$
y = f_{w}(x) = w_{0}x
$$
với $w_{0}$ is the only one parameter of model.

## 2.3 Loss function.

Using MSE loss function to evaluate model performance.
$$
L(w) = \frac{1}{N}\sum_{i=1}^N(t^{(i)} - f_{w}(x^{(i)}))^2 
$$
## 2.4 Optimization.

Using **Gradient Descent algorithm** to find the best $w_{0}$ value so that it can cực tiểu hóa loss function MSE.

For each iteration $t$, we update:
$$
w_{0}^{(t+1)} = w_{0}^{(t)} - \alpha L'(w_{0}^{(t)})
$$
t đại diện cho thời điểm/vòng lặp thứ t, $w_{0}$ ở mỗi vòng lặp sẽ bằng $w_{0}$ ở vòng lặp trước đó trừ đi $\alpha$ nhân với giá trị đạo hàm của L'(w) tại với $w_{0}$ ở thời điểm $t$.

![[Pasted image 20250926235137.png]]

Nhưng, vấn đề của hàm 1 biến $w_{0}x$ là luôn đi qua gốc tọa độ, với sample $x^{(i)}$ như vậy, sẽ không có đường thẳng nào đi qua gốc tọa độ thật sự tốt để biểu diễn toàn bộ dataset.

---
## 2.5 Model.

Ta thực nghiệm với một model mới.
$$
y = f_{w}(x) = w_{0} + w_{1}x
$$
Với $w = (w_{0}, w_{1})$ là bộ tham số cần tối ưu hóa của model.

## 2.6 Loss function.

Hàm mất mát MSE để evaluate model's performance sẽ có thay đổi:
$$
L(w) = \frac{1}{N}\sum_{i=1}^N(t^{(i)} - wx^{(i)})^2
$$
$$
= \frac{1}{N}\sum_{i=1}^N(t^{(i)} - (w_{0} + w_{1}x^{(i)}))^2
$$
## 2.7 Optimization.

Sử dụng Gradient Descent để tìm ra tập tham số $w = (w_{0}, w_{1})$ sao cho có thể minimize loss function MSE.

Calculate partial deriative of this loss function theo $w_{0}$ và $w_{1}$
$$
\frac{\partial L(\mathbf{w})}{\partial w_0} = \frac{1}{N}\sum_{i=1}^N \frac{\partial}{\partial w_{0}}(t^{(i)} - (w_{0} + w_{1}x^{(i)}))^2
$$
$$
= \frac{1}{N}\sum_{i=1}^N 2((w_{0} +w_{1}x^{(i)}) - t^{(i)})
$$

$$
 \frac{\partial L(\mathbf{w})}{\partial w_{1}} = \frac{1}{N} \sum_{i=1}^N 2((w_{0} + w_{1}x^{(i)})-t^{(i)})x^{(i)}
$$

# Gradient.

Định nghĩa **gradient $\nabla_{w}L(w)$** của loss function MSE $L(w)$ với vector thành phần là các đạo hàm riêng của $L(w)$ theo từng tham số của bộ tham số $w = (w_{0}, w_{1})$.

$$
\nabla_w L(\mathbf{w}) = \begin{bmatrix} \frac{\partial L(\mathbf{w})}{\partial w_0} \\ \frac{\partial L(\mathbf{w})}{\partial w_1} \end{bmatrix} = \frac{2}{N} \begin{bmatrix} \sum_{i=1}^N ((w_0 + w_1 x^{(i)}) - t^{(i)}) \cdot 1 \\ \sum_{i=1}^N ((w_0 + w_1 x^{(i)}) - t^{(i)}) \cdot x^{(i)} \end{bmatrix} = \frac{2}{N} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{t})
$$
Ký hiệu:
$$
\mathbf{X} = \begin{bmatrix} 1 & x^{(1)} \\ 1 & x^{(2)} \\ \vdots & \vdots \\ 1 & x^{(N)} \end{bmatrix}, \quad \mathbf{w} = \begin{bmatrix} w_0 \\ w_1 \end{bmatrix}, \quad \mathbf{t} = \begin{bmatrix} t^{(1)} \\ t^{(2)} \\ \vdots \\ t^{(N)} \end{bmatrix}
$$

>[!question]
>Tại sao có cột 1 ở vector X?
>- Đặt nhân tử chung, $w_{1}x^{(i)}$ thì rút $x^{(i)}$, $w_{0} \Leftrightarrow w_{0}\times_{1}$ nên rút 1. 

>[!question]
>Tại sao lại có $X^T$ ở đây thay vì $x^{(i)}$?

$\mathbf{X}: n \times 2$

$\mathbf{w}: 2 \times 1$

$\mathbf{X}\mathbf{w} - \mathbf{t} = \mathbf{e}: n \times 1$

Nếu thử $\mathbf{X} \times \mathbf{e}$:

$\mathbf{X}$ là $(n \times 2)$, $\mathbf{e}$ là $(n \times 1)$ $\rightarrow$ không nhân được (vì inner dimension $2 \neq n$).

Còn $\mathbf{X}^T \mathbf{e}$:

$\mathbf{X}^T$ là $(2 \times n)$, $\mathbf{e}$ là $(n \times 1)$ $\rightarrow$ kết quả $(2 \times 1)$, cùng shape với $\mathbf{w}$.

>[!danger] Important
>Gradient $\nabla_{w}L(w)$ là vector trỏ về hướng giá trị L(w) tăng, ta đang cần minimize hàm mất mát nên ta phải cập nhật giá trị bộ tham số $w = (w_{0}, w_{1})$ theo hướng ngược lại với đạo hàm (như đầu bài đã nói lí do): $-\nabla_{w}L(w)$


Đầu vào: 
- Hàm mất mát $L(\mathbf{w})=\text{MSE}(w_0)$ cần cực tiểu hóa.
- Giá trị khởi tạo $\mathbf{w}$ tại $t=0$ (initial value).
- Hệ số học (learning rate) $\alpha > 0$.
while [điều kiện dừng chưa thỏa] do:
		$\mathbf{g}_t = \nabla_w L(\mathbf{w})$
		$\mathbf{w} = \mathbf{w} - \alpha \mathbf{g}_t$
		$t = t+1$
----
# Learning rate và xấp xỉ bậc hai chưa xong.

