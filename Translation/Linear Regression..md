---
tags:
  - LinearRegression
  - ML
Date: 2025-09-16
Relevant: "[[3.1 Linear Regression.]]"
---
## Explanation.

Trong hàm tuyến tính $f(x) = w_{0} + w_{1}x_{1}+w_{2}x_{2} +\dots + w_{n}x_{n}$:

*   **$w_0$ được gọi là hệ số chặn (intercept term) hoặc bias term.**
*   Nó đại diện cho giá trị của $f(x)$ khi tất cả các biến đầu vào $x_1, x_2, \dots, x_n$ đều bằng 0.

Hãy xem xét hai trường hợp:

1.  **Khi có $w_0$:**
    Nếu bạn đặt tất cả các $x_i = 0$, ta có:
    $f(0) = w_{0} + w_{1}(0) + w_{2}(0) +\dots + w_{n}(0)$
    $f(0) = w_{0}$

    Điều này có nghĩa là khi tất cả các đầu vào bằng 0, giá trị của hàm là $w_0$. Nếu $w_0 \neq 0$, thì hàm sẽ không đi qua gốc tọa độ $(0, 0, \dots, 0)$.

2.  **Khi bỏ $w_0$ (tức là $w_0 = 0$):**
    Hàm trở thành:
    $f(x) = w_{1}x_{1}+w_{2}x_{2} +\dots + w_{n}x_{n}$

    Bây giờ, nếu đặt tất cả các $x_i = 0$, ta có:
    $f(0) = w_{1}(0)+w_{2}(0) +\dots + w_{n}(0)$
    $f(0) = 0$

    **Do đó, khi $w_0$ bị loại bỏ (hoặc bằng 0), hàm $f(x)$ sẽ luôn đi qua gốc tọa độ $(0, 0, \dots, 0)$** vì khi tất cả các đầu vào là 0, đầu ra cũng là 0.

Trong ngữ cảnh hình học, $w_0$ xác định điểm mà đường thẳng (hoặc mặt phẳng, siêu mặt phẳng) cắt trục $f(x)$ (hoặc trục y trong không gian 2D) khi tất cả các biến độc lập bằng 0. Nếu $w_0=0$, điểm cắt đó chính là gốc tọa độ.