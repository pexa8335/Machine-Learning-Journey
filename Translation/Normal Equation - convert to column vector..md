---
tags:
  - LinearRegression
  - Math
  - ML
  - RSS
Date: 2025-09-17
Relevant:
  - "[[Mathematical proof - Normal Equation.]]"
---
>[!question]
>Vì sao phải chuyển từ cột sang hàng?

### 1. Gradient bản chất là gì?

- Về định nghĩa:
    
    ∇f(w)=[∂f∂w1,∂f∂w2,…,∂f∂wn+1]\nabla f(w) = \left[ \frac{\partial f}{\partial w_1}, \frac{\partial f}{\partial w_2}, \ldots, \frac{\partial f}{\partial w_{n+1}} \right]∇f(w)=[∂w1​∂f​,∂w2​∂f​,…,∂wn+1​∂f​]
- Về hình thức: nó có thể được viết dưới dạng:
    
    - **vector hàng** 1×(n+1)1 \times (n+1)1×(n+1), hoặc
        
    - **vector cột** (n+1)×1(n+1) \times 1(n+1)×1.
        

Cả hai cách đều **đúng về mặt toán học**, chỉ khác nhau do **quy ước**.

---

### 2. Tại sao ML thường chọn vector cột?

Trong Machine Learning & tối ưu hóa:

- Tham số www được quy ước là vector cột (n+1)×1(n+1)\times 1(n+1)×1.
    
- Khi cập nhật theo gradient descent:
    
    w←w−η∇f(w)w \leftarrow w - \eta \nabla f(w)w←w−η∇f(w)
    
    thì thuận tiện hơn nếu ∇f(w)\nabla f(w)∇f(w) cũng là vector cột, vì trừ hai vector cột là hợp lý ngay.
    

Nếu gradient được viết như vector hàng, bạn vẫn dùng được, nhưng lúc update phải **transpose lại** để khớp kích thước.

---

### 3. Vậy có “bắt buộc” phải chuyển không?

- **Không bắt buộc.**
    
- Việc chuyển từ yTAy^T AyTA sang ATyA^T yATy chỉ để giữ cho gradient là vector cột, cùng shape với www.
    
- Nếu bạn làm việc trong quy ước gradient là vector hàng → giữ nguyên yTAy^T AyTA cũng đúng.