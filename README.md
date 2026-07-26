# 🧠 BÁO CÁO BÀI TẬP LỚN — NHẬP MÔN TRÍ TUỆ NHÂN TẠO

## Đề số 5: Cài đặt Perceptron và Mạng Nơ-ron Nhiều Lớp (MLP) từ đầu bằng NumPy

<div align="center">

| Thông tin | Chi tiết |
|-----------|----------|
| **Môn học** | Nhập môn Trí tuệ Nhân tạo |
| **Mã học phần** | CSE703041-2-3-25(N02) |
| **Trường** | Trường Công nghệ Thông tin — Đại học Phenikaa |
| **Giảng viên** | ThS. Phan Thị Hoài |
| **Khoá** | K18 — ICT3 |
| **Thời gian** | Tháng 7 năm 2026 |

</div>

---

## 👥 Thành viên nhóm

| Họ và tên | MSSV |
|-----------|------|
| Ngô Quang Thiện | 24102651 |
| Nguyễn Hưng Vũ | 24102769 |

---

## 🎯 Mục tiêu đề tài

Đề tài nhằm nghiên cứu và hiện thực hoá từ đầu (from-scratch) hai mô hình nền tảng của học sâu — **Perceptron** và **Mạng nơ-ron nhiều lớp (MLP)** — hoàn toàn bằng thư viện **NumPy**, không sử dụng bất kỳ framework Deep Learning nào (PyTorch, TensorFlow, Keras).

Thông qua quá trình cài đặt thủ công, nhóm làm rõ:
- Cơ chế **lan truyền thuận (Forward Propagation)** và **lan truyền ngược (Backpropagation)**
- Nguyên lý **cập nhật trọng số** thông qua quy tắc đạo hàm dây chuyền (Chain Rule)
- Ảnh hưởng của **hàm kích hoạt**, **thuật toán tối ưu** và **kỹ thuật regularization** đến khả năng học của mô hình

---

## 📂 Cấu trúc thư mục

```text
BAO_CAO_NHAP_MON_TRI_TUE_NHAN_TAO/
│
├── 📓 De_So_5.ipynb                    # Notebook chính — toàn bộ code, đồ thị và nhận xét
│
├── 📁 bao_cao_latex/
│   ├── ban_nop.tex                     # Mã nguồn LaTeX — báo cáo hoàn chỉnh
│   ├── bao_cao_de5_hoanchinh.pdf       # File PDF báo cáo đã biên dịch (18 trang)
│   ├── logoo.jpg                       # Logo trường
│   ├── 40.jpg – 50.jpg                 # Hình ảnh minh hoạ trong báo cáo
│   └── 17.jpg – 20.jpg                 # Hình ảnh dự phòng
│
├── 🖼️ XOR.png                          # Decision Boundary của Perceptron trên XOR
├── 🖼️ train_xor_mlp.png               # Decision Boundary & Confusion Matrix MLP trên XOR
├── 🖼️ forward_backward.png            # Đồ thị các hàm kích hoạt và đạo hàm
├── 🖼️ train_iris.png                  # Loss & Confusion Matrix của MLP trên Iris
├── 🖼️ so_sanh_SGD_SGDwithMomentum.png # So sánh tốc độ hội tụ SGD vs SGD+Momentum
├── 🖼️ thu_nghiem_1_2_3_an.png         # Thử nghiệm kiến trúc 1, 2, 3 lớp ẩn
├── 🖼️ l2.png                          # Đánh giá L2 Regularization chống Overfitting
│
└── 📄 README.md                        # File hướng dẫn này
```

---

## ⚙️ Các tính năng & thuật toán đã cài đặt

### 1. Perceptron một lớp (Single-Layer Perceptron)
- Hàm kích hoạt bước nhảy (Step Function)
- Quy tắc học Perceptron (Perceptron Learning Rule)
- Thực nghiệm trên bài toán XOR để minh hoạ giới hạn phân tách tuyến tính

### 2. Mạng nơ-ron nhiều lớp (MLP — Multi-Layer Perceptron)
- **Kiến trúc linh hoạt:** Số lớp ẩn và số nơ-ron mỗi lớp tuỳ chỉnh qua tham số `layer_sizes`
- **Forward Propagation:** Phép nhân ma trận NumPy qua các lớp
- **Backward Propagation:** Chain Rule — tính gradient chính xác qua toàn bộ mạng
- **Hàm kích hoạt:**
  - `Sigmoid` — cho lớp ẩn (bài toán nhị phân)
  - `ReLU` — cho lớp ẩn (hội tụ nhanh hơn)
  - `Softmax` — cho lớp đầu ra (phân loại nhiều lớp)
- **Hàm mất mát:**
  - `MSE` — hồi quy
  - `Cross-Entropy` — phân loại

### 3. Yêu cầu mở rộng
- **Thuật toán tối ưu:** So sánh `SGD` và `SGD with Momentum` ($\beta = 0.9$)
- **Thử nghiệm kiến trúc:** Đánh giá ảnh hưởng của 1, 2, 3 lớp ẩn
- **L2 Regularization (Weight Decay):** Kỹ thuật chống overfitting

---

## 📐 Quy ước ký hiệu toán học

Toàn bộ báo cáo tuân theo quy ước phổ biến trong học sâu (Deep Learning):
**chữ hoa = ma trận**, **chữ thường = vector hoặc đại lượng vô hướng tùy ngữ cảnh**.

| Ký hiệu | Ý nghĩa |
|---------|---------|
| $X$ | Ma trận dữ liệu đầu vào (nhiều mẫu) |
| $x$ | Vector đặc trưng một mẫu |
| $W^{(l)},\ b^{(l)}$ | Trọng số & bias lớp $l$ |
| $Z^{(l)},\ A^{(l)}$ | Giá trị trước/sau kích hoạt lớp $l$ |
| $\hat{y} = A^{(L)}$ | Đầu ra dự đoán của mô hình |
| $\delta^{(l)}$ | Sai số lan truyền ngược lớp $l$ |
| $\eta,\ \beta,\ \lambda$ | Learning rate, Momentum, hệ số L2 |

---

## 📊 Kết quả thực nghiệm

| Thực nghiệm | Mô hình | Kết quả |
|-------------|---------|---------|
| Bài toán XOR | Perceptron 1 lớp | ❌ Chỉ đạt ~75% — không hội tụ |
| Bài toán XOR | MLP (2–4–1, ReLU) | ✅ **100% accuracy** — đường biên phi tuyến |
| Tập Iris | MLP (4–8–3, ReLU+Softmax) | ✅ **96% – 100%** trên tập test |
| SGD vs Momentum | MLP trên Iris | ✅ Momentum hội tụ nhanh hơn ~3x |
| Kiến trúc lớp ẩn | 1 / 2 / 3 lớp ẩn | ✅ Kiến trúc [4–32–16–3] tối ưu nhất |
| L2 Regularization | MLP lớn (4–64–32–3) | ✅ Val Loss giảm từ 0.2 xuống 0.1 |

---

## 🚀 Hướng dẫn chạy

### Yêu cầu môi trường
- Python 3.9 trở lên
- Các thư viện: `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

### Cài đặt thư viện
```bash
pip install numpy matplotlib seaborn scikit-learn
```

### Chạy Notebook
Mở file `De_So_5.ipynb` trên **VS Code** hoặc **Jupyter Notebook / JupyterLab** và chọn **Run All Cells**.

Hoặc xem trực tiếp trên Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13ptl7R3-004uGGPBXaLPsomsKylZyP8T?usp=sharing)

---

## 📄 Báo cáo

File báo cáo hoàn chỉnh (18 trang, định dạng LaTeX → PDF) được đặt tại:

📑 [`bao_cao_latex/bao_cao_de_so_5.pdf`](bao_cao_latex/bao_cao_de_so_5.pdf)

**Nội dung báo cáo bao gồm:**
1. Giới thiệu & Mục tiêu đề tài
2. Cơ sở lý thuyết (Quy ước ký hiệu, Perceptron, Activation Functions, Forward/Backpropagation, Loss Functions, Optimization, L2 Regularization)
3. Yêu cầu cài đặt cơ bản (XOR, MLP)
4. Yêu cầu mở rộng (SGD vs Momentum, số lớp ẩn, L2)
5. Dữ liệu & Môi trường thực nghiệm

---

## 📚 Tài liệu tham khảo

- Ian Goodfellow, Yoshua Bengio, Aaron Courville — *Deep Learning* (MIT Press, 2016)
- CS231n — *Convolutional Neural Networks for Visual Recognition* (Stanford University)
- NumPy Documentation — https://numpy.org/doc/
- Scikit-learn Documentation — https://scikit-learn.org/stable/

---

<div align="center">
  <sub>Đại học Phenikaa — Khoa Khoa học Dữ liệu và Trí tuệ Nhân tạo — Hà Nội, 2026</sub>
</div>
