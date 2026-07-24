# BÁO CÁO BÀI TẬP LỚN: NHẬP MÔN TRÍ TUỆ NHÂN TẠO

## ĐỀ SỐ 5: Cài đặt Perceptron và Mạng Nơ-ron Nhiều Lớp (MLP) Từ Đầu Bằng NumPy

> **Môn học:** Nhập môn Trí tuệ Nhân tạo  
> **Yêu cầu cốt lõi:** Lập trình from-scratch 100% bằng **NumPy** thuần, không sử dụng các framework Deep Learning (PyTorch, TensorFlow, Keras).

---

## 📌 1. Cấu trúc Thư mục Bài làm

```text
.
├── De_So_5.ipynb                   # Jupyter Notebook chính chứa toàn bộ Code + Đồ thị + Nhận xét
├── Bao_Cao_Phan_3.md               # Báo cáo chi tiết nội dung Phần 3 dưới dạng Markdown
├── Bao_Cao_Phan_3.docx             # File Báo cáo Word hoàn chỉnh
├── XOR.png                         # Biểu đồ Decision Boundary Perceptron trên XOR
├── train_xor_mlp.png               # Biểu đồ Decision Boundary & Confusion Matrix MLP trên XOR
├── forward_backward.png            # Biểu đồ Trực quan hóa các Hàm kích hoạt & Đạo hàm
├── train_iris.png                  # Biểu đồ Loss & Confusion Matrix MLP trên Iris
├── so_sanh_SGD_SGDwithMomentum.png # Biểu đồ So sánh tốc độ hội tụ SGD vs SGD with Momentum
├── thu_nghiem_1_2_3_an.png         # Biểu đồ Thử nghiệm số lượng Lớp ẩn (1, 2, 3 lớp)
├── l2.png                          # Biểu đồ Đánh giá hiệu quả L2 Regularization chống Overfitting
└── README.md                       # File hướng dẫn và tóm tắt bài làm
```

---

## 🚀 2. Các Tính năng & Thuật toán Đã Cài đặt

1. **Perceptron 1 Lớp (Single-Layer Perceptron):**
   * Hàm bước nhảy (Step function).
   * Luật học Perceptron (Perceptron Learning Rule).
   * Thực nghiệm bài toán XOR & Chứng minh giới hạn phân tách tuyến tính.

2. **Mạng Nơ-ron Nhiều Lớp (MLP - Multi-Layer Perceptron):**
   * **Forward Propagation:** Hỗ trợ đa dạng số lớp ẩn và nơ-ron tùy biến.
   * **Backward Propagation:** Tự tính toán Đạo hàm riêng theo quy tắc chuỗi (Chain Rule) bằng ma trận NumPy.
   * **Hàm kích hoạt:** Sigmoid, ReLU (cho Lớp ẩn) và Softmax (cho Lớp Output đa phân lớp).
   * **Hàm mất mát:** Binary Cross-Entropy và Categorical Cross-Entropy.

3. **Yêu cầu Mở rộng:**
   * **Thuật toán Tối ưu:** SGD truyền thống và **SGD with Momentum** ($\beta = 0.9$).
   * **Thử nghiệm Kiến trúc:** Đánh giá ảnh hưởng của số lượng lớp ẩn (1, 2, 3 lớp).
   * **Kỹ thuật Chống Overfitting:** **L2 Regularization** (Weight Decay).

---

## 📊 3. Hướng dẫn Chạy Bài làm

### Môi trường yêu cầu:
* Python 3.9+
* Thư viện: `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

### Cài đặt thư viện:
```bash
pip install numpy matplotlib seaborn scikit-learn
```

### Chạy Notebook:
Mở file `De_So_5.ipynb` trên VS Code hoặc Jupyter Notebook / Jupyter Lab và chọn **Run All Cells**.

---

## 📝 4. Kết quả Đạt được

* **Bài toán XOR:** Perceptron 1 lớp thất bại (50% accuracy), trong khi MLP 1 lớp ẩn giải quyết 100% chính xác với đường biên phi tuyến rõ rệt.
* **Tập dữ liệu Iris:** MLP đạt độ chính xác **96% - 100%** trên tập Test.
* **SGD vs Momentum:** Momentum bứt tốc độ hội tụ nhanh gấp nhiều lần so với SGD thuần.
* **L2 Regularization:** Khắc phục triệt để hiện tượng Overfitting khi huấn luyện trên tập dữ liệu khan hiếm.
