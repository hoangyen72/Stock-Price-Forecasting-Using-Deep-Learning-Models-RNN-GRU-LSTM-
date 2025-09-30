# Stock-Price-Forecasting-Using-Deep-Learning-Models-RNN-GRU-LSTM-
## 🎯 Tổng quan Dự án
Dự án này tập trung vào việc ứng dụng các mô hình **Học Sâu (Deep Learning)**, cụ thể là **Mạng nơ-ron hồi quy (RNN)** và các biến thể **LSTM** và **GRU**, để giải quyết bài toán **Dự báo chuỗi thời gian** trên thị trường chứng khoán Việt Nam.

Mục tiêu chính là đánh giá một cách có hệ thống hiệu quả của các kiến trúc mô hình khác nhau và các bộ tính năng đầu vào (từ dữ liệu đơn biến đơn giản đến dữ liệu đa biến phức tạp được tăng cường bằng các chỉ báo kỹ thuật) trên các loại tài sản tài chính khác nhau.

---

## ✨ Tính năng Chính & Kỹ thuật

| **Lĩnh vực**            | **Kỹ thuật/Mô tả**                                                                 |
|--------------------------|------------------------------------------------------------------------------------|
| **Mô hình Học Sâu**      | Thực hiện và so sánh toàn diện các kiến trúc RNN, LSTM, và GRU.                    |
| **Kỹ thuật Tính năng**   | Áp dụng cả kịch bản Đơn biến (chỉ sử dụng giá đóng cửa) và Đa biến (giá Cao nhất, Thấp nhất, Đóng cửa). |
| **Tích hợp Chỉ báo**     | Tăng cường hiệu suất mô hình bằng cách bổ sung các Chỉ báo kỹ thuật (ví dụ: MACD, RSI) làm các biến đầu vào. |
| **Dữ liệu Nghiên cứu**   | Thử nghiệm trên ba bộ dữ liệu tài chính Việt Nam: **VN30**, **HPG** (cổ phiếu vốn hóa lớn), và **SZC** (cổ phiếu vốn hóa vừa). |
| **Đánh giá Hiệu suất**   | Đánh giá mô hình nghiêm ngặt bằng các chỉ số chuỗi thời gian tiêu chuẩn: **MSE, RMSE, MAPE, R²**. |
| **Tối ưu hóa**           | Sử dụng phương pháp **Grid Search** để tinh chỉnh các siêu tham số quan trọng.     |

---

## 🚀 Kết quả và Phát hiện Quan trọng

- **Mô hình Đa biến vượt trội**: Việc sử dụng thêm thông tin về giá cao nhất và thấp nhất giúp cải thiện đáng kể độ chính xác dự báo so với chỉ dùng giá đóng cửa.  
- **Chỉ báo Kỹ thuật giúp tăng hiệu quả**: Các mô hình có thêm chỉ báo kỹ thuật luôn cho kết quả dự báo tốt hơn, với MSE, RMSE, MAPE giảm và R² tăng.  
- **GRU và RNN hiệu quả hơn LSTM**: Với dữ liệu chứng khoán Việt Nam, GRU và RNN cơ bản cho dự báo ổn định và chính xác hơn LSTM. Cấu trúc đơn giản của chúng phù hợp hơn với dữ liệu có nhiễu và biến động cao.  
- **Dự báo chỉ số chính xác hơn cổ phiếu đơn lẻ**: VN30 đạt độ chính xác cao nhất (MAPE ≈ 1–2%), trong khi cổ phiếu HPG và SZC thấp hơn (MAPE ≈ 2–4%).  

---

## ⚙️ Công nghệ sử dụng

- **Ngôn ngữ**: Python  
- **Deep Learning**: TensorFlow, Keras  
- **Xử lý Dữ liệu**: Pandas, NumPy  
- **Phân tích Kỹ thuật**: Thư viện TA (MACD, RSI, …)  
- **Trực quan hóa**: Matplotlib, Seaborn  

---

## 🔮 Hạn chế và Hướng phát triển Tương lai

- **Yếu tố Ngoại sinh**: Mô hình chưa xét đến tin tức vĩ mô, chính sách, tâm lý đám đông.  
- **Tối ưu hóa Siêu tham số**: Grid Search tốn tài nguyên và có thể chưa đạt tối ưu toàn cục.  
- **Phát triển**: Cần mở rộng thử nghiệm trên nhiều mã cổ phiếu và tối ưu hóa môi trường huấn luyện để tăng quy mô.  
