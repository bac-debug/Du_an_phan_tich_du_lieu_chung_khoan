<h2 align="center">
    <a href="https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin">
 🎓 Faculty of Information Technology (DaiNam University)
    </a>
</h2>
<h2 align="center">
    Hệ Thống Phân Tích và Dự Báo Chứng Khoán
Tích Hợp
AI Tạo Sinh (Gemini) và Học Máy (XGBoost)
</h2>
<div align="center">
    <p align="center">
        <img src="aiotlab_logo.png" alt="AIoTLab Logo" width="170"/>
        <img src="fitdnu_logo (3).png" alt="AIoTLab Logo" width="180"/>
        <img src="dnu_logo.png" alt="DaiNam University Logo" width="200"/>
    </p>

[![AIoTLab](https://img.shields.io/badge/AIoTLab-green?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Faculty of Information Technology](https://img.shields.io/badge/Faculty%20of%20Information%20Technology-blue?style=for-the-badge)](https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-orange?style=for-the-badge)](https://dainam.edu.vn)

</div>

## 📖 1. Giới thiệu hệ thống
Đây là một ứng dụng Client-Server viết bằng Java Socket cho phép:

    - Người dùng (User) đăng ký, đăng nhập, xem danh sách sách, mượn sách.

    - Quản trị viên (Admin) thêm, sửa, xóa sách và duyệt yêu cầu mượn.

    - Server lưu dữ liệu vào file data.txt để quản lý người dùng, sách và danh sách mượn.

Cấu trúc chính:

    - Server.java: Xử lý kết nối, lưu dữ liệu, quản lý người dùng và sách.

    - ClientUser.java: Giao diện console cho người dùng.

    - ClientAdmin.java: Giao diện console cho quản trị viên.


## 🔧 2. Các công nghệ được sử dụng
  
- **☕ Java SE 8+**

- **🌐 Java Socket (TCP/IP)**

- **💾 File I/O (đọc/ghi dữ liệu vào data.txt)**

- **🖥 Eclipse IDE**
## 🚀 3. Một số hình ảnh hệ thống
<p align="center">
    <em>Giao diện khi kết nối thành công với Server</em><br/>
    <img width="1401" height="842" alt="Auto Send" src="Screenshot 2025-09-18 082422.png" />
</p>
<p align="center">
    <em>Giao diện khi người dùng đăng nhập</em><br/>
    <img width="1401" height="842" alt="Auto Send" src="Screenshot 2025-09-25 074950.png" />
</p>

<p align="center">
    <em>Giao diện người dùng</em><br/>
    <img width="1387" height="819" alt="UI Main" src="Screenshot 2025-09-25 100808.png" />
</p>

<p align="center">
    <em>Giao diện Admin</em><br/>
    <img width="1401" height="842" alt="Auto Send" src="Screenshot 2025-09-18 110641.png" />
</p>

---
## ⚙️ 4. Các bước cài đặt
### 4.1. Yêu cầu hệ thống
```
    - Cài đặt Java JDK 8+ (kiểm tra bằng lệnh java -version và javac -version).

    - Cài đặt Git để clone repository.

    - (Khuyến khích) Cài đặt Eclipse IDE hoặc IntelliJ IDEA để dễ quản lý project.
```
### 4.2. Cấu trúc thư mục
```
Du_an_quan_li_thu_vien/
    │── src/
    │   ├── Server/
    │   │   └── Server.java
    │   └── user/
    │       ├── ClientAdmin.java
    │       └── ClientUser.java
    │── data.txt

```

### 4.3. Chạy Server
```
    - Vào thư mục src/Server/Server.java.

    - Chuột phải → Run As → Java Application.

    - Server đã sẵn sàng lắng nghe kết nối từ client.
```
### 4.4. Chạy ClientUser (người dùng)

```
    - Vào thư mục src/user/ClientUser.java.

    - Chuột phải → Run As → Java Application.
```
### 4.5. Chạy ClientAdmin (quản trị viên)
```
    - Vào thư mục src/user/ClientAdmin.java.

    - Chuột phải → Run As → Java Application.
```

## 📝 5. Liên hệ

- Khoa: Công nghệ thông tin - Trường Đại học Đại Nam
- Lớp: CNTT 16-04
- Tôi: Nguyễn Văn Bắc
- Email: **nguyenbacdz04@gmail.com**  

---

<p align="center">
    ✍️ <em>README này được thiết kế bởi Bac Nguyen</em>
</p>

    
