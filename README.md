HƯỚNG DẪN SỬ DỤNG


1. GIỚI THIỆU:

Đề tài “Dự đoán kết quả học tập của học sinh” xây dựng hệ thống Machine Learning nhằm dự đoán điểm cuối kỳ G3 của học sinh dựa trên các yếu tố nhân khẩu học, xã hội và quá trình học tập.

2. YÊU CẦU TRƯỚC KHI SỬ DỤNG:

- Máy tính cần có:

    + Windows 10 hoặc Windows 11 64-bit.
    + Docker Desktop.
    + Tối thiểu khoảng 4 GB RAM trống.
    + Kết nối Internet trong lần build đầu tiên.
    + Các cổng 1433, 8000 và 8501 chưa bị chương trình khác sử dụng.

- Không bắt buộc cài đặt:

    + Python.
    + SQL Server.
    + FastAPI.
    + Streamlit.
    + Các thư viện trong requirements.txt.

3. CÀI ĐẶT VÀ KIỂM TRA DOCKER DESKTOP:

    3.1. Cài đặt Docker desktop: 

    Tải và cài đặt Docker Desktop cho Windows. Trong quá trình cài đặt, nên bật chế độ sử dụng WSL 2. Sau khi cài đặt xong, mở Docker Desktop và chờ đến khi Docker chuyển sang trạng thái đang chạy.

    3.2. Kiểm tra Docker:

    Mở PowerShell hoặc Command Prompt và chạy: 
        docker --version
        docker compose version
    Nếu cửa sổ hiện thị số phiên bản Docker và Docker compose thì quá trình cài đặt thành công.

4. Giải nén project:

Giải nén file Nhom9-De18-student-performance-ml.rar vào 1 thư mục dễ tìm thấy.
Cấu trúc thư mục có dạng như sau:
            Nhom9-De18-student-performance-ml/
        ├── api/
        ├── artifacts/
        ├── data/
        ├── database/
        ├── deployment/
        ├── src/
        ├── streamlit_app/
        ├── tests/
        ├── requirements.txt
        └── .env.example
        └── ...
* Lưu ý: nên chạy Docker tại thư mục gốc của project, không chạy trực tiếp bên trong thư mục deployment.

5. Tạo file cấu hình môi trường:

- Trong thư mục gốc của project, tạo file mới tên .env bằng cách sao chép và đổi tên file từ .env.example

- Mở file .env và cấu hình:
    API_BASE_URL=http://api:8000

    DB_AUTH_MODE=sql
    DB_SERVER=db,1433
    DB_NAME=StudentPerformanceDB
    DB_USER=sa
    DB_PASSWORD=StudentPerf_2026!

    MSSQL_SA_PASSWORD=StudentPerf_2026!

* Mật khẩu SQL Server phải đủ mạnh, có chữ hoa, chữ thường, số, ký tự đặc biệt và tối thiểu 8 ký tự.

6. Mở PowerShell tại thư mục Project

- Có thể mở File Explorer, vào thư mục project, nhấn vào thanh địa chỉ, nhập powershell rồi nhấn Enter.
- Hoặc vào thư mục project, ấn chuột phải vào không gian trống trong thư mục và chọn "Open in Terminal"

7. Khởi động hệ thống bằng Docker:

- Tại Terminal/Powershell của thư mục của project chạy:
    docker compose --env-file .env -f deployment/docker-compose.yml up --build
- Ý nghĩa của lệnh:

    + docker compose: chạy nhiều container cùng lúc.

    + --env-file .env: đọc các biến môi trường từ file .env.

    + -f deployment/docker-compose.yml: chỉ định file cấu hình Compose.

    + up: khởi động hệ thống.

    + --build: tạo hoặc cập nhật lại image từ Dockerfile.

    Trong lần chạy đầu tiên, Docker phải tải image Python, image SQL Server và cài thư viện nên có thể mất vài phút.

8. Kiểm tra trạng thái container

- Mở một cửa sổ PowerShell khác tại thư mục project và chạy:

    docker compose --env-file .env -f deployment/docker-compose.yml ps

- Trạng thái mong đợi:

    student-performance-db           Up (healthy)
    student-performance-api          Up (healthy)
    student-performance-streamlit    Up
    student-performance-init-db      Exited (0)

- Container init-db dừng với mã Exited (0) là bình thường vì nó chỉ chạy một lần để khởi tạo cơ sở dữ liệu.

9. Truy cập giao diện hệ thống

- Sau khi hệ thống khởi động thành công, mở trình duyệt và truy cập:

    http://localhost:8501

-Đây là giao diện chính của ứng dụng. Người dùng có thể nhập thông tin học sinh, chọn model và xem kết quả dự đoán.

10. Sử dụng chức năng dự đoán một học sinh

- Mở trang Predict trên giao diện Streamlit.
- Nhập các thông tin của học sinh.
- Chọn model phù hợp.
- Nhấn nút dự đoán.
- Xem kết quả điểm G3 do hệ thống trả về.
- Lịch sử dự đoán sẽ được lưu vào SQL Server nếu chức năng lưu lịch sử đã được bật.
- Các trường đầu vào có thể bao gồm môn học, trường, giới tính, tuổi, nơi ở, trình độ học vấn của cha mẹ, thời gian học, số lần nghỉ học và điểm G1, G2.
- Project có thể chia model thành hai nhóm:

    Model_Have_A_B: có sử dụng điểm quá trình, tương ứng với G1 và G2.
    Model_Dont_Have_A_B: không sử dụng điểm quá trình.

11. Sử dụng chức năng dự đoán hàng loạt

- Mở trang Batch Predict.
- Chuẩn bị file dữ liệu đúng cấu trúc mà hệ thống yêu cầu.
- Tải file lên giao diện.
- Chọn model.
- Thực hiện dự đoán.
- Xem hoặc tải kết quả dự đoán.

12. Kiểm tra FastAPI
    
    12.1. Kiểm tra trạng thái API

    http://localhost:8000/health
    - Nếu API hoạt động, kết quả thường có dạng:
        {
        "status": "ok"
        }

    12.2. Mở tài liệu API

        http://localhost:8000/docs
    - Đây là giao diện Swagger, cho phép kiểm tra trực tiếp các endpoint dự đoán, lịch sử và thông tin model.
    - Ngoài ra có thể truy cập:
        http://localhost:8000/redoc

13. Xem log hệ thống

- Xem toàn bộ log:
    docker compose --env-file .env -f deployment/docker-compose.yml logs
- Theo dõi log liên tục:
    docker compose --env-file .env -f deployment/docker-compose.yml logs -f
- Xem log từng service:
    docker compose --env-file .env -f deployment/docker-compose.yml logs db
    docker compose --env-file .env -f deployment/docker-compose.yml logs init-db
    docker compose --env-file .env -f deployment/docker-compose.yml logs api
    docker compose --env-file .env -f deployment/docker-compose.yml logs streamlit

14. Dừng và chạy lại hệ thống

- Dừng hệ thống:
    docker compose --env-file .env -f deployment/docker-compose.yml down
- Chạy lại hệ thống:
    docker compose --env-file .env -f deployment/docker-compose.yml up
- Nếu không thay đổi mã nguồn thì không cần thêm tham số --build.

15. Xóa và tạo lại toàn bộ database

- Để xóa container và volume dữ liệu SQL Server:
    docker compose --env-file .env -f deployment/docker-compose.yml down -v
- Sau đó chạy lại:
    docker compose --env-file .env -f deployment/docker-compose.yml up --build
- Cảnh báo: lệnh down -v sẽ xóa toàn bộ dữ liệu đã lưu, bao gồm lịch sử dự đoán.

16. Kết nối SQL Server bằng SSMS

- Thông tin kết nối: 

Thông tin                                       Giá trị

Server name                                     localhost,1433
Authentication                                  SQL Server Authentication
Login                                           sa
Password                                        Giá trị MSSQL_SA_PASSWORD trong file .env
Database                                        StudentPerformanceDB

- Nếu SSMS báo lỗi chứng chỉ, bật tùy chọn Trust server certificate.

17. Một số lỗi thường gặp và cách xử lý

Lỗi                                Cách xử lý

Docker Desktop chưa chạy        -  Mở Docker Desktop, chờ Docker hoạt động rồi chạy lại lệnh.
Không tìm thấy file Compose     -  Dùng đúng lệnh có tham số -f deployment/docker-
                                   compose.yml.
SQL Server không khởi động      -  Kiểm tra mật khẩu trong .env và xem log service db.
Cổng 1433 đã được sử dụng       -  Đổi ánh xạ cổng thành 1434:1433 trong docker-compose.yml.
Cổng 8000 đã được sử dụng       -  Đổi ánh xạ ngoài thành 8001:8000.
Cổng 8501 đã được sử dụng       -  Đổi ánh xạ ngoài thành 8502:8501.
Streamlit không gọi được API    -  Đảm bảo API_BASE_URL=http://api:8000 trong môi 
                                   trường container.
init-db hiển thị Exited (0)     -  Đây là trạng thái bình thường sau khi khởi tạo 
                                   xong database.

18. Các câu lệnh Docker thường dùng

Mục đích                  Câu lệnh

Khởi động và build    -   docker compose --env-file .env -f deployment/docker-
                          compose.yml up --build
Chạy nền              -   docker compose --env-file .env -f deployment/docker-
                          compose.yml up -d
Xem trạng thái        -   docker compose --env-file .env -f deployment/docker-
                          compose.yml ps
Xem log               -   docker compose --env-file .env -f deployment/docker-
                          compose.yml logs -f
Dừng hệ thống         -   docker compose --env-file .env -f deployment/docker-
                          compose.yml down
Khởi động lại         -   docker compose --env-file .env -f deployment/docker-
                          compose.yml restart
Xóa cả dữ liệu        -   docker compose --env-file .env -f deployment/docker-
                          compose.yml down -v

19. Quy trình sử dụng tóm tắt

- Mở Docker Desktop.
- Mở PowerShell tại thư mục project.
- Chạy lệnh docker compose để khởi động hệ thống.
- Truy cập http://localhost:8501.
- Thực hiện dự đoán trên giao diện.
- Dừng hệ thống bằng docker compose down sau khi sử dụng.

20. Kết luận

- Docker giúp hệ thống dự đoán kết quả học tập có thể chạy đồng nhất trên nhiều máy tính. SQL Server, FastAPI, Streamlit, thư viện Python và model Machine Learning đều được đóng gói trong các container.
- Câu lệnh quan trọng nhất để khởi động hệ thống là:
    docker compose --env-file .env -f deployment/docker-compose.yml up --build
- Sau khi khởi động, người dùng truy cập:
    Giao diện: http://localhost:8501
    API:       http://localhost:8000/docs
    Kiểm tra:  http://localhost:8000/health