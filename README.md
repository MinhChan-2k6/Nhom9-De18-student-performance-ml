# I. LINK REPO GITHUB: 
`https://github.com/MinhChan-2k6/Nhom9-De18-student-performance-ml`
## GIỚI THIỆU:
Đề tài “Dự đoán kết quả học tập của học sinh” xây dựng hệ thống Machine Learning nhằm dự đoán điểm cuối kỳ G3 của học sinh dựa trên các yếu tố nhân khẩu học, xã hội và quá trình học tập.

# II. LINK VIDEO DEMO: 
`LINK VIDEO`

# III. BẢNG PHÂN CÔNG CÔNG VIỆC VÀ % ĐÓNG GÓP
## Phân công công việc và tỷ lệ đóng góp


|--------------------------|--------------------|--------------|------------------------------------------------------|----------------|
|         Họ và tên        |        MSSV        |   Vai trò    |                 Công việc thực hiện                  | Tỷ lệ đóng góp |
|--------------------------|--------------------|--------------|------------------------------------------------------|----------------|
| Phạm Minh Chấn           |     2451012017     | Trưởng nhóm  | Quản lý Repo Github, Hưỡng dẫn, điều hành công việc  |                |
|                          |                    |Điều hành, hỗ | Kiểm thử hệ thống, xây dựng docker, kiểm cuối dự án  |       25%      |
|                          |                    |trợ các thành | video demo, chung với Thiên Tiên viết báo cáo.       |                |
|                          |                    |viên          |                                                      |                |
|--------------------------|--------------------|--------------|------------------------------------------------------|----------------|
| Nguyễn Thị Huyền Trang   |     2451012116     | Thành viên   | Thu thập dữ liệu gốc, kiểm tra, làm sạch, phân tích  |                |
|                          |                    |Dữ liệu và EDA| dữ liệu chia tập dữ liệu và xây dựng các biểu đồ     |       25%      |
|                          |                    |              | thống kê, thiết kế sơ đồ kiến trúc, viết Readme.md   |                |
|--------------------------|--------------------|--------------|------------------------------------------------------|----------------|
| Nguyễn Phan Lan Thanh    |     2451012090     | Thành viên   | Xây dựng FastAPI, thiết kế cơ sở dữ liệu, xây dựng   |                |
|                          |                    |API, Streamlit| giao diện Streamlit, tích hợp chức năng dự đoán và   |       25%      |
|                          |                    |,Database     | lưu lịch sử dự đoán.                                 |                |
|--------------------------|--------------------|--------------|------------------------------------------------------|----------------|
| Lê Nguyễn Thiên Tiên     |     2451012108     | Thành viên   | Tiền xử lý đặc trưng, xây dựng mô hình, đánh giá bằng|                |
|                          |                    |Xây dựng, đánh| MAE, RMSE, R², và lưu lịch sử dự đoán mô hình, viết  |       25%      |
|                          |                    |giá mô hình   | báo cáo, kiểm tra lại file Readme.md                 |                |
|--------------------------|--------------------|--------------|------------------------------------------------------|----------------|
|                            TỔNG                                                                                     |       100%     |
|---------------------------------------------------------------------------------------------------------------------|----------------|


# IV. KHAI BÁO SỬ DỤNG CÔNG CỤ AI


|--------------------------|------------------------------------------------|----------------------------------------------------------|
|    Công cụ AI sử dụng    |                 Mục đích sử dụng               |                 Phạm vi áp dụng vào bài                  |
|--------------------------|------------------------------------------------|----------------------------------------------------------|
|         ChatGPT          | Hỗ trợ phân tích yêu cầu đề tài, gợi ý hướng   | Áp dụng trong quá trình tìm hiểu bài toán, xây dựng cấu  |
|         (PLUS)           | phát triên đề tài, giải thích quy trình xử lý  | trúc, tiền xử lý dữ liệu, huấn luyện mô hình, đánh giá   |
|                          | dữ liệu và pipeline học máy.                   | kết quả và độ hoàn thiện của dự án.                      |
|                          |------------------------------------------------|                                                          |
|                          | Hỗ trợ code, giải thích lỗi, sửa code, kiến    |                                                          |
|                          | thức không hiểu, gợi ý cách hướng code         |                                                          |
|                          |------------------------------------------------|                                                          |
|                          | Hỗ trợ kiến thức về Repo Github, cách dùng GIT |                                                          |
|                          | Hỗ trợ cách dùng và kiến thức cơ bản Docker    |                                                          |
|--------------------------|------------------------------------------------|----------------------------------------------------------| 
|          Claude          | Hỗ trợ kiểm tra logic, mã nguồn, phát hiện lỗi | Áp dụng trong quá trình cần rà xoát lại code, sửa lỗi khi|
|          (FREE)          | và gợi ý cách sửa các phần chưa hiểu rõ, cải   | chạy code không đúng, hỗ trợ hoàn thiện các file code    |
|                          | thiện cách tổ chức một số chương trình.        | chưa rõ cách triển khai.                                 |
|--------------------------|-----------------------------------------------------------------------------------------------------------|




# V. FILE MODEL ĐÃ LƯU & HƯỚNG DẪN SỬ DỤNG
## ĐƯỜNG DẪN TỚI FILE MODEL
- Nhóm sử dụng 2 model được huấn luyện và lưu tại:
    + Model_Dont_Have_A_B - Dự đoán G3 không sử dụng giá trị G1 và G2: 
        `artifacts/Model_Dont_Have_A_B/best_regression_model.pkl`
    + Model_Have_A_B - Dự đoán G3 có sử dụng giá trị G1 và G2: 
        `artifacts/Model_Have_A_B/best_regression_model.pkl`

## HƯỠNG DẪN SỬ DỤNG
1. YÊU CẦU TRƯỚC KHI SỬ DỤNG VÀ CHÚ THÍCH:
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
- Chú thích:
>Dòng chữ màu xanh là Code, có thể coppy paste để chạy lệnh nhưng chú ý coppy chuỗi lệnh SAU ký tự ">"

2. CÀI ĐẶT VÀ KIỂM TRA DOCKER DESKTOP:
    2.1. Cài đặt Docker desktop.
    2.2. Kiểm tra Docker.
        Mở Terminal hoặc Command Prompt và chạy:
            docker --version
            docker compose version
        Nếu cửa sổ hiện thị số phiên bản Docker và Docker compose thì quá trình cài đặt thành công.

3. Giải nén project:
Giải nén file Nhom9-De18-student-performance-ml.rar vào 1 thư mục dễ tìm thấy.
* Lưu ý: nên chạy Docker tại thư mục gốc của project, không chạy trực tiếp bên trong thư mục deployment.

4. Tạo file cấu hình môi trường:
- Trong thư mục gốc của project, tạo file mới tên .env bằng cách sao chép và đổi tên file từ .env.example
- Mở file .env và cấu hình:
``
    API_BASE_URL=http://api:8000

    DB_AUTH_MODE=sql
    DB_SERVER=db,1433
    DB_NAME=StudentPerformanceDB
    DB_USER=sa

    DB_PASSWORD=<Mật khẩu>
    MSSQL_SA_PASSWORD=<Mật khẩu>
``
* Mật khẩu SQL Server phải đủ mạnh, có chữ hoa, chữ thường, số, ký tự đặc biệt và tối thiểu 8 ký tự.
* 2 trường DB_PASSWORD và MSSQL_SA_PASSWORD phải nhập mật khẩu giống nhau!

5. Mở Terminal tại thư mục Project
- Có thể mở File Explorer, vào thư mục project, nhấn vào thanh địa chỉ, nhập terminal rồi nhấn Enter.
- Hoặc vào thư mục project, ấn chuột phải vào không gian trống trong thư mục và chọn "Open in Terminal"

6. Khởi động hệ thống bằng Docker:
- Tại Terminal/Command Prompt của thư mục của project chạy 
>docker compose --env-file .env -f deployment/docker-compose.yml up --build
Trong lần chạy đầu tiên, Docker phải tải image Python, image SQL Server và cài thư viện nên có thể mất vài phút.

7. Kiểm tra trạng thái container
- Mở một cửa sổ Terminal khác tại thư mục project và chạy:

>docker compose --env-file .env -f deployment/docker-compose.yml ps

- Trạng thái mong đợi:
``
    student-performance-db           Up (healthy)
    student-performance-api          Up (healthy)
    student-performance-streamlit    Up
    student-performance-init-db      Exited (0)
``
- Container init-db dừng với mã Exited (0) là bình thường vì nó chỉ chạy một lần để khởi tạo cơ sở dữ liệu.

8. Truy cập giao diện hệ thống
- Sau khi hệ thống khởi động thành công, mở trình duyệt và truy cập: http://localhost:8501
-Đây là giao diện chính của ứng dụng. Người dùng có thể nhập thông tin học sinh, chọn model và xem kết quả dự đoán.

9. Sử dụng chức năng dự đoán một học sinh
- Mở trang Predict trên giao diện Streamlit.
- Nhập các thông tin của học sinh.
- Chọn model phù hợp và nhập thông tin học sinh.
- Nhấn nút dự đoán.
- Xem kết quả điểm G3 do hệ thống trả về.
- Lịch sử dự đoán sẽ được lưu vào SQL Server nếu chức năng lưu lịch sử đã được bật.
- Project có thể chia model thành hai nhóm:
    Model_Have_A_B: có sử dụng điểm quá trình, tương ứng với G1 và G2.
    Model_Dont_Have_A_B: không sử dụng điểm quá trình.

10. Sử dụng chức năng dự đoán hàng loạt
- Mở trang Batch Predict.
- Chuẩn bị file dữ liệu đúng cấu trúc mà hệ thống yêu cầu.
- Tải file lên giao diện.
- Chọn model.
- Thực hiện dự đoán.
- Xem hoặc tải kết quả dự đoán.

11. Dừng và chạy lại hệ thống
- Dừng hệ thống:
>docker compose --env-file .env -f deployment/docker-compose.yml down
- Chạy lại hệ thống:
>docker compose --env-file .env -f deployment/docker-compose.yml up
- Nếu không thay đổi mã nguồn thì không cần thêm tham số --build.

12. Kết nối SQL Server bằng SSMS
- Thông tin kết nối: 

Thông tin                                       Giá trị

Server name                                     localhost,1433
Authentication                                  SQL Server Authentication
Login                                           sa
Password                                        Giá trị MSSQL_SA_PASSWORD trong file .env
Database                                        StudentPerformanceDB

- Nếu SSMS báo lỗi chứng chỉ, bật tùy chọn Trust server certificate.

13. Một số lỗi thường gặp và cách xử lý
Lỗi                                Cách xử lý

Docker Desktop chưa chạy        -  Mở Docker Desktop, chờ Docker hoạt động rồi chạy lại lệnh.
Không tìm thấy file Compose     -  Dùng đúng lệnh có tham số -f deployment/docker-compose.yml.
SQL Server không khởi động      -  Kiểm tra mật khẩu trong .env và xem log service db.
Cổng 1433 đã được sử dụng       -  Đổi ánh xạ cổng thành 1434:1433 trong docker-compose.yml.
Cổng 8000 đã được sử dụng       -  Đổi ánh xạ ngoài thành 8001:8000.
Cổng 8501 đã được sử dụng       -  Đổi ánh xạ ngoài thành 8502:8501.
Streamlit không gọi được API    -  Đảm bảo API_BASE_URL=http://api:8000 trong môi trường container.
init-db hiển thị Exited (0)     -  Đây là trạng thái bình thường sau khi khởi tạo xong database.

14. Kết luận
- Docker giúp hệ thống dự đoán kết quả học tập có thể chạy đồng nhất trên nhiều máy tính. SQL Server, FastAPI, Streamlit, thư viện Python và model Machine Learning đều được đóng gói trong các container.
- Câu lệnh quan trọng nhất để khởi động hệ thống là:
    docker compose --env-file .env -f deployment/docker-compose.yml up --build
- Sau khi khởi động, người dùng truy cập:
    Giao diện: http://localhost:8501
    API:       http://localhost:8000/docs
    Kiểm tra:  http://localhost:8000/health