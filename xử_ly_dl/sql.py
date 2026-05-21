import pandas as pd
from sqlalchemy import create_engine
import urllib

# Cấu hình SQL Server (Port 1433)
server = 'localhost,1433'    
database = 'fintech'         
username = 'sa'              
password = 'huong6978'       

try:
    # 1. Tạo kết nối tới SQL Server
    print("Đang kết nối tới SQL Server...")
    connection_string = (
        r'DRIVER={ODBC Driver 17 for SQL Server};'
        rf'SERVER={server};'
        rf'DATABASE={database};'
        rf'UID={username};'
        rf'PWD={password};'
    )
    params = urllib.parse.quote_plus(connection_string)
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

    # 2. Đọc dữ liệu thô từ bảng 'credit_risk' bằng câu lệnh SELECT cụ thể
    print("Đang đọc dữ liệu từ bảng credit_risk...")
    
    query = """
        SELECT 
            person_age, 
            person_income, 
            person_home_ownership, 
            person_emp_length, 
            loan_intent, 
            loan_amnt, 
            loan_int_rate, 
            loan_percent_income, 
            cb_person_default_on_file,
            loan_status -- Target
        FROM credit_risk
    """
    df_filtered = pd.read_sql(query, engine)

    # 3. Lưu bảng dữ liệu đã lọc sang bảng mới (credit_risk_staging)
    table_name = 'credit_risk_staging'
    print(f"Đang lưu dữ liệu sang bảng staging: {table_name}...")
    
    df_filtered.to_sql(table_name, engine, if_exists='replace', index=False)

    print(f"Thành công! Đã chuyển dữ liệu từ 'credit_risk' sang '{table_name}' với {df_filtered.shape[1]} cột và {len(df_filtered)} dòng.")

except Exception as e:
    print(f"Có lỗi xảy ra: {e}")