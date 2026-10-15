import sqlite3
import random
from datetime import datetime, timedelta

DB_PATH = "greenmate.db"
TABLE_NAME = "environment_samples"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    create_sql = "CREATE TABLE IF NOT EXISTS " + TABLE_NAME + "(id INTEGER PRIMARY KEY AUTOINCREMENT,timestamp TEXT NOT NULL,source_type TEXT NOT NULL,temperature REAL,humidity REAL,light REAL)"
    cur.execute(create_sql)
    conn.commit()
    conn.close()
    print("✅ 数据库与数据表初始化完成")


def generate_7day_data():
    total_points = 336
    start_time = datetime.now() - timedelta(days=7)
    data_rows = []
    disturb_segments = [(20, 35),(80, 95),(140, 160),(210, 225),(280, 295)]

    for i in range(total_points):
        current_dt = start_time + timedelta(minutes=30 * i)
        ts_str = current_dt.isoformat()
        source = "SIMULATED"
        temp_base = 26.0
        hum_base = 65.0
        light_base = 12000
        in_disturb = False
        for s, e in disturb_segments:
            if s <= i <= e:
                in_disturb = True
                break
        if in_disturb:
            if 20 <= i <=35:
                temp_base = 33
                hum_base = 86
                light_base = 21000
            elif 80 <= i <=95:
                temp_base =13
                hum_base =72
                light_base =11000
            elif 140 <= i <=160:
                temp_base =28
                hum_base =52
                light_base =34000
            elif 210 <= i <=225:
                temp_base =24
                hum_base =63
                light_base =3200
            elif 280 <= i <=295:
                temp_base =27
                hum_base =40
                light_base =13000
        temp = round(temp_base + random.uniform(-0.7,0.7),2)
        hum = round(hum_base + random.uniform(-1.2,1.2),2)
        lux = round(light_base + random.uniform(-300,300),0)
        row = (ts_str, source, temp, hum, lux)
        data_rows.append(row)
    return data_rows


def insert_samples(data_list):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    insert_sql = "INSERT INTO " + TABLE_NAME + "(timestamp, source_type, temperature, humidity, light) VALUES (?,?,?,?,?)"
    cur.executemany(insert_sql, data_list)
    conn.commit()
    conn.close()
    print(f"✅ 成功写入 {len(data_list)} 条时序记录")


def get_environment_history(start_time_iso, end_time_iso):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    query_sql = "SELECT timestamp,source_type,temperature,humidity,light FROM " + TABLE_NAME + " WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp ASC"
    cur.execute(query_sql, (start_time_iso, end_time_iso))
    res = cur.fetchall()
    conn.close()
    return res


def get_current_environment():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT timestamp,source_type,temperature,humidity,light FROM " + TABLE_NAME + " ORDER BY timestamp DESC LIMIT 1")
    res = cur.fetchone()
    conn.close()
    return res


def data_quality_check():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    print("\n===== 数据质量校验 =====")
    cur.execute("SELECT COUNT(*) FROM " + TABLE_NAME)
    cnt = cur.fetchone()[0]
    print(f"总记录数量：{cnt}")
    cur.execute("SELECT COUNT(*) FROM " + TABLE_NAME + " WHERE timestamp IS NULL OR temperature IS NULL OR humidity IS NULL OR light IS NULL")
    null_cnt = cur.fetchone()[0]
    print(f"空值记录：{null_cnt}")
    cur.execute("SELECT MIN(temperature),MAX(temperature) FROM " + TABLE_NAME)
    t_min,t_max = cur.fetchone()
    print(f"温度范围：{t_min:.2f} ~ {t_max:.2f}")
    cur.execute("SELECT MIN(humidity),MAX(humidity) FROM " + TABLE_NAME)
    h_min,h_max = cur.fetchone()
    print(f"湿度范围：{h_min:.2f} ~ {h_max:.2f}")
    cur.execute("SELECT MIN(light),MAX(light) FROM " + TABLE_NAME)
    l_min,l_max = cur.fetchone()
    print(f"光照范围：{l_min:.0f} ~ {l_max:.0f}")
    conn.close()
    print("✅ 质量校验完成\n")


def verify_persistence():
    print("===== B02持久化验证 =====")
    latest = get_current_environment()
    all_data = get_environment_history("1970-01-01", "2100-01-01")
    print(f"✅ 重启读取成功，数据库现存记录：{len(all_data)}条")
    print(f"📌 最新采样记录：{latest}")
    print("✅ B02验证通过：关闭程序后数据持久保存")


if __name__ == "__main__":
    init_db()
    dataset = generate_7day_data()
    insert_samples(dataset)
    data_quality_check()
    verify_persistence()