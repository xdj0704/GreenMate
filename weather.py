"""
GreenMate B03 · 外部天气预报模块
---------------------------------
数据源：Open-Meteo（https://open-meteo.com）
  - 完全免费，无需 API Key，无需注册
  - 数据来自 ECMWF / NOAA / DWD 等国家气象服务
  - provider 标记为 PUBLIC，与本地 SIMULATED 数据严格区分

用法（独立运行验证）：
    python weather.py 南京
    python weather.py 苏州
    python weather.py --fail          # 制造失败场景
"""

import sys
import datetime
import requests

# Windows GBK 终端兼容：强制 stdout/stderr 使用 UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── 地理编码：城市名 → 经纬度（Open-Meteo Geocoding API）────────────────────
GEO_URL      = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# 需要提取的小时级预报字段（不透传原始 JSON，只取核心字段）
HOURLY_VARS = [
    "temperature_2m",           # 气温 ℃
    "relative_humidity_2m",     # 相对湿度 %
    "precipitation_probability",# 降水概率 %
    "cloud_cover",              # 云量 %
    "shortwave_radiation",      # 短波辐射 W/m²（光照代理指标）
]

REQUEST_TIMEOUT = 10  # 秒，超时即触发失败场景


def _geocode(location: str) -> tuple[float, float, str]:
    """
    城市名 → (latitude, longitude, display_name)。
    失败抛出 RuntimeError。
    """
    resp = requests.get(
        GEO_URL,
        params={"name": location, "count": 1, "language": "zh", "format": "json"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    results = resp.json().get("results")
    if not results:
        raise RuntimeError(f"地理编码失败：找不到城市 '{location}'")
    r = results[0]
    return r["latitude"], r["longitude"], r.get("name", location)


def _extract_forecast(raw: dict, lat: float, lon: float,
                       display_name: str) -> dict:
    """
    从原始 JSON 中提取最近一个整点小时的预报值。
    只返回核心字段，不透传原始 JSON。
    """
    hourly = raw.get("hourly", )
    times  = hourly.get("time", [])
    if not times:
        raise RuntimeError("预报数据为空，hourly.time 缺失")

    # 取距当前最近的未来整点（或当前小时）
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:00")
    idx = 0
    for i, t in enumerate(times):
        if t >= now_str:
            idx = i
            break

    def _val(key):
        arr = hourly.get(key, [])
        return arr[idx] if idx < len(arr) else None

    return {
        "location":               display_name,
        "latitude":               lat,
        "longitude":              lon,
        "forecast_time":          times[idx] if idx < len(times) else None,
        "temperature":            _val("temperature_2m"),
        "relative_humidity":      _val("relative_humidity_2m"),
        "precipitation_probability": _val("precipitation_probability"),
        "cloud_cover":            _val("cloud_cover"),
        "radiation":              _val("shortwave_radiation"),
    }


def fetch_weather(location: str,
                  timeout: int = REQUEST_TIMEOUT) -> dict:
    """
    主入口：城市名 → weather_evidence 结构体。

    返回结构固定包含：
      provider, retrieved_at, query, status, error,
      以及核心预报字段（status=ok 时填充）。
    失败时 status='error'，error 字段说明原因，核心字段置 None。
    """
    retrieved_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    base_evidence = {
        "provider":     "Open-Meteo (open-meteo.com)",
        "retrieved_at": retrieved_at,
        "query":        location,
        "status":       "error",
        "error":        None,
        # 核心字段（失败时为 None）
        "location":               None,
        "forecast_time":          None,
        "temperature":            None,
        "relative_humidity":      None,
        "precipitation_probability": None,
        "cloud_cover":            None,
        "radiation":              None,
    }

    try:
        # ① 地理编码
        lat, lon, display_name = _geocode(location)

        # ② 获取预报
        resp = requests.get(
            FORECAST_URL,
            params={
                "latitude":  lat,
                "longitude": lon,
                "hourly":    ",".join(HOURLY_VARS),
                "forecast_days": 1,
                "timezone":  "Asia/Shanghai",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        raw = resp.json()

        # ③ 提取核心字段
        forecast = _extract_forecast(raw, lat, lon, display_name)
        base_evidence.update({"status": "ok", "error": None, **forecast})

    except requests.exceptions.Timeout:
        base_evidence["error"] = f"请求超时（>{timeout}s）"
    except requests.exceptions.HTTPError as e:
        base_evidence["error"] = f"HTTP错误 {e.response.status_code}: {e}"
    except RuntimeError as e:
        base_evidence["error"] = str(e)
    except Exception as e:
        base_evidence["error"] = f"{type(e).__name__}: {e}"

    return base_evidence


def _print_evidence(ev: dict, label: str = "") -> None:
    """控制台打印 weather_evidence，风格与 GreenMate S1-S5 run_log 一致。"""
    tag = f" · {label}" if label else ""
    print(f"\n[Weather] {'='*50}")
    print(f"[Weather] 查询地点：{ev['query']}{tag}")
    print(f"[Weather] provider ：{ev['provider']}")
    print(f"[Weather] retrieved：{ev['retrieved_at']}")
    print(f"[Weather] status   ：{ev['status']}")
    if ev["status"] == "ok":
        print(f"[Weather] 解析地点：{ev['location']}  "
              f"({ev.get('latitude')}, {ev.get('longitude')})")
        print(f"[Weather] 预报时刻：{ev['forecast_time']}")
        print(f"[Weather]   温度={ev['temperature']}℃  "
              f"湿度={ev['relative_humidity']}%  "
              f"降水概率={ev['precipitation_probability']}%  "
              f"云量={ev['cloud_cover']}%  "
              f"辐射={ev['radiation']}W/m²")
    else:
        print(f"[Weather] error    ：{ev['error']}")
    print(f"[Weather] {'='*50}\n")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "南京"

    if arg == "--fail":
        # FailureTest：传入无效地名，触发地理编码失败
        print("[Weather] ▶ FailureTest：传入无效地名 'INVALID_CITY_XYZ'")
        ev = fetch_weather("INVALID_CITY_XYZ")
        _print_evidence(ev, label="FailureTest-无效地名")

        # FailureTest：制造超时（timeout=0.001s）
        print("[Weather] ▶ FailureTest：强制超时（timeout=0.001s）")
        ev2 = fetch_weather("南京", timeout=0.001)
        _print_evidence(ev2, label="FailureTest-超时")
    else:
        print(f"[Weather] ▶ 查询：{arg}")
        ev = fetch_weather(arg)
        _print_evidence(ev)

