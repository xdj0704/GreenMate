"""B02 数据接入模块：统一将外部数据源转换为引擎标准输入格式。

支持三种接入方式：
  1. CSV 文件导入（离线历史数据）
  2. HTTP 轮询（HTTP GET JSON API）
  3. MQTT 订阅（实时传感器推送）

输出格式统一为引擎 timeseries：list of (时刻str, 温度, 湿度, 光照, CO₂)
"""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable

logger = logging.getLogger("greenmate.ingest")

# ── 标准行格式 ──────────────────────────────────────────────────
# (time_str, temp_c, humidity_pct, light_lux, co2_ppm)
Row = tuple[str, float, float, float, float]

# 引擎期望的字段名（用于 HTTP JSON 响应映射）
_FIELD_ALIASES = {
    "temperature": ("temp", "temperature", "temperature_c", "temp_c", "t"),
    "humidity":    ("humidity", "humidity_pct", "rh", "h", "hum"),
    "light":       ("light", "light_lux", "lux", "illuminance"),
    "co2":         ("co2", "co2_ppm", "carbon_dioxide", "ppm"),
}


def _extract(record: dict) -> Row:
    """从任意扁平 JSON 对象中提取标准字段，字段名大小写不敏感。"""
    lower = {k.lower(): v for k, v in record.items()}

    def pick(aliases: tuple) -> float:
        for alias in aliases:
            if alias in lower:
                return float(lower[alias])
        raise KeyError(f"未找到字段（尝试了 {aliases}）")

    ts = (
        lower.get("time")
        or lower.get("timestamp")
        or lower.get("datetime")
        or datetime.now().strftime("%H:%M")
    )
    # 如果是 ISO 格式取 HH:MM
    if "T" in str(ts) or (len(str(ts)) > 8 and "-" in str(ts)):
        try:
            ts = datetime.fromisoformat(str(ts)).strftime("%H:%M")
        except ValueError:
            ts = str(ts)[:5]

    return (
        str(ts),
        pick(_FIELD_ALIASES["temperature"]),
        pick(_FIELD_ALIASES["humidity"]),
        pick(_FIELD_ALIASES["light"]),
        pick(_FIELD_ALIASES["co2"]),
    )


# ══════════════════════════════════════════════════════════════════
# 1. CSV 导入
# ══════════════════════════════════════════════════════════════════

def from_csv(path: str, time_col: str = "time") -> list[Row]:
    """从 CSV 文件读取历史时序数据。

    CSV 要求包含列：time（或 timestamp）、temperature、humidity、light、co2
    列名大小写不敏感，顺序无关。

    示例 CSV：
        time,temperature,humidity,light,co2
        08:30,23.6,57,9800,812
        09:30,24.0,58,10200,820

    Args:
        path: CSV 文件路径
        time_col: 时间列名（默认 "time"）

    Returns:
        list[Row] 按文件顺序排列

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 缺少必要列或数值无法解析
    """
    rows: list[Row] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 文件为空或无表头：{path}")
        for i, record in enumerate(reader, start=2):
            try:
                rows.append(_extract(record))
            except (KeyError, ValueError) as e:
                raise ValueError(f"CSV 第 {i} 行解析失败：{e}") from e
    if not rows:
        raise ValueError(f"CSV 文件无数据行：{path}")
    logger.info("CSV 导入完成：%s 行，来源 %s", len(rows), path)
    return rows


# ══════════════════════════════════════════════════════════════════
# 2. HTTP 轮询
# ══════════════════════════════════════════════════════════════════

class HTTPPoller:
    """定期 GET 一个 JSON API，累积最近 N 条读数。

    Args:
        url: 传感器 API 端点，响应需为 JSON 对象或对象数组
        interval_s: 轮询间隔（秒），默认 60
        window: 保留最近多少条（默认 7，对应约 6h@1条/h）
        on_update: 每次新数据到达时调用的回调，参数为 list[Row]
        timeout_s: 单次请求超时（秒）

    用法：
        poller = HTTPPoller("http://sensor-api/latest", interval_s=60)
        poller.start()
        # ... 在主循环里
        timeseries = poller.get_timeseries()
        poller.stop()
    """

    def __init__(
        self,
        url: str,
        interval_s: int = 60,
        window: int = 7,
        on_update: Callable[[list[Row]], None] | None = None,
        timeout_s: int = 10,
    ):
        self.url = url
        self.interval_s = interval_s
        self.window = window
        self.on_update = on_update
        self.timeout_s = timeout_s

        self._buffer: list[Row] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _fetch_once(self) -> None:
        try:
            import urllib.request
            req = urllib.request.Request(
                self.url,
                headers={"Accept": "application/json", "User-Agent": "GreenMate-B02/1.0"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.warning("HTTP 轮询失败 [%s]：%s", self.url, e)
            return

        # 支持单对象 {} 或数组 [{}]
        records = data if isinstance(data, list) else [data]
        new_rows: list[Row] = []
        for record in records:
            try:
                new_rows.append(_extract(record))
            except (KeyError, ValueError) as e:
                logger.warning("HTTP 响应字段解析失败：%s — %s", e, record)

        if not new_rows:
            return

        with self._lock:
            self._buffer.extend(new_rows)
            # 保留最近 window 条
            if len(self._buffer) > self.window:
                self._buffer = self._buffer[-self.window:]

        logger.debug("HTTP 轮询新增 %d 条，缓冲区 %d 条", len(new_rows), len(self._buffer))
        if self.on_update:
            self.on_update(list(self._buffer))

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._fetch_once()
            self._stop_event.wait(self.interval_s)

    def start(self) -> "HTTPPoller":
        """启动后台轮询线程。"""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="greenmate-http-poller")
        self._thread.start()
        logger.info("HTTP 轮询已启动：%s（每 %ds）", self.url, self.interval_s)
        return self

    def stop(self) -> None:
        """停止轮询线程。"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s + 2)
        logger.info("HTTP 轮询已停止")

    def get_timeseries(self) -> list[Row]:
        """返回当前缓冲区的时序副本。"""
        with self._lock:
            return list(self._buffer)

    def ready(self) -> bool:
        """是否已有足够数据（至少3条）。"""
        with self._lock:
            return len(self._buffer) >= 3


# ══════════════════════════════════════════════════════════════════
# 3. MQTT 订阅
# ══════════════════════════════════════════════════════════════════

class MQTTSubscriber:
    """订阅 MQTT topic，累积最近 N 条读数。

    需要安装：pip install paho-mqtt>=2.0.0

    Args:
        broker: MQTT broker 地址（如 "192.168.1.100"）
        port: MQTT 端口（默认 1883）
        topic: 订阅的 topic（支持通配符，如 "greenhouse/A/#"）
        window: 保留最近多少条（默认 7）
        on_update: 新数据回调
        username / password: 可选认证
        client_id: MQTT client ID

    用法：
        sub = MQTTSubscriber("192.168.1.100", topic="greenhouse/A/sensors")
        sub.start()
        timeseries = sub.get_timeseries()
        sub.stop()

    消息格式（JSON）：
        {"time":"14:30","temperature":26.2,"humidity":65,"light":11900,"co2":847}
    """

    def __init__(
        self,
        broker: str,
        port: int = 1883,
        topic: str = "greenhouse/#",
        window: int = 7,
        on_update: Callable[[list[Row]], None] | None = None,
        username: str | None = None,
        password: str | None = None,
        client_id: str = "greenmate-b02",
    ):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.window = window
        self.on_update = on_update
        self.username = username
        self.password = password
        self.client_id = client_id

        self._buffer: list[Row] = []
        self._lock = threading.Lock()
        self._client = None

    def _on_message(self, _client, _userdata, msg) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            row = _extract(payload)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("MQTT 消息解析失败 [%s]：%s", msg.topic, e)
            return

        with self._lock:
            self._buffer.append(row)
            if len(self._buffer) > self.window:
                self._buffer = self._buffer[-self.window:]

        logger.debug("MQTT 收到：%s", row)
        if self.on_update:
            self.on_update(list(self._buffer))

    def start(self) -> "MQTTSubscriber":
        """连接 broker 并开始订阅。"""
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except ImportError as e:
            raise ImportError("MQTT 接入需要 paho-mqtt：pip install paho-mqtt>=2.0.0") from e

        client = mqtt.Client(client_id=self.client_id)
        if self.username:
            client.username_pw_set(self.username, self.password)
        client.on_message = self._on_message
        client.connect(self.broker, self.port, keepalive=60)
        client.subscribe(self.topic)
        client.loop_start()
        self._client = client
        logger.info("MQTT 已连接：%s:%d，topic=%s", self.broker, self.port, self.topic)
        return self

    def stop(self) -> None:
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            logger.info("MQTT 已断开")

    def get_timeseries(self) -> list[Row]:
        with self._lock:
            return list(self._buffer)

    def ready(self) -> bool:
        with self._lock:
            return len(self._buffer) >= 3


# ══════════════════════════════════════════════════════════════════
# 4. 统一入口：IngestSource
# ══════════════════════════════════════════════════════════════════

class IngestSource:
    """统一数据接入门面，隔离引擎与具体数据源。

    用法示例：

        # CSV
        src = IngestSource.from_csv("data/history.csv")

        # HTTP 轮询
        src = IngestSource.from_http("http://sensor-api/latest", interval_s=60)
        src.start()

        # MQTT
        src = IngestSource.from_mqtt("192.168.1.100", topic="greenhouse/A/sensors")
        src.start()

        # 获取引擎标准输入
        timeseries = src.get_timeseries()
        # -> list[Row] 可直接传给 render.build_context()
    """

    def __init__(self, rows: list[Row] | None = None, _backend=None):
        self._static_rows = rows        # CSV / 静态
        self._backend = _backend        # HTTPPoller / MQTTSubscriber

    # ── 工厂方法 ──────────────────────────────────────────────────

    @classmethod
    def from_csv(cls, path: str) -> "IngestSource":
        """从 CSV 文件构建静态数据源。"""
        rows = from_csv(path)
        return cls(rows=rows)

    @classmethod
    def from_http(cls, url: str, interval_s: int = 60, window: int = 7,
                  on_update: Callable | None = None) -> "IngestSource":
        """构建 HTTP 轮询数据源（需调用 start() 启动）。"""
        backend = HTTPPoller(url, interval_s=interval_s, window=window, on_update=on_update)
        return cls(_backend=backend)

    @classmethod
    def from_mqtt(cls, broker: str, port: int = 1883, topic: str = "greenhouse/#",
                  window: int = 7, on_update: Callable | None = None,
                  username: str | None = None, password: str | None = None) -> "IngestSource":
        """构建 MQTT 订阅数据源（需调用 start() 启动）。"""
        backend = MQTTSubscriber(broker, port=port, topic=topic, window=window,
                                 on_update=on_update, username=username, password=password)
        return cls(_backend=backend)

    @classmethod
    def from_rows(cls, rows: list[Row]) -> "IngestSource":
        """直接从 Row 列表构建（用于测试或手动输入）。"""
        return cls(rows=rows)

    # ── 生命周期 ───────────────────────────────────────────────────

    def start(self) -> "IngestSource":
        """启动后台数据采集（HTTP/MQTT）。"""
        if self._backend is not None:
            self._backend.start()
        return self

    def stop(self) -> None:
        """停止后台数据采集。"""
        if self._backend is not None:
            self._backend.stop()

    def ready(self) -> bool:
        """是否有足够数据可分析（至少3条）。"""
        if self._static_rows is not None:
            return len(self._static_rows) >= 3
        if self._backend is not None:
            return self._backend.ready()
        return False

    # ── 核心输出 ───────────────────────────────────────────────────

    def get_timeseries(self) -> list[Row]:
        """返回引擎标准时序列表，可直接传给 render.build_context()。"""
        if self._static_rows is not None:
            return list(self._static_rows)
        if self._backend is not None:
            return self._backend.get_timeseries()
        return []

    def status(self) -> dict:
        """返回接入状态摘要（用于调试和 Web 界面）。"""
        rows = self.get_timeseries()
        return {
            "source_type": (
                "csv" if self._static_rows is not None
                else type(self._backend).__name__ if self._backend else "none"
            ),
            "row_count": len(rows),
            "ready": self.ready(),
            "latest": rows[-1] if rows else None,
        }
