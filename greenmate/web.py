"""Web 界面：零依赖标准库 HTTP 服务，浏览器查看五大场景与四大视图。

用法：
    python main.py web              # 启动并自动打开浏览器
    python main.py web --no-browser # 启动但不打开浏览器
"""

import html
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from greenmate import scenarios
from greenmate.render import build_context, build_judgment_doc

HOST = "127.0.0.1"
PORT = 8765

# 关注等级图标 → 页面主色类
_LEVEL_CLASS = {"🟢": "ok", "🟡": "warn", "🟠": "watch", "🔴": "alert"}
# 证据状态 → 徽章文案与样式类
_STATUS_BADGE = {
    "ok": ("可用", "st-ok"),
    "partial": ("部分", "st-part"),
    "unavailable": ("不可用", "st-un"),
}

CSS = """
:root{
  --green:#2e7d32;--green-dark:#1b5e20;--green-soft:#e8f2e8;
  --ink:#22301f;--muted:#6b7a66;--bg:#f2f6f1;--line:#dfe7dc;
}
*{box-sizing:border-box}
body{margin:0;font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;
  background:var(--bg);color:var(--ink);line-height:1.6}
.topbar{background:linear-gradient(135deg,var(--green-dark),var(--green));
  color:#fff;padding:14px 22px}
.brand{font-size:20px;font-weight:700;letter-spacing:.5px}
.meta{font-size:12.5px;opacity:.92;margin-top:2px}
.tabs{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
.tab{padding:6px 14px;border-radius:999px;background:rgba(255,255,255,.14);
  color:#fff;text-decoration:none;font-size:13.5px;
  border:1px solid rgba(255,255,255,.35)}
.tab.active{background:#fff;color:var(--green-dark);font-weight:700}
main{max-width:1080px;margin:0 auto;padding:20px 22px 10px}
.hero{display:flex;gap:30px;flex-wrap:wrap;border-radius:10px;padding:16px 20px;
  box-shadow:0 1px 4px rgba(0,0,0,.06);margin-bottom:18px;background:#fff;
  border-left:6px solid var(--green)}
.hero.ok{border-color:#16a34a;background:#f0faf1}
.hero.warn{border-color:#d97706;background:#fdf7ec}
.hero.watch{border-color:#ea580c;background:#fdf3ec}
.hero.alert{border-color:#dc2626;background:#fdf0f0}
.hero .label{display:block;font-size:12px;color:var(--muted)}
.hero strong{font-size:17px}
section{background:#fff;border-radius:10px;padding:18px 20px;
  box-shadow:0 1px 4px rgba(0,0,0,.05);margin-bottom:18px}
h2{margin:0 0 12px;font-size:16.5px;color:var(--green-dark)}
h2::before{content:"▌";color:var(--green);margin-right:6px}
table{width:100%;border-collapse:collapse;font-size:14px}
th{background:var(--green-soft);color:var(--green-dark);text-align:left;
  padding:8px 10px;border-bottom:2px solid var(--line)}
td{padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}
tr.cur td{background:#fffbe6;font-weight:700}
.chips{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
.chip{background:var(--green-soft);border-radius:8px;padding:8px 12px;
  font-size:13px;max-width:100%}
.chip b{color:var(--green-dark)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:12px}
.card{border:1px solid var(--line);border-radius:10px;padding:12px 14px;
  background:#fbfdfb}
.card .p{font-size:13px;color:var(--muted)}
.card .v{font-size:21px;font-weight:700;margin:2px 0}
.card .t{font-size:13px;color:var(--green)}
.card .s{margin-top:6px;font-size:13px;font-weight:600}
.aux{margin-top:12px;font-size:13.5px;color:var(--muted)}
.aux b{color:var(--ink)}
.chart{width:100%;height:auto;display:block}
.legend{font-size:13px;color:var(--muted);margin-bottom:4px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;
  margin:0 5px -1px 14px}
.sw-t{background:#d9534f}.sw-h{background:#3b82c4}
.grid-line{stroke:#e5ece3;stroke-width:1}
.line-t{fill:none;stroke:#d9534f;stroke-width:2.5}
.line-h{fill:none;stroke:#3b82c4;stroke-width:2.5}
.dot-t{fill:#d9534f}.dot-h{fill:#3b82c4}
svg text{font-size:12px;fill:#55604f}
.lab-t{fill:#b23431;font-weight:600}
.lab-h{fill:#2b5f92}
.lab-x{fill:#8a967f}
.limit{font-family:Consolas,monospace;color:#b45309;font-size:13px;
  margin-top:8px}
.nodes{margin:10px 0 0;padding-left:20px}
.nodes li{margin:4px 0;font-size:14px}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;
  font-weight:600}
.st-ok{background:#e5f6e8;color:#15803d}
.st-part{background:#fdf3e0;color:#b45309}
.st-un{background:#fde8e8;color:#b91c1c}
.vd-item{border:1px solid var(--line);border-radius:8px;padding:10px 14px;
  margin:8px 0;background:#fbfdfb;font-size:13.5px}
.vd-item b{color:var(--green-dark)}
.goal{color:var(--muted);margin-top:2px}
blockquote{margin:10px 0 0;padding:10px 14px;background:var(--green-soft);
  border-left:4px solid var(--green);border-radius:6px;font-size:14px}
details{margin-top:4px}
summary{cursor:pointer;color:var(--green-dark);font-weight:600;font-size:14px}
pre{background:#f6f8f5;border:1px solid var(--line);border-radius:8px;
  padding:12px;overflow:auto;font-size:12.5px;line-height:1.5}
footer{text-align:center;color:var(--muted);font-size:12.5px;padding:8px 0 26px}
"""


def _esc(v):
    return html.escape(str(v))


def _chart_svg(series):
    """温度/湿度双序列内联 SVG 折线图（各自独立纵轴归一化）。"""
    times = [r[0] for r in series]
    temps = [float(r[1]) for r in series]
    hums = [float(r[2]) for r in series]
    w, h, pad = 760, 280, 46
    n = len(times)
    span = w - 2 * pad

    def xs(i):
        return pad + (span * i / (n - 1) if n > 1 else span / 2)

    def make_y(vals):
        lo, hi = min(vals), max(vals)
        if hi == lo:
            hi = lo + 1.0
        margin = (hi - lo) * 0.15
        lo, hi = lo - margin, hi + margin
        inner = h - 2 * pad
        return lambda v: pad + (1 - (v - lo) / (hi - lo)) * inner

    yt, yh = make_y(temps), make_y(hums)

    def poly(vals, y):
        return " ".join(f"{xs(i):.1f},{y(v):.1f}" for i, v in enumerate(vals))

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
             f'aria-label="温度与湿度演化图">']
    for gy in range(5):
        y = pad + (h - 2 * pad) * gy / 4
        parts.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{w - pad}" '
                     f'y2="{y:.1f}" class="grid-line"/>')
    parts.append(f'<polyline points="{poly(temps, yt)}" class="line-t"/>')
    parts.append(f'<polyline points="{poly(hums, yh)}" class="line-h"/>')
    for i, v in enumerate(temps):
        parts.append(f'<circle cx="{xs(i):.1f}" cy="{yt(v):.1f}" r="3.5" '
                     f'class="dot-t"/>')
        parts.append(f'<text x="{xs(i):.1f}" y="{yt(v) - 9:.1f}" '
                     f'text-anchor="middle" class="lab-t">{v:.1f}°</text>')
    for i, v in enumerate(hums):
        parts.append(f'<circle cx="{xs(i):.1f}" cy="{yh(v):.1f}" r="3.5" '
                     f'class="dot-h"/>')
        parts.append(f'<text x="{xs(i):.1f}" y="{yh(v) + 16:.1f}" '
                     f'text-anchor="middle" class="lab-h">{v:.0f}%</text>')
    for i, tm in enumerate(times):
        parts.append(f'<text x="{xs(i):.1f}" y="{h - 12}" '
                     f'text-anchor="middle" class="lab-x">{_esc(tm)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _sec_input(sc, ctx):
    rows = []
    last = len(sc["timeseries"]) - 1
    for i, (tm, tp, hm, lt, c2) in enumerate(sc["timeseries"]):
        cur = ' class="cur"' if i == last else ""
        tag = "（当前）" if i == last else ""
        rows.append(f"<tr{cur}><td>{tm}{tag}</td><td>{tp}</td>"
                    f"<td>{hm}</td><td>{lt}</td><td>{c2}</td></tr>")
    chips = (
        f'<div class="chip"><b>外部天气：</b>{_esc(sc["weather_text"])}</div>'
        f'<div class="chip"><b>RAG：</b>{_esc(sc["rag"]["source"])} — '
        f'{_esc(sc["rag_summary"])}</div>'
        f'<div class="chip"><b>模型：</b>{_esc(sc["model_output_line"])}</div>'
    )
    return (
        "<section><h2>输入数据与外部证据</h2>"
        "<table><tr><th>时刻</th><th>温度(°C)</th><th>湿度(%RH)</th>"
        "<th>光照(lux)</th><th>CO₂(ppm)</th></tr>"
        + "".join(rows) + "</table>"
        f'<div class="chips">{chips}</div></section>'
    )


def _sec_judgment(sc, ctx):
    doc = build_judgment_doc(sc, ctx)
    body = json.dumps(doc, ensure_ascii=False, indent=2)
    return (
        "<section><h2>结构化判断（JSON）</h2>"
        "<details><summary>展开 / 收起</summary>"
        f"<pre>{_esc(body)}</pre></details></section>"
    )


def _sec_view_a(sc, ctx):
    va = sc["view_a"]
    cards = []
    for name, value_tpl, trend_tpl, status in va["rows"]:
        cards.append(
            f'<div class="card"><div class="p">{_esc(name)}</div>'
            f'<div class="v">{_esc(value_tpl.format(**ctx))}</div>'
            f'<div class="t">{_esc(trend_tpl.format(**ctx))}</div>'
            f'<div class="s">{_esc(status)}</div></div>'
        )
    aux = (
        f'<div class="aux"><b>短时预测：</b>{_esc(va["short_forecast"])}　'
        f'<b>明日风险：</b>{_esc(va["risk"])}　'
        f'<b>证据完整度：</b>{_esc(va["integrity"])}</div>'
    )
    return (
        '<section><h2>视图A · 状态总览【现在怎样？】</h2>'
        f'<div class="cards">{"".join(cards)}</div>{aux}</section>'
    )


def _sec_view_b(sc, ctx):
    vb = sc["view_b"]
    nodes = "".join(f"<li>{_esc(n)}</li>" for n in vb["nodes"])
    intro = vb.get("intro", "")
    intro_html = f"<p>{_esc(intro)}</p>" if intro else ""
    return (
        '<section><h2>视图B · 状态演化【为什么变成这样？】</h2>'
        f'{intro_html}'
        '<div class="legend">温度(°C)<span class="sw sw-t"></span>'
        '湿度(%RH)<span class="sw sw-h"></span></div>'
        + _chart_svg(sc["timeseries"])
        + f'<div class="limit">{_esc(vb["limit_line"])}</div>'
        f'<ul class="nodes">{nodes}</ul>'
        f'<p><b>漂移驱动分析：</b>{_esc(vb["driver"])}</p>'
        f'<p><b>{_esc(vb["significance_title"])}：</b>'
        f'{_esc(vb["significance"])}</p></section>'
    )


def _sec_view_c(sc, ctx):
    vc = sc["view_c"]
    support = "".join(
        f"<tr><td>{_esc(label)}</td><td>{_esc(content)}</td>"
        f"<td>{_esc(strength)}</td></tr>"
        for label, content, strength in vc["support"]
    )
    mitigate = "".join(
        f"<tr><td>{_esc(label)}</td><td>{_esc(content)}</td></tr>"
        for label, content in vc["mitigate"]
    )
    gaps = "".join(f"<li>{_esc(g)}</li>" for g in vc["gaps"])
    return (
        '<section><h2>视图C · 证据空间【凭什么这样判断？】</h2>'
        f'<p><b>▌支持“{_esc(vc["support_title"])}”的证据</b></p>'
        "<table><tr><th>证据</th><th>内容</th><th>支持力度</th></tr>"
        + support + "</table>"
        '<p><b>▌反向提示证据（缓和因素）</b></p>'
        "<table><tr><th>证据</th><th>内容</th></tr>"
        + mitigate + "</table>"
        '<p><b>▌证据不足项</b></p>'
        f'<ul class="nodes">{gaps}</ul>'
        f'<p><b>▌无分歧项：</b>{_esc(vc["no_dispute"])}</p></section>'
    )


def _sec_view_d(sc, ctx):
    vd = sc["view_d"]
    conds = "".join(f"<li>{_esc(c)}</li>" for c in vd["conditions"])
    items = []
    for item in vd["verifications"]:
        title, inp, expect = item[0], item[1], item[2]
        goal = item[3] if len(item) > 3 else ""
        goal_html = f'<div class="goal">→ {_esc(goal)}</div>' if goal else ""
        items.append(
            f'<div class="vd-item"><b>{_esc(title)}</b>'
            f'<div>{_esc(inp)}</div><div>{_esc(expect)}</div>{goal_html}</div>'
        )
    return (
        '<section><h2>视图D · 场景回放与对比【判断是否可验证？】</h2>'
        f'<p><b>本次输入条件（{_esc(sc["run_id"])}）：</b></p>'
        f'<ul class="nodes">{conds}</ul>'
        f'<p><b>复现验证方案：</b></p>{"".join(items)}</section>'
    )


def render_page(sc):
    ctx = build_context(sc["timeseries"])
    tabs = []
    for sid in scenarios.all_scenario_ids():
        s = scenarios.get_scenario(sid)
        active = " active" if sid == sc["id"] else ""
        tabs.append(f'<a class="tab{active}" href="/?s={sid}">'
                    f'{sid} · {_esc(s["name"])}</a>')
    cls = _LEVEL_CLASS.get(sc["attention_icon"], "ok")
    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GreenMate · {sc['id']} {_esc(sc['name'])}</title>
<style>{CSS}</style>
</head>
<body>
<header class="topbar">
  <div class="brand">GreenMate 连续状态认知工作台</div>
  <div class="meta">run_id: {_esc(sc['run_id'])} · {_esc(sc['location'])} ·
    {_esc(sc['analysis_time'])} · {_esc(sc['engine_version'])}</div>
  <nav class="tabs">{''.join(tabs)}</nav>
</header>
<main>
<section class="hero {cls}">
  <div><span class="label">状态判断</span><strong>{_esc(sc['current_state'])}</strong></div>
  <div><span class="label">关注等级</span><strong>{sc['attention_icon']} {_esc(sc['attention_level'])}</strong></div>
  <div><span class="label">证据完整度</span><strong>{_esc(sc['evidence_integrity'])}</strong></div>
</section>
{_sec_input(sc, ctx)}
{_sec_judgment(sc, ctx)}
{_sec_view_a(sc, ctx)}
{_sec_view_b(sc, ctx)}
{_sec_view_c(sc, ctx)}
{_sec_view_d(sc, ctx)}
<section><h2>自然语言综合解释</h2>
  <p>基于 <code>{_esc(sc['run_id'])}</code> 结构化判断：</p>
  <p>{_esc(sc['narrative_body'])}</p>
  <blockquote>{_esc(sc['narrative_closing'])}</blockquote>
</section>
</main>
<footer>GreenMate-Engine v1.0 · 零依赖标准库渲染 · 演示数据，判断输出仅供参考</footer>
</body>
</html>
"""
    return page


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            qs = parse_qs(parsed.query)
            sid = (qs.get("s", ["S2"])[0] or "S2").upper()
            sc = scenarios.get_scenario(sid)
            if sc is None:
                sc = scenarios.get_scenario("S2")
            self._send(200, render_page(sc), "text/html; charset=utf-8")
        else:
            self._send(404, '<h1>404</h1><p><a href="/">返回工作台</a></p>',
                       "text/html; charset=utf-8")


def serve(open_browser=True):
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}/"
    print(f"GreenMate Web 界面已启动：{url}")
    print(f"场景切换：{'、'.join(scenarios.all_scenario_ids())}（页签或 /?s=S3）")
    print("按 Ctrl+C 停止服务")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
        print("\n服务已停止")


if __name__ == "__main__":
    serve(open_browser="--no-browser" not in sys.argv)
