"""Local evidence report; explicitly an offline view of recorded real runs."""
import html
import json
from pathlib import Path
from .library import safe

def generate(mission):
    folder=mission.folder;s=mission.summary
    frames=sorted(p.name for p in folder.glob('*_dashboard.png'))
    decisions=''.join('<tr><td>{}</td><td>{:.2f}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
        html.escape(x['opening']),x['G'], '—' if x['p_D'] is None else '{:.4f}'.format(x['p_D']),
        '—' if x['D'] is None else '{:.4f}'.format(x['D']),html.escape(x['action']),html.escape(x['reason'])) for x in mission.decisions)
    status='到达目标' if s['success'] else '未完成：'+s['failure']
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>INTERACT · 真实运行记录</title>
<style>body{margin:0;background:#09121e;color:#e8f1f7;font:15px 'Microsoft YaHei',sans-serif}main{max-width:1280px;margin:36px auto;padding:0 24px}h1{font-size:32px;letter-spacing:2px}h2{font-size:20px;color:#58e3d1}p{color:#91a5bb;line-height:1.8}.stats{display:flex;gap:16px;flex-wrap:wrap}.stat{padding:22px;background:#111f30;border:1px solid #26364a;border-radius:12px;min-width:175px}.stat b{display:block;font-size:28px;margin-top:10px}img{width:100%;border:1px solid #26364a;border-radius:12px}button,select,a{color:#58e3d1}button,select{background:#172a3d;border:1px solid #26364a;padding:10px;border-radius:6px;margin:12px 6px 12px 0}table{width:100%;border-collapse:collapse;background:#111f30}td,th{text-align:left;padding:16px;border-bottom:1px solid #26364a}th{color:#91a5bb}code{background:#172a3d;padding:4px;border-radius:4px}</style><main>
<h1>↗ INTERACT <span style="font-size:16px;color:#91a5bb">自主交互导航实验记录</span></h1>
<p>这是已记录物理运行的离线证据查看器。实时三维、在线地图和机载视觉请运行 main.py。图像来自同一 MuJoCo model/data 的真实运行。</p>
<div class="stats"><div class="stat">任务结果<b>STATUS</b></div><div class="stat">仿真总耗时 / 含交互恢复<b>TIME s</b></div><div class="stat">候选数量<b>COUNT</b></div><div class="stat">Probe / 重规划<b>PROBES / REPLANS</b></div></div>
<h2>同步三视图关键帧</h2><button onclick="step(-1)">← 上一帧</button><select id="pick" onchange="show(+this.value)"></select><button onclick="step(1)">下一帧 →</button><img id="frame" alt="已记录的同步三视图">
<h2>门决策</h2><table><tr><th>开口</th><th>G / m</th><th>p_D</th><th>D</th><th>动作</th><th>原因</th></tr>ROWS</table>
<h2>数据与适用范围</h2><p>已训练 YOLOv8n / ONNX，适用范围为合成 MuJoCo 场景；检测分数不等于材质或可推动概率。累积测距生成歧义开口，低速验证使用 IMU / 姿态 / 电机输入估计外力，区分受阻、可通行与未完成。理想定位与自身柔性多体状态仪器；门 Probe 保留全部外接触法向力仪器；门角由测距拟合。低 G 短路分支的 p_D/D 保留为空。</p>
<p><a href="summary.json">结果摘要</a> · <a href="decisions.json">完整决策与缓存来源</a> · <a href="flight.csv">飞行与控制记录</a> · <a href="sensors.jsonl">RGB / 测距 / 位姿时序</a> · <a href="online_map.json">最终认知地图</a> · <a href="events.json">地图事件</a> · <a href="weight_history.json">候选权重历史</a></p>
<script>const frames=FRAMES;let index=0;const select=document.getElementById('pick');frames.forEach((f,i)=>{const o=document.createElement('option');o.value=i;o.textContent=f.replace('_dashboard.png','');select.append(o)});function show(i){if(!frames.length)return;index=Math.max(0,Math.min(frames.length-1,i));document.getElementById('frame').src=frames[index];select.value=index}function step(n){show(index+n)}show(frames.length-1);</script></main></html>'''
    for key,value in [('STATUS',html.escape(status)),('TIME','{:.2f}'.format(s['simulated_time_s'])),('COUNT',str(1<<mission.cfg.sobol_power)),
                       ('PROBES',str(s['probes'])),('REPLANS',str(s['replans'])),('ROWS',decisions),('FRAMES',json.dumps(frames))]:document=document.replace(key,value)
    (folder/'report.html').write_text(document,encoding='utf-8')
