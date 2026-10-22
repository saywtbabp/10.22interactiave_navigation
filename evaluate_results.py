# -*- coding: utf-8 -*-
"""Audit and index all completed real task outputs, including failures."""
import runtime
import json
import html
import csv
import numpy as np
from pathlib import Path
from configuration import ROOT
from experiments.audit import audit
from experiments.library import save_json

def main():
    records=[]
    for p in sorted((ROOT/'results').rglob('summary.json')):
        folder=p.parent
        if not (folder/'flight.csv').exists():continue
        summary=json.loads(p.read_text(encoding='utf-8'));checks=audit(folder)
        records.append(dict(folder=folder.relative_to(ROOT/'results').as_posix(),summary=summary,audit=checks))
    save_json(ROOT/'results'/'all_runs.json',records)
    rows=[]
    for r in records:
        s=r['summary'];link=r['folder']+('/report.html' if (ROOT/'results'/r['folder']/'report.html').exists() else '/summary.json')
        name=r['folder'];status='成功' if s['success'] else '失败：'+s['failure']
        rows.append('<tr><td><a href="{}">{}</a></td><td>{}</td><td>{}</td><td>{}</td><td>{:.2f}</td><td>{}</td><td>{:.2f}</td></tr>'.format(
            html.escape(link),html.escape(name),s['variant'],s['baseline'],status,s['simulated_time_s'],1<<s['config']['sobol_power'],s['wall_contact_peak_N']))
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>INTERACT · 全部运行</title><style>body{background:#09121e;color:#e8f1f7;font:14px 'Microsoft YaHei',sans-serif;margin:40px}a{color:#58e3d1}table{width:100%;border-collapse:collapse}td,th{padding:14px;border-bottom:1px solid #26364a;text-align:left}p{color:#91a5bb}</style><h1>INTERACT · 全部真实运行记录</h1><p>包含开发失败与 32 点调试，不能将它们混作正式 1024 验证。具体最终验收范围见项目 VALIDATION.md；这些是有限场景结果，不是统计成功率。</p><table><tr><th>结果目录</th><th>场景</th><th>策略</th><th>结果</th><th>仿真秒</th><th>N</th><th>墙接触 N</th></tr>ROWS</table></html>'''.replace('ROWS',''.join(rows))
    (ROOT/'results'/'index.html').write_text(document,encoding='utf-8');print('Audited',len(records),'real task folders',flush=True)
    for cache in (ROOT/'results'/'cache').glob('*/worlds.npz'):
        with np.load(cache,allow_pickle=False) as a:
            if not a['done'].all():continue
            with cache.with_suffix('.csv').open('w',encoding='utf-8-sig',newline='') as f:
                out=csv.writer(f);out.writerow(['id','I_hinge_kg_m2','tau_f_Nm','J_door_s','J_bypass_s','y_i','elapsed_door_s','door_reason','bypass_reason','pred_F_peak_N','pred_delta_angle_rad'])
                for i,theta in enumerate(a['theta']):
                    out.writerow([i,*theta,a['J_door'][i],a['J_bypass'][i],int(a['J_door'][i]<a['J_bypass'][i]),
                                  a['elapsed_door'][i],a['door_reason'][i],a['bypass_reason'][i],*a['z_hat'][i]])
    return records

if __name__=='__main__':main()
