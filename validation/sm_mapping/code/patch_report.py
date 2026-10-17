# sm-blind: write section 8 (comparison + diagnosis) of R16_V1_SM_MAPPING_BLIND_CN.md from results/DIAGNOSIS.json;
# sections 1-7 (blind part) are left untouched. Also records code hashes in DIAGNOSIS.json and updates STATUS.json.
import os, sys, json, re, hashlib, datetime, subprocess
sys.stdout.reconfigure(encoding='utf-8')
MY = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\sm_mapping_blind'
RES = os.path.join(MY, 'results'); WORK = os.path.join(MY, 'work')
D = json.load(open(os.path.join(RES, 'DIAGNOSIS.json'), encoding='utf-8'))
B = json.load(open(os.path.join(RES, 'BLIND_RESULTS.json'), encoding='utf-8'))
c, it2, it1, it3, rec = D['comparison_central'], D['item2_scales'], D['item1_tadpole'], D['item3_qcd'], D['recommended']
md, pp, L2 = it2['mr_decomposition_dmc'], it2['smdr_L25_single_scale_producer_fits_172p6'], it2['smdr_L2_own_fits']
xr = it1['mr_FJ_x_route_scale_scan']
f = lambda x, n=3: ('%+.' + str(n) + 'f') % x
L = ['## 8. 与生产数字的对照与诊断（2026-10-15 揭盲后追加；第 1–7 节盲算内容未改）', '',
     '符号约定：本节 Δm_c = Δλ(μ_b)/|dλ/dM_t|（生产约定，+ 表示临界 m_t 上移）；第 4 节表中的"M_t 当量"是它的相反数。', '',
     '### 8.1 中心值对照（(172.60, 0.1180, 125.13)，μ = μ_b）', '',
     '| 量 | 盲核验 mr 匹配 | 盲核验 SMDR 匹配 + 自写 RGE | 生产 | mr − 生产 | SMDR − 生产 |', '|---|---|---|---|---|---|']
for k, nm in (('lambda', 'λ'), ('yt', 'y_t'), ('g2', 'g₂'), ('gY', 'g_Y'), ('g3', 'g₃')):
    v = c[k]
    L.append('| %s | %.6f | %.6f | %s | %+.1e | %+.1e |' % (nm, v['blind_mr'], v['blind_smdr_own_rge'], repr(v['production']), v['diff_mr'], v['diff_smdr']))
L += ['| dλ(μ_b)/dM_t [GeV⁻¹] | %.6f（d²λ/dM_t² 外推到 173.0：%.5f） | | −0.00656（D120 S = −0.006611 @173.0） | | |' % (
    D['comparison_dlam_dMt']['blind'], D['comparison_dlam_dMt']['blind'] + 0.4 * B['derivatives_at_mub']['d2lam_dMt2']),
      '| m̄_t(m̄_t) [GeV] | %.3f | | ≈162.82（D120：162.802 @ M_t=172.580，按斜率移到 172.60） | | |' % B['mtbar']['central_4l'], '',
      '结论：同为 SMDR 匹配时，盲算与生产在 μ_b 处 λ 差 %.0e（%.4f GeV），y_t 差 %.0e，完全一致；mr 中心比 SMDR 低 %.1e（%.3f GeV），这是两套二圈定阶组织方式之差（已计入方案项）。g_Y 的 %.1e 来自 W/Z 质量约定与输入集（W/Z 约定单项即可使 g_Y(M_t) 变 1.6×10⁻⁴），对 λ_b 影响 <2×10⁻⁵。' % (
          abs(c['lambda']['diff_smdr']), abs(c['lambda']['diff_smdr']) / 0.006592, abs(c['yt']['diff_smdr']), abs(c['lambda']['diff_mr']),
          abs(c['lambda']['diff_mr']) / 0.006592, c['gY']['diff_mr']), '',
      '### 8.2 匹配标度项相差约 10 倍的逐项定位', '',
      '**(0) 先复现两边。** 用生产者 `c148_fit.c` 的 PROVENANCE 副本（`work/drivers/c148_fit_copy.c`，只把 M_h、M_Z/M_W、G_F 的圈数改为可选参数）链接本目录独立编译的 SMDR：ξ=1 拟合与生产逐位相同（差 0）；ξ=½/2 的 Δλ_b = %+.4e/%+.4e（生产 %+.4e/%+.4e），即 Δm_c = %s/%s GeV，复现 run02。盲算 mr 的 ξ=½/2 为 %s/%s GeV，**与 BKPV Table II 的 δ_μ（+0.17/−0.36）数值一致**（BKPV 是其输入下 M_t^cri 的位移，本处是 PDG 2026 输入下 λ(μ_b) 的 M_t 当量，二者灵敏度相近）：mr 再现的是文献的 G_F 方案口径。' % (
          D['reproduction_of_producer_run02']['dlam_b_xi0p5']['own'], D['reproduction_of_producer_run02']['dlam_b_xi2']['own'],
          D['reproduction_of_producer_run02']['dlam_b_xi0p5']['producer'], D['reproduction_of_producer_run02']['dlam_b_xi2']['producer'],
          f(pp['xi0.5']), f(pp['xi2']), f(md['mu0_0.5']['full']), f(md['mu0_2']['full'])), '',
      '**(1) tadpole 方案。** mr 的耦合走 y 展开（λ = G_F M_H²/√2·(1+δ_H)、y_t = 2^{3/4}G_F^{1/2}M_t(1+δ_t) 等），FJ tadpole 在系数中恒等抵消：一圈 H 的 y10 = −18.31，而 x10 − dr10 = −18.32，其中 x10 = 193、dr10 = 212 各自很大。mr 没有"无 tadpole MS-bar VEV"的耦合选项，无法在 mr 内做与 SMDR 同方案的对照。mr 内能做的同族对照是 FJ"x 路线"（FJ 跑动质量除以 FJ v²，不重展开）：ξ=½/2 的 Δm_c = %s/%s GeV，比 y 路线更大；μ₀=M_t 处中心差 Δλ_b = %+.1e。结论：真正受 FJ tadpole 影响的组织方式确实放大标度依赖；但 BKPV/mr 的 −0.36/+0.17 来自 y 路线，与 tadpole 无关，根源是 G_F 方案下 λ 与 y_t 的二圈定阶组织。' % (
          f(xr['mu0_0.5']['dmc']), f(xr['mu0_2']['dmc']), xr['central_minus_yroute_dlam_b']), '',
      '**(2) 变的是哪些标度。** mr 用单一 μ₀ 同时作用于 λ、y_t、g、g′、α(μ₀) 及 α_s 的退耦点。把 α_s 改为在 M_t 退耦、再跑到 μ₀，Δm_c 只变 ≤0.004 GeV，所以不是来源。线性分解（ξ=½/2，GeV，各项之和与整体相差 <0.001）：',
      '- y_t：%s/%s，其中纯 QCD 塔 %s/%s，非 QCD %s/%s；' % (f(md['mu0_0.5']['yt']), f(md['mu0_2']['yt']), f(md['mu0_0.5']['yt_QCD_tower']), f(md['mu0_2']['yt_QCD_tower']),
                                                       f(md['mu0_0.5']['yt_nonQCD']), f(md['mu0_2']['yt_nonQCD'])),
      '- λ：%s/%s；' % (f(md['mu0_0.5']['lam']), f(md['mu0_2']['lam'])),
      '- g₃、g、g′：都 <0.004。',
      '',
      'SMDR（生产拟合，L2.5，单独改一个求值标度，M_t = 172.60）：',
      '- M_t：%s/%s；' % (f(pp['t0.5']), f(pp['t2'])),
      '- M_h：%s/%s；' % (f(pp['h0.5']), f(pp['h2'])),
      '- G_F、M_Z/M_W：<0.001；退耦：%s/%s。' % (f(pp['d0.5']), f(pp['d2'])),
      '',
      '两大项在标度联动变化时相互抵消，合计 %s/%s。本目录再跑 L2（M_h、M_Z/M_W、G_F 去掉领头三圈，只到二圈）：' % (f(pp['xi0.5']), f(pp['xi2'])),
      '- M_t：%s/%s；' % (f(L2['L2_t0.5']), f(L2['L2_t2'])),
      '- M_h：%s/%s；' % (f(L2['L2_h0.5']), f(L2['L2_h2'])),
      '- 联动合计只有 %s/%s，反而比 L2.5 更小。' % (f(L2['L2_xi0.5']), f(L2['L2_xi2'])),
      '',
      '领头三圈项本身只把 m_c 移 %.3f GeV，并把 M_h 项的标度依赖减半；SMDR 整体标度依赖小，主要是抵消，而非收敛。' % abs(it2['smdr_L25_minus_L2_at_xi1_dmc']),
      '',
      '**结论**：约 10 倍的差距大致是两个因子相乘：',
      '- G_F 方案中 λ 与 y_t 的非 QCD 对数项更大且同号，约 ×3–4；',
      '- SMDR 联动变化时 M_t、M_h 两项相消，约 ×2–3。',
      '',
      'SMDR 各标度独立变化时的包络为 [%s, %s] GeV（各项半宽平方和 %.3f）。' % (f(it2['smdr_L25_independent_envelope'][0]), f(it2['smdr_L25_independent_envelope'][1]), it2['smdr_L25_quadrature_halfranges']), '',
      '**(3) y_t 的 QCD 截断。** mr 的 y01–y04 在 μ₀ ≠ M_t 处与 n_f=6 四圈 RG 一致到 10⁻¹⁵，对数项本身没有问题。纯 QCD 塔的标度依赖 %s/%s，与生产 RG 重构的 +0.064/−0.057 及 SMDR 的 M_t 项 %s/%s 同量级。它们都是 O(α_s⁵) 截断（重整子增长）的表现，不是两边差异的来源。注意：这与"极点级数五圈项 0.075"是同一阶效应的两种估计，不应再平方相加。' % (
          f(it3['qcd_tower_dmc']['xi0p5']), f(it3['qcd_tower_dmc']['xi2']), f(it3['smdr_Mt_piece']['xi0p5']), f(it3['smdr_Mt_piece']['xi2'])), '',
      '**(4) 重整子（读到原文，arXiv:1605.03609，`inputs/20261015_beneke2016/`）。**',
      '- 摘要：计入内部 b、c 质量效应后，该不确定度"约 110 MeV"；五圈及以上对质量关系的额外贡献"约 300 MeV"。',
      '- Eq.(4.7)（p.11）：δ⁽⁵⁺⁾m_P = 0.272 +0.016/−0.041(N) ±0.001(c₄) ±0.011(α_s) ±0.066(歧义) GeV。',
      '- p.13：把极点质量的最终不确定度定为约 70 MeV 的歧义（轻夸克无质量）。',
      '',
      '盲算假设的 0.11 GeV 就是摘要中含 b、c 的值，与 D120 的引用一致。0.272 GeV 是单侧余项（全阶 pole 质量高于四圈截断值），不是对称误差。', '',
      '### 8.3 我认为最可辩护的 SM 侧理论误差（对临界 m_t，GeV）', '',
      '| 项 | m̄_t(m̄_t) 表述 | pole 表述（四圈截断约定） | 理由 |', '|---|---|---|---|',
      '| 匹配标度 | [%s, %s] | 用同一值 | SMDR 各求值标度在 [½,2] 内独立变化，M_t 项扣除纯 QCD 塔；不依赖 M_t/M_h 的偶然相消 |' % (f(rec['mbar']['matching_scale_envelope'][0]), f(rec['mbar']['matching_scale_envelope'][1])),
      '| 组织方式（mr 对 SMDR，μ₀=M_t，同一 M_W） | %.3f | %.3f | 两套完整二圈定阶结果之差，直接计算；包含 SMDR 领头三圈项（%.3f） |' % (rec['mbar']['scheme_spread_mr_vs_smdr'], rec['mbar']['scheme_spread_mr_vs_smdr'], rec['mbar']['leading3loop_size_info']),
      '| RGE 截断 / α_s 阶 / W-Z 极点约定 | %.3f / %.3f / %.3f | 同左 | 第 4 节 |' % (rec['mbar']['rge'], rec['mbar']['alpha_s_order'], rec['mbar']['WZ_pole_convention']),
      '| 极点级数 O(α_s⁵) | — | %.3f | 与纯 QCD 塔标度变化（≈0.06）及 Beneke 歧义（0.066–0.071）是同一阶，只计一次 |' % rec['pole_4loop_truncated']['pole_series_O_as5'],
      '| **合成（平方和）** | **±%.2f** | **±%.2f** | 若把纯 QCD 塔留在标度包络内、不另计极点项：±%.2f |' % (rec['mbar']['total'], rec['pole_4loop_truncated']['total'], rec['pole_4loop_truncated']['alt_total_with_QCD_tower_in_envelope']),
      '', '与生产 run02（±0.047 / ±0.086）的差别：',
      '- 我不用"所有标度联动"，因为它的小值依赖 M_t 与 M_h 两项的相消；',
      '- 方案项用直接计算的 mr − SMDR = 0.042，生产用 BKPV 表推得 0.022。',
      '', 'm̄_t 表述达到 ≤±0.1；pole 表述为 ±%.3f，处在边界上（严格说略超 0.1；若不另计极点项为 ±%.3f）。' % (
          rec['pole_4loop_truncated']['total'], rec['pole_4loop_truncated']['alt_total_with_QCD_tower_in_envelope']), '',
      '### 8.4 PRL 表述建议（两种口径都如实写）',
      '1. 以 m̄_t(m̄_t) 表述为主：SM 侧理论误差 ±0.07 GeV。方法写明：无 tadpole MS-bar 的 NNLO⁺ 匹配（SMDR），三圈 + 已知四/五圈 RGE；各求值标度在 [½,2] 内独立变化，再加两套二圈组织方式之差。',
      '2. pole 质量表述：±0.10 GeV，并明确是"四圈截断的 pole–MS-bar 关系"。同时写出：全阶求和的 pole 质量约高 0.27 GeV（Beneke 等 Eq.(4.7)），并带 0.07–0.11 GeV 的不可约重整子歧义；与直接测量（MC 质量）比较时，另有约 0.5 GeV 的解释误差（D120 已引 Hoang 2020）。',
      '3. 另一口径要点名，不隐去：G_F 方案的二圈定阶匹配（BKPV/mr）在 μ₀ ∈ [M_t/2, 2M_t] 的变化为 −0.36/+0.17 GeV，本工作已独立复现。SMDR 全标度联动的 +0.033/−0.019 依赖两项相消，也不宜单独作为唯一误差。建议正文给 ±0.07（m̄_t）/ ±0.10（pole），脚注给出 BKPV 口径作为保守参照，并说明差异来源（8.2）。',
      '4. 不要把"pole 级数五圈项"与"y_t 纯 QCD 的标度变化"叠加，二者是同一阶。参数性误差（M_h、α_s）单列，不并入理论误差。', '',
      '### 8.5 已证明 / 条件成立 / 未证明（诊断部分）',
      '- **已证明**：',
      '  - 生产 run02 的 SMDR 标度变化数值被本目录独立编译逐位复现；',
      '  - 盲算 mr 的标度变化与 BKPV 的 δ_μ 数值一致；',
      '  - 两者相差的来源已逐项分解：G_F 方案中 λ 与 y_t 的非 QCD 对数项大且同号，SMDR 联动变化中 M_t、M_h 两项相消；α_s 退耦标度、纯 QCD 截断、tadpole（y 路线）都不是来源；',
      '  - Beneke 等的数值已与原文核对。',
      '- **条件成立**：推荐误差依赖两个约定——"各求值标度独立变化"，以及"方案项取 mr − SMDR"。M_t 项扣除纯 QCD 塔是近似，因为两种展开方向在 O(a⁵) 上不同。',
      '- **未证明**：',
      '  - 真实的三圈电弱/混合阈值大小；',
      '  - G_F 方案与无 tadpole 方案哪一个更接近全阶结果。',
      '', '诊断文件：`results/DIAGNOSIS.json`（汇总）、`results/diag_smdr_scale.json`、`results/diag_mr_decomp.json`、`results/diag_smdr/`；代码 `work/diag_smdr_scale.py`、`work/diag_mr_decomp.py`、`work/diag_final.py`、`work/patch_report.py`、`work/drivers/c148_fit_copy.c`。', '']
rp = os.path.join(MY, 'R16_V1_SM_MAPPING_BLIND_CN.md')
txt = open(rp, encoding='utf-8').read()
i = txt.index('## 8. '); j = txt.index('## 9. ')
txt = txt[:i] + '\n'.join(L) + '\n' + txt[j:]
open(rp, 'w', encoding='utf-8').write(txt)
for fn in ('diag_smdr_scale.py', 'diag_mr_decomp.py', 'diag_final.py', 'patch_report.py', 'drivers/c148_fit_copy.c'):
    D.setdefault('code_sha256', {})['work/' + fn] = hashlib.sha256(open(os.path.join(WORK, fn), 'rb').read()).hexdigest()
D['report_section'] = 'R16_V1_SM_MAPPING_BLIND_CN.md section 8'
json.dump(D, open(os.path.join(RES, 'DIAGNOSIS.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
st = json.load(open(os.path.join(MY, 'STATUS.json'), encoding='utf-8'))
for x in ['diagnosis: reproduced producer run02 SMDR scale variation bit-exactly with own build (c148_fit_copy)',
          'diagnosis: mr scale variation numerically = BKPV delta_mu; decomposed (y_t nonQCD + lambda same-sign in G_F scheme; SMDR M_t/M_h cancellation)',
          'diagnosis: Beneke 2016 read (110 MeV incl b,c; 70 MeV; delta5+ = 0.272 +- 0.066)',
          'report section 8 written; results/DIAGNOSIS.json']:
    if x not in st['done']:
        st['done'].append(x)
for x in ['results/DIAGNOSIS.json', 'results/diag_smdr_scale.json', 'results/diag_mr_decomp.json', 'work/drivers/c148_fit_copy.c']:
    if x not in st['files']:
        st['files'].append(x)
st['key_numbers'].update({'diag_recommended_mbar_GeV': round(D['recommended']['mbar']['total'], 3),
                          'diag_recommended_pole4l_GeV': round(D['recommended']['pole_4loop_truncated']['total'], 3),
                          'diag_smdr_independent_envelope': [round(x, 3) for x in D['item2_scales']['smdr_L25_independent_envelope']],
                          'diag_lam_mub_smdr_blind_minus_prod': c['lambda']['diff_smdr']})
st['next_command'] = 'none (diagnosis complete); await coordinator'
st['remaining'] = []
st['complete'] = True
st['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
json.dump(st, open(os.path.join(MY, 'STATUS.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('section 8 written (%d lines); STATUS updated' % len(L))
