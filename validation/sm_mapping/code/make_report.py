# sm-blind R16 V1: generate R16_V1_SM_MAPPING_BLIND_CN.md from results/BLIND_RESULTS.json (numbers never typed by hand)
import os, sys, json, math, datetime, hashlib
sys.stdout.reconfigure(encoding='utf-8')
MY = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\sm_mapping_blind'
RES = os.path.join(MY, 'results')
B = json.load(open(os.path.join(RES, 'BLIND_RESULTS.json'), encoding='utf-8'))
runs = json.load(open(os.path.join(RES, 'prod_runs.json')))
V = json.load(open(os.path.join(RES, 'validation_B1_B2_B3.json')))
VT = json.load(open(os.path.join(RES, 'validation_T1_T2_B1.json')))
TC = json.load(open(os.path.join(RES, 'rge_termcheck_vs_Buttazzo.json')))
for fn in ('rg_check_y0i.py', 'make_report.py'):
    B['code_sha256']['work/' + fn] = hashlib.sha256(open(os.path.join(MY, 'work', fn), 'rb').read()).hexdigest()
json.dump(B, open(os.path.join(RES, 'BLIND_RESULTS.json'), 'w', encoding='utf-8'), indent=1)
C, M, D = B['central_at_mub'], B['central_at_Mt'], B['derivatives_at_mub']
E, Y = B['errors']['lam'], B['errors']['yt']
S, SM = B['error_budget_lambda_mub_standard'], B['error_budget_lambda_mub_standard_Mt_equiv_GeV']
CS = B['error_budget_lambda_mub_conservative']
A = B['mu0_asymmetric']; T = B['tadpole_convention_test']
dl = abs(D['dlam_dMt'])
f6 = lambda x: '%.6f' % x
e = lambda x: '%+.2e' % x
L = []
w = L.append
w('# R16 V1：SM 映射（Planck 附近标度 MS-bar 耦合）盲独立核验报告')
w('')
w('代理：sm-blind；生成时间（UTC）：%s；结果文件：`results/BLIND_RESULTS.json`（时间戳 %s，含代码与输入 SHA-256）。' % (
    datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'), B['timestamp_utc']))
w('')
w('**盲规则执行**：未读取 `04_calculations/`、`07_reviews/`（除本目录）、`00_HANDOFF/`（除通用规则）及生产者下载目录；未对其做 Grep/Glob。全部代码、下载与结果都在本目录。')
w('')
w('## 1. 约定与输入')
w('- λ 归一：V = λ(|H|² − v²/2)²（m_h² = 2λv²）；g_Y = g′（非 GUT），g₁ = √(5/3) g_Y；MS-bar，μ ≥ μ₀ 为含 top 的完整 SM（6 味）。')
w('- μ_b = M̄_P/722 = %.4e GeV（M̄_P = 2.435×10¹⁸ GeV）；节点 μ(u) = μ_b/√cosh(2u)，u = 0, 0.5, …, 5。' % B['conventions']['mu_b_GeV'])
I = B['inputs']
w('- PDG 2026（`inputs/20261014_pdg/`，2026-10-14 下载）：M_t = %.2f（按 pole 解释），α_s(M_Z) = %.4f，M_h = %.2f，M_W = %.4f（Γ_W = %.2f），M_Z = %.4f（Γ_Z = %.4f），G_F = %.7e GeV⁻²，m_b(m_b) = %.3f。' % (
    I['Mt'], I['asMZ'], I['Mh'], I['MW_PDG'], I['GW'], I['MZ_PDG'], I['GZ'], I['GF'], I['mbmb']))
w('- mr 输入的 W/Z pole 质量（复极点 s = M̄² − iM̄Γ̄ 约定）：M̄_W = %.4f，M̄_Z = %.4f GeV（把 PDG 值直接当 pole 的做法作为方案变体）。' % (I['MW_pole'], I['MZ_pole']))
w('')
w('## 2. 方法（独立实现）')
w('- **匹配 A（主）**：mr（Kniehl–Pikelner–Veretin，arXiv:1601.08143，GitHub master；release 1.3.2 仅取其捆绑的 TSIL-1.3 与 Boost 子集）用 MinGW-w64 g++ 5.3 编译；完整二圈 EW、O(αα_s)、O(α²)，y_t 的 QCD 到四圈；α(μ) 由 G_F 定（AlphaGF），备选 AlphaSolve。耦合在 Python 中按 P2MS 同一公式由各阶系数重组（与 P2MS 输出逐位一致）。')
w('- **匹配 B（对照）**：SMDR 1.3（Martin–Robertson）`SMDR_Fit_Inputs`，输入集 {G_F, α, M_Z}（M_W 为预言）。')
w('- **α_s**：自写，5 味 4 圈跑动 + 在 μ₀ 处用 on-shell M_t 的 3 圈退耦（a²、a³ 常数已与 CKS 1997 解析值核对）+ 6 味跑动；变体 5 圈/4 圈。')
w('- **RGE**：自写积分器（DOP853，rtol 1e-11）。三圈 β 函数由 mr 的 `math/rge.m`（BPV 2012–13）自动转换；四圈：g₃ 全 SM、g₂/g_Y（1912.07624）、y_t 的 O(g₃⁸)、λ 的 O(y_t⁴g₃⁶)（这些项的表达式从 SMDR `betas.c` 程序化读取）；五圈纯 QCD（g₃、y_t，自写）。')
w('- **m̄_t(m̄_t)**：自写 QCD 四圈 pole↔MS-bar（c₄ = 8.49π⁴，Marquard et al. 2015）。')
w('')
w('## 3. 核心结果（M_t, α_s, M_h) = (172.60, 0.1180, 125.13)')
w('')
w('| 量 | μ = M_t | μ = μ_b | dX(μ_b)/dM_t [GeV⁻¹] | dX(μ_b)/dα_s |')
w('|---|---|---|---|---|')
for k, nm in (('lam', 'λ'), ('yt', 'y_t'), ('g2', 'g₂'), ('gY', 'g_Y'), ('g3', 'g₃')):
    w('| %s | %s | %s | %+.6f | %+.4f |' % (nm, f6(M[k]), f6(C[k]), D['d%s_dMt' % k], D['d%s_das' % k]))
w('')
w('另：g₁(μ_b) = %s；d²λ(μ_b)/dM_t² = %.2e GeV⁻²；dλ(μ_b)/dM_h = %+.6f GeV⁻¹；dλ(μ_b)/dM_W = %+.2e GeV⁻¹。λ(M̄_P) = %s，λ(M_Pl=1.2209e19) = %s。' % (
    f6(C['g1']), D['d2lam_dMt2'], D['dlam_dMh'], D['dlam_dMW'], f6(B['central_at_MPbar']['lam']), f6(B['central_at_MPl']['lam'])))
w('')
w('### 3.1 各 hazard 节点（中心输入）')
w('')
w('| u | μ(u) [GeV] | λ | y_t | g₁ | g₂ | g₃ |')
w('|---|---|---|---|---|---|---|')
for n in B['nodes_central']:
    w('| %.1f | %.4e | %s | %.5f | %.5f | %.5f | %.5f |' % (n['u'], n['mu'], f6(n['lam']), n['yt'], n['g1'], n['g2'], n['g3']))
w('')
w('### 3.2 M_t 网格：λ(μ_b)、y_t(μ_b) 与 m̄_t(m̄_t)')
w('')
w('| M_t | λ(μ_b), α_s=0.1171 | 0.1180 | 0.1189 | y_t(μ_b), 0.1180 | λ(M_t) | y_t(M_t) | m̄_t(m̄_t) |')
w('|---|---|---|---|---|---|---|---|')
for g in B['grid']:
    w('| %.1f | %s | %s | %s | %.5f | %s | %.5f | %.3f |' % (g['Mt'], f6(g['lam_mub_as0.1171']), f6(g['lam_mub_as0.1180']), f6(g['lam_mub_as0.1189']),
                                                          g['yt_mub_as0.1180'], f6(g['lam_Mt']), g['yt_Mt'], g['mtbar_mtbar']))
mt = B['mtbar']
w('')
w('m̄_t(m̄_t)（M_t = 172.60，α_s = 0.1180）：四圈 %.3f，三圈 %.3f，二圈 %.3f，一圈 %.3f GeV；c₄ ± 0.25π⁴ → %+.3f/%+.3f GeV；α_s = 0.1171/0.1189 → %.3f/%.3f GeV；5 圈跑动+4 圈退耦 %.3f GeV。' % (
    mt['central_4l'], mt['central_3l'], mt['central_2l'], mt['central_1l'], mt['c4_minus'] - mt['central_4l'], mt['c4_plus'] - mt['central_4l'],
    mt['as_0p1171'], mt['as_0p1189'], mt['as_5l4d']))
w('')
w('## 4. 理论误差（全部重算；Δλ(μ_b) 与 M_t 当量 Δλ/|dλ/dM_t|）')
w('')
w('| 来源 | Δλ(μ_b) | M_t 当量 [GeV] | Δy_t(μ_b) |')
w('|---|---|---|---|')
rows = [
    ('(a) 匹配标度 μ₀∈[M_t/2,2M_t]（真在 μ₀ 处算阈值，含 α_s 在 μ₀ 退耦）：+/−', None, None),
    ('(b) RGE 截断：4+5 圈项总量（3 圈 vs 中心）', E['b_rge_4l5l_total(3l-only minus central)'], Y['b_rge_4l5l_total(3l-only minus central)']),
    ('(b) 其中五圈 QCD', E['b_rge_5loop_only'], Y['b_rge_5loop_only']),
    ('(b) 四圈非 QCD 部分（full vs 仅纯 QCD）', E['b_rge_4l_nonQCD(full vs qcd-only)'], Y['b_rge_4l_nonQCD(full vs qcd-only)']),
    ('(b) 仅二圈 RGE（参考，不计入）', E['b_rge_2loop_vs_central'], Y['b_rge_2loop_vs_central']),
    ('(b) α_s：5 圈跑动+4 圈退耦 vs 4/3', E['b_as_5l_run_4l_dec'], Y['b_as_5l_run_4l_dec']),
    ('(b) pole 级数：四圈 QCD 项大小（去掉 x04）', E['b_pole_series_4loop_term(no_x04)'], Y['b_pole_series_4loop_term(no_x04)']),
    ('(b) pole 级数：重整子模糊度 0.11 GeV（假设值）', S['b_pole_renormalon_0p11GeV'], B['error_budget_yt_mub_standard']['b_pole_renormalon_0p11GeV']),
    ('(b) 参考：二圈 O(α²) 项大小（去掉 x20）', E['b_EW2loop_size(no_x20)'], Y['b_EW2loop_size(no_x20)']),
    ('(c) α(μ)：AlphaSolve vs AlphaGF', E['c_alpha_solve_vs_GF'], Y['c_alpha_solve_vs_GF']),
    ('(c) W/Z：PDG 值直接当 pole vs M̄ 转换', E['c_MWMZ_PDG_as_pole'], Y['c_MWMZ_PDG_as_pole']),
    ('(c) 实现/组织方式：SMDR − mr（同 M_W）', E['c_SMDR_minus_mr_MWpred(implementation)'], Y['c_SMDR_minus_mr_MWpred(implementation)']),
    ('(c) 输入集：M_W 实测 vs SMDR 预言 %.4f' % B['smdr_central']['MW_PDG_pred'], E['c_input_set(mr MW_pred - mr MW_meas)'], Y['c_input_set(mr MW_pred - mr MW_meas)']),
    ('(c) tadpole 约定检验：FJ 跑动质量比值不重展开（报告，不计入）', T['delta_vs_central_at_mub']['lam'], T['delta_vs_central_at_mub']['yt'])]
for nm, x, y in rows:
    if x is None:
        w('| %s | %+.5f / %+.5f | %+.3f / %+.3f | %+.5f / %+.5f |' % (nm, A['lam']['plus'], A['lam']['minus'], -A['lam']['plus'] / dl, -A['lam']['minus'] / dl,
                                                                   A['yt']['plus'], A['yt']['minus']))
    else:
        w('| %s | %s | %.3f | %s |' % (nm, e(x), abs(x) / dl, e(y)))
w('')
w('**采用的合成（标准）**：(a) 取最大偏离 %.5f + (b) RGE、α_s、重整子 0.11 GeV + (c) α、W/Z 约定、SMDR−mr、输入集，平方和 → **Δλ(μ_b) = ±%.5f（M_t 当量 ±%.3f GeV）**；y_t(μ_b) 同法 ±%.5f。保守变体（再把整个四圈 QCD 项当误差）：±%.5f（±%.3f GeV）。主导项是匹配标度（M_t 当量 −%.2f/+%.2f GeV），与 BKPV 2015 表 I 中 M_t^cri 的 μ 变化（−0.36/+0.17 GeV）同量级、同号型。' % (
    S['a_mu0_maxdev'], S['quadrature'], SM['quadrature'], B['error_budget_yt_mub_standard']['quadrature'], CS['quadrature'], CS['quadrature'] / dl,
    A['lam']['plus'] / dl, -A['lam']['minus'] / dl))
w('')
w('μ₀ 扫描中 λ(μ_b)：' + '，'.join('%s: %s' % (k, f6(v)) for k, v in A['lam']['values'].items()) + '。回跑到 M_t 的 λ(M_t)：' +
  '，'.join('%s: %s' % (k, f6(v)) for k, v in A['lam_Mt_backrun'].items()) + '；y_t(M_t)：' + '，'.join('%s: %.6f' % (k, v) for k, v in A['yt_Mt_backrun'].items()) + '。')
w('纯 QCD 部分（只用 x01–x04、纯 QCD 四圈回跑）：y_t(M_t) = ' + '，'.join('%s: %.6f' % (k, v) for k, v in A['yt_Mt_QCDonly_pureQCDbackrun'].items()) +
  '——相对 ±(3–4)×10⁻⁴（≈60–70 MeV），即 pole 质量级数 O(α_s⁵) 截断（重整子增长）的表现，而非实现错误：mr 的 y01–y04 在四个 μ₀ ≠ M_t 处与由 y0i(M_t) 及 n_f = 6 四圈 β/γ_m 推出的 RG 预言一致到 %.0e（`work/rg_check_y0i.py` → `results/rg_check_y0i.json`）。' % max(B['rg_check_mr_qcd_coeffs']['max_rel_dev_by_order'].values()))
w('')
w('## 5. 文献基准复现')
b1 = V['B1']
w('| 基准 | 本工作 | 文献 | 判断 |')
w('|---|---|---|---|')
o = b1['mr_NNLO_plus_3lQCD']; p = b1['published_Mt_scale']
w('| Buttazzo 2013 §3：λ(M_t)（其输入，NNLO） | %.5f | 0.12604 ± 0.00030 | 通过（Δ = %+.1e） |' % (o['lam'], o['lam'] - 0.12604))
w('| Buttazzo 2013：y_t(M_t)（NNLO + 3 圈 QCD） | %.5f | 0.93690 ± 0.00050 | 通过（Δ = %+.1e） |' % (o['yt'], o['yt'] - 0.93690))
w('| Buttazzo 2013：g₂(M_t) / g_Y(M_t) | %.5f / %.5f | 0.64779 / 0.35830 | 部分（PDG 值当 pole 时 +3×10⁻⁴；M̄ 转换后 0.64788/0.35824） |' % (o['g2'], o['gY']))
w('| Buttazzo 2013：g₃(M_t) | %.5f | 1.1666 | 未通过：文献值比标准 4 圈跑动+3 圈退耦高 0.0018（SMDR 与本工作一致），原因未查明 |' % o['g3'])
pp = b1['Planck_from_published_Mt_values_4lqcd']; pub = b1['published_Planck']
w('| Buttazzo Eq.(sys:Pl)：由其 M_t 处耦合跑到 M_Pl | λ %.5f, y_t %.4f, g₁ %.4f, g₂ %.4f, g₃ %.4f | −0.0143, 0.3825, 0.6154, 0.5055, 0.4873 | 通过（到所给位数） |' % (
    pp['lam'], pp['yt'], pp['g1'], pp['g2'], pp['g3']))
b2 = V['B2']
w('| BKPV 2015（PRL 115, 201802）临界 M_t（λ = β_λ = 0；M_H = 125.7, α_s = 0.1185） | M_t^cri = %.3f GeV，log₁₀μ_cri = %.3f | 171.44，17.752 | 通过（Δ = %+.0f MeV） |' % (
    b2['Mt_cri'], b2['log10_mu_cri'], (b2['Mt_cri'] - 171.44) * 1000))
b3 = V['B3']; sf = b3['smdr_fit']; rf = b3['reference']
w('| SMDR ReferenceModel_2025（自编译 SMDR 复算） | g₃ %.8f, y_t %.7f, λ %.7f | %.8f, %.7f, %.7f | 通过（≤5×10⁻⁶） |' % (
    sf['g3_in'], sf['yt_in'], sf['lambda_in'], rf['g3'], rf['yt'], rf['lam']))
w('| β 函数：rge.m(BPV) vs Buttazzo 附录逐项 | %d 个单项式，%d 个不符 | — | 通过 |' % (TC['n_monomials_compared'], TC['n_mismatch']))
mx = max(r['maxrel'] for r in VT['T2'] if r['loop'] <= 4)
w('| β 函数：自写 Python vs SMDR betas.c（3 个随机点，1–4 圈） | 最大相对差 %.1e | — | 通过（5 圈 1e-9，来自数值系数舍入） |' % mx)
w('')
w('## 6. 实现层面的发现（与生产对照时请优先检查）')
for n in B['notes']:
    w('- ' + n)
w('- mr 与 SMDR 在 μ = M_t：Δλ = %+.2e，Δy_t = %+.2e，Δg₂ = %+.2e，Δg_Y = %+.2e，Δg₃ = %+.2e（SMDR − mr）；μ_b 处 Δλ = %+.2e（%.3f GeV）。自写 RGE 与 SMDR 自带 RG（5 圈选项）跑同一 SMDR 初值到 μ_b：Δλ = %+.1e。' % (
    B['smdr_central']['at_Mt']['lam'] - M['lam'], B['smdr_central']['at_Mt']['yt'] - M['yt'], B['smdr_central']['at_Mt']['g2'] - M['g2'],
    B['smdr_central']['at_Mt']['gY'] - M['gY'], B['smdr_central']['at_Mt']['g3'] - M['g3'], E['c_SMDR_minus_mr'], abs(E['c_SMDR_minus_mr']) / dl,
    B['smdr_central']['smdr_own_RG_mub']['lam'] - B['smdr_central']['at_mub']['lam']))
w('- FJ tadpole 大小（说明为何 y-展开更稳）：t_x10 = %.1f vs t_y10 = %.2f；H_x10 = %.1f vs H_y10 = %.2f；dr10 = %.1f。一圈关系 y = x − dr（玻色子平方质量）、y_t = x_t − dr/2 数值成立到 ~10⁻² 量级的二阶差。' % (
    T['tadpole_sizes']['t_x10'], T['tadpole_sizes']['t_y10'], T['tadpole_sizes']['H_x10'], T['tadpole_sizes']['H_y10'], T['tadpole_sizes']['dr10']))
w('')
w('## 7. 结论分级')
w('### 已证明（本轮内独立复算与交叉核验）')
w('1. 三圈 SM β 函数：BPV 三圈（mr `rge.m`）与 Buttazzo et al. 附录逐项一致（%d/%d），与 SMDR `betas.c` 数值一致到 10⁻¹⁵（1–4 圈）。' % (TC['n_monomials_compared'] - TC['n_mismatch'], TC['n_monomials_compared']))
w('2. 两个以上文献基准复现：BKPV 临界 M_t（差 %+.0f MeV）；Buttazzo 的 Planck 标度耦合（由其 M_t 处耦合出发，到所给位数）；Buttazzo 的 λ(M_t)、y_t(M_t) 在其理论误差内；SMDR 参考点 ≤5×10⁻⁶。' % ((b2['Mt_cri'] - 171.44) * 1000))
w('3. 在第 1 节约定下：λ(μ_b) = %s，y_t(μ_b) = %s，g₂ = %s，g_Y = %s，g₃ = %s；dλ(μ_b)/dM_t = %+.5f GeV⁻¹，dλ(μ_b)/dα_s = %+.3f。两套匹配（mr、SMDR）在 μ_b 处 λ 相差 %.1e（%.3f GeV）。' % (
    f6(C['lam']), f6(C['yt']), f6(C['g2']), f6(C['gY']), f6(C['g3']), D['dlam_dMt'], D['dlam_das'], abs(E['c_SMDR_minus_mr']), abs(E['c_SMDR_minus_mr']) / dl))
w('4. mr 以 OpenMP 多线程运行时结果非确定（TSIL 竞态），单线程结果确定且与 P2MS 自身组装逐位一致。')
w('5. mr 纯 QCD 系数 y01–y04 的标度依赖与 n_f = 6 四圈 QCD RG 精确一致（%.0e）；因此 y_t 匹配标度依赖中的纯 QCD 部分（≈±60–70 MeV）是级数截断效应。' % max(B['rg_check_mr_qcd_coeffs']['max_rel_dev_by_order'].values()))
w('### 条件成立')
w('1. 误差合成依赖约定：(a) 用 μ₀ ∈ [M_t/2, 2M_t] 的最大偏离；重整子模糊度取 0.11 GeV（文献量级，本轮未重读原文）；保守变体把整个四圈 QCD 项计入。')
w('2. 输入：M_t 按 pole 解释（任务书要求）；W/Z 采用 M̄ 复极点约定（PDG 直接当 pole 的差 %.1e 已列入）；SMDR 用 Δα_had⁽⁵⁾ = 0.02783（未更新到 2026）、α(0) CODATA；mr 环路中 M_b = 4.78；y_τ(μ₀) 固定 0.009995、y_b 由 m_b(m_b) 4 圈 QCD 跑动；G_F 取自 PDG 2026 表的文本抽取值。' % abs(E['c_MWMZ_PDG_as_pole']))
w('3. 四圈 RGE 中 λ 仅含 O(y_t⁴g₃⁶)、y_t 仅含纯 QCD（与 SMDR 同）；其余四圈项未知，其影响以已知四圈项总量（%.1e）为量级估计。' % abs(E['b_rge_4l5l_total(3l-only minus central)']))
w('### 未证明')
w('1. Buttazzo 2013 的 g₃(M_t) = 1.1666 与标准处理差 0.0018 的来源。')
w('2. SMDR 内部的匹配标度变化（其 Fit_Inputs 在 Q ≠ M_t 的输出由 RG 跑动得到，不构成标度变化）。')
w('3. 三圈 EW/混合阈值（超出 SMDR 已含部分）的实际大小；本报告只用 μ₀ 变化估计。')
w('4. M_t^MC 与 pole 质量的非微扰关系；α_s 在 M_Z–M_t 间的 QED 修正（≲10⁻⁵，略去）。')
w('')
w('## 8. 与生产数字的对照（留空，待主会话发来生产数字后填写）')
w('')
w('| 量 | 盲核验 | 生产 | 差 | 诊断 |')
w('|---|---|---|---|---|')
for nm in ('λ(μ_b)', 'y_t(μ_b)', 'g₂(μ_b)', 'g_Y(μ_b)', 'g₃(μ_b)', 'dλ/dM_t', 'dλ/dα_s', 'λ(M_t)', 'y_t(M_t)', 'm̄_t(m̄_t)'):
    w('| %s |  |  |  |  |' % nm)
w('')
w('## 9. 文件')
w('- 代码：`work/smcore.py`（α_s、m̄_t、RGE、驱动接口）、`work/rge_convert_check.py` → `work/rge_bpv3_gen.py`、`work/production.py`、`work/post.py`、`work/diag_mu0.py`、`work/test_core.py`、`work/build_all.py`、`work/drivers/{mr_driver.cpp,smdr_driver.c}`。')
w('- 结果：`results/BLIND_RESULTS.json`（主）、`results/prod_runs.json`、`results/prod_mr_records.json`、`results/prod_smdr_records.json`、`results/validation_*.json`、`results/rge_termcheck_vs_Buttazzo.json`、`results/diag_mu0.json`。')
w('- 下载（各含 SOURCES.json：URL、日期、SHA-256）：`inputs/20261014_{mr_github,mr_release,smdr,tsil,arxiv_pdfs,arxiv_src,pdg}`；解包目录 `*_extracted`。')
w('- 代码 SHA-256（节选）：' + '；'.join('%s %s…' % (k, v[:16]) for k, v in list(B['code_sha256'].items())[:6]))
open(os.path.join(MY, 'R16_V1_SM_MAPPING_BLIND_CN.md'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('report written,', len(L), 'lines')
