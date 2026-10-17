import os, sys, math, json, time
WORK = r'D:\0PhD\Perimeter\Research\2026\Directional_Null_Energy_Cosmology\07_reviews\R16_v12_round3\sm_mapping_blind\work'
sys.path.insert(0, WORK)
import smcore as S
Mt, Mh, MW, MZ, asMZ = 173.34, 125.15, 80.384, 91.1876, 0.1184
Gf = 1 / (math.sqrt(2) * 246.21971 ** 2)
a6 = S.as6(Mt, asMZ, Mt, MZ, 4, 3)
pt = dict(Mb=4.78, MW=MW, MZ=MZ, MH=Mh, Mt=Mt, Gf=Gf, as_mu=a6, mu=Mt)
for nth in (1, 1, 4, 4, 4):
    S.MR_THREADS = nth
    t0 = time.time(); r = S.mr_eval([pt, pt])
    for rec in r:
        c = S.mr_couplings(rec, 'GF', S.ALL_ORDERS)
        print('threads', nth, '%.2fs' % (time.time() - t0), 'P2MS g2 %.10f g1 %.10f lam %.10f yt %.10f | mine g2 %.10f g1 %.10f | Wy10 %.8f Wy20 %.6f Zy20 %.6f Hy20 %.6f' % (
            rec['GF_g2'], rec['GF_g1'], rec['GF_lam'], rec['GF_yt'], c['g2'], c['g1'], rec['W_y10'], rec['W_y20'], rec['Z_y20'], rec['H_y20']))