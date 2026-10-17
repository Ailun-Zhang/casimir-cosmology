/* PROVENANCE: 04_calculations/code/20261014_claude_v12_r3/sm_mapping/c148_fit.c (copied 2026-10-15 by code-higgs).
   SMDR 1.3 refit driver used by run02_fits.py; build: python third_party/smdr/fetch_and_build.py --driver this file. */
/* SPDX-License-Identifier: GPL-2.0-or-later
   C148_Fit_Inputs below is SMDR_Fit_Inputs of SMDR 1.3 (src/fit_obs.c), Copyright (C) 2019 S.P. Martin and
   D.G. Robertson, distributed under the GNU General Public License version 2 or (at your option) any later
   version (third_party/smdr/licenses/SMDR-1.3_LICENSE.txt). Modified 2026-10-14 (agent sm-mapping): the
   evaluation scales are arguments (see the next comment). This file is distributed under the same GPL terms,
   not under the package's MIT license. */
/* C148 run02 driver (agent sm-mapping, 2026-10-14).
   C148_Fit_Inputs = SMDR 1.3 src/fit_obs.c SMDR_Fit_Inputs (lines 11-345) copied verbatim except that the
   hard-wired evaluation scales are arguments: Mt pole at Qt (SMDR: SMDR_Mt_EXPT), Mh at Qh (160), MZ and MW at
   QZ (160), GFermi at QG (SMDR_Mt_EXPT), QCD/QED matching at Qdec (SMDR_MZ_EXPT).
   Usage:  c148_fit time
           c148_fit fit Mt Mh alphaS xit xih xiZ xiG xidec Qout tol outfile
   (scales: Qt = xit*Mt, Qh = xih*160, QZ = xiZ*160, QG = xiG*Mt, Qdec = xidec*MZ_EXPT) */
#include "smdr_internal.h"
#include <time.h>
#define ZEROSAFE(a) (((a) > (SMDR_TOL)) ? (a) : (SMDR_TOL))

static double now(void) { return (double) clock() / CLOCKS_PER_SEC; }

int C148_Fit_Inputs (SMDR_REAL Q_target, SMDR_REAL alphaS_5_MZ_target, SMDR_REAL alpha_target,
                     SMDR_REAL GFermi_target, SMDR_REAL MZ_target, SMDR_REAL Mh_target, SMDR_REAL Mt_target,
                     SMDR_REAL mbmb_target, SMDR_REAL mcmc_target, SMDR_REAL ms_2GeV_target,
                     SMDR_REAL md_2GeV_target, SMDR_REAL mu_2GeV_target, SMDR_REAL Mtau_target,
                     SMDR_REAL Mmuon_target, SMDR_REAL Melectron_target, SMDR_REAL Delta_alpha_had_target,
                     SMDR_REAL error_target,
                     SMDR_REAL Qt, SMDR_REAL Qh, SMDR_REAL QZ, SMDR_REAL QG, SMDR_REAL Qdec)
{
  int j;
  SMDR_REAL vratio, v2ratio, KZ, Ka;
  SMDR_REAL err_total, err_alphaS, err_alpha, err_MZ, err_GFermi, err_Mh;
  SMDR_REAL err_Mt, err_mbmb, err_mcmc, err_ms_2GeV, err_md_2GeV;
  SMDR_REAL err_mu_2GeV, err_Mtau, err_Mmuon, err_Melectron;
  SMDR_REAL g3rat, lambdarat, ytrat, ybrat, ycrat, ysrat, ydrat, yurat, ytaurat, ymurat, yerat;
  SMDR_REAL del_v, del_g3, del_g, del_gp, del_yt, del_yb, del_yc, del_lambda;
  SMDR_REAL del_ys, del_yu, del_yd, del_ytau, del_ymu, del_ye;
  int result_niters = -1;
  #include "accelcoeffs.h"

  SMDR_MTPOLE_TOLERANCE = 1.0e-8;
  SMDR_MHPOLE_TOLERANCE = 1.0e-8;
  if (error_target < 1.0e-7) { SMDR_MTPOLE_TOLERANCE = error_target/10.; SMDR_MHPOLE_TOLERANCE = error_target/10.; }
  if (SMDR_MTPOLE_TOLERANCE < 1.01e-15) SMDR_MTPOLE_TOLERANCE = 1.01e-15;
  if (SMDR_MHPOLE_TOLERANCE < 1.01e-15) SMDR_MHPOLE_TOLERANCE = 1.01e-15;
  #include "AUTOrefmodel.h"
  SMDR_RGeval_SM (Mt_target, 5);
  SMDR_Save_Inputs ();
  SMDR_Delta_alpha_had_5_MZ_in = Delta_alpha_had_target;
  for (j=0; j<12; j++) {
    v2ratio = SMDR_GFermi/GFermi_target;
    vratio = SMDR_SQRT (v2ratio);
    KZ = (SMDR_g_in * SMDR_g_in + SMDR_gp_in * SMDR_gp_in) * (MZ_target * MZ_target)/
         (SMDR_MZ_PDG * SMDR_MZ_PDG * v2ratio);
    Ka = ((SMDR_g_in * SMDR_g_in * SMDR_gp_in * SMDR_gp_in)/(SMDR_g_in * SMDR_g_in + SMDR_gp_in * SMDR_gp_in)) *
         (alpha_target/SMDR_alpha);
    del_g  = (SMDR_SQRT(KZ * (1. + SMDR_SQRT(1. - 4. * Ka/KZ))/2.)/g_in) - 1.;
    del_gp = (SMDR_SQRT(KZ * (1. - SMDR_SQRT(1. - 4. * Ka/KZ))/2.)/gp_in) - 1.;
    del_g3 = SMDR_SQRT (alphaS_5_MZ_target/SMDR_alphaS_5_MZ) - 1.;
    del_v = vratio - 1.;
    del_lambda = ((Mh_target * Mh_target)/(SMDR_Mh_pole * SMDR_Mh_pole * v2ratio)) - 1.;
    del_yt = (Mt_target/(SMDR_Mt_pole * vratio)) - 1.;
    del_yb = (mbmb_target/(ZEROSAFE(SMDR_mbmb) * vratio)) - 1.;
    del_yc = (mcmc_target/(ZEROSAFE(SMDR_mcmc) * vratio)) - 1.;
    del_ys = (ms_2GeV_target/(ZEROSAFE(SMDR_ms_2GeV) * vratio)) - 1.;
    del_yd = (md_2GeV_target/(ZEROSAFE(SMDR_md_2GeV) * vratio)) - 1.;
    del_yu = (mu_2GeV_target/(ZEROSAFE(SMDR_mu_2GeV) * vratio)) - 1.;
    del_ytau = (Mtau_target/(ZEROSAFE(SMDR_Mtau_pole) * vratio)) - 1.;
    del_ymu = (Mmuon_target/(ZEROSAFE(SMDR_Mmuon_pole) * vratio)) - 1.;
    del_ye = (Melectron_target/(ZEROSAFE(SMDR_Melectron_pole) * vratio)) - 1.;
    g3rat = 1. + cog3_g3 * del_g3 + cog3_g32 * del_g3 * del_g3 + cog3_yt * del_yt;
    lambdarat = 1. + del_lambda + colambda_yt * del_yt + colambda_g3 * del_g3;
    ytrat = 1. + coyt_yt * del_yt + coyt_yt2 * del_yt * del_yt + coyt_g3 * del_g3;
    ybrat = 1. + coyb_yb * del_yb + coyb_yb2 * del_yb * del_yb + + coyb_g3 * del_g3 + coyb_yt * del_yt;
    ycrat = 1. + coyc_yc * del_yc + coyc_yc2 * del_yc * del_yc + coyc_g3 * del_g3 + coyc_yt * del_yt + coyc_yb * del_yb;
    ysrat = 1. + del_ys + coys_g3 * del_g3 + coys_yt * del_yt + coys_yb * del_yb;
    ydrat = 1. + del_yd + coyd_g3 * del_g3 + coyd_yt * del_yt + coyd_yb * del_yb;
    yurat = 1. + del_yu + coyu_g3 * del_g3 + coyu_yt * del_yt + coyu_yb * del_yb;
    ytaurat = 1. + del_ytau; ymurat = 1. + del_ymu; yerat = 1. + del_ye;
    if ((ybrat < 0) || (mbmb_target < SMDR_TOL)) ybrat = 0;
    if ((ycrat < 0) || (mcmc_target < SMDR_TOL)) ycrat = 0;
    if ((ysrat < 0) || (ms_2GeV_target < SMDR_TOL)) ysrat = 0;
    if ((ydrat < 0) || (md_2GeV_target < SMDR_TOL)) ydrat = 0;
    if ((yurat < 0) || (mu_2GeV_target < SMDR_TOL)) yurat = 0;
    if ((ytaurat < 0) || (Mtau_target < SMDR_TOL)) ytaurat = 0;
    if ((ymurat < 0) || (Mmuon_target < SMDR_TOL)) ymurat = 0;
    if ((yerat < 0) || (Melectron_target < SMDR_TOL)) yerat = 0;
    SMDR_v_in *= vratio; SMDR_g3_in *= g3rat; SMDR_g_in *= 1. + del_g; SMDR_gp_in *= 1. + del_gp;
    SMDR_lambda_in *= lambdarat; SMDR_yt_in *= ytrat; SMDR_yb_in *= ybrat; SMDR_yc_in *= ycrat;
    SMDR_ys_in *= ysrat; SMDR_yd_in *= ydrat; SMDR_yu_in *= yurat; SMDR_ytau_in *= ytaurat;
    SMDR_ymu_in *= ymurat; SMDR_ye_in *= yerat;
    err_alphaS = SMDR_alphaS_5_MZ/alphaS_5_MZ_target - 1.;
    err_alpha = SMDR_alpha/alpha_target - 1.;
    err_MZ = SMDR_MZ_PDG/MZ_target - 1.;
    err_GFermi = SMDR_GFermi/GFermi_target - 1.;
    err_Mh = SMDR_Mh_pole/Mh_target - 1.;
    err_Mt = SMDR_Mt_pole/Mt_target - 1.;
    err_mbmb = (SMDR_mbmb - mbmb_target)/SMDR_mbmb_EXPT;
    err_mcmc = (SMDR_mcmc - mcmc_target)/SMDR_mcmc_EXPT;
    err_ms_2GeV = (SMDR_ms_2GeV - ms_2GeV_target)/SMDR_ms_2GeV_EXPT;
    err_md_2GeV = (SMDR_md_2GeV - md_2GeV_target)/SMDR_md_2GeV_EXPT;
    err_mu_2GeV = (SMDR_mu_2GeV - mu_2GeV_target)/SMDR_mu_2GeV_EXPT;
    err_Mtau = (SMDR_Mtau_pole - Mtau_target)/SMDR_Mtau_EXPT;
    err_Mmuon = (SMDR_Mmuon_pole - Mmuon_target)/SMDR_Mmuon_EXPT;
    err_Melectron = (SMDR_Melectron_pole - Melectron_target)/SMDR_Melectron_EXPT;
    err_total = SMDR_FABS(err_alphaS) + SMDR_FABS(err_alpha) + SMDR_FABS(err_MZ) + SMDR_FABS(err_GFermi) +
                SMDR_FABS(err_Mh) + SMDR_FABS(err_Mt) + SMDR_FABS(err_mbmb) + SMDR_FABS(err_mcmc) +
                SMDR_FABS(err_ms_2GeV) + SMDR_FABS(err_md_2GeV) + SMDR_FABS(err_mu_2GeV) + SMDR_FABS(err_Mtau) +
                SMDR_FABS(err_Mmuon) + SMDR_FABS(err_Melectron);
    if (j>0) {
      printf("Iteration %d: Total fractional error = %.15Le  (t=%.1fs)\n", j, err_total, now());
      fflush(stdout);
      if (err_total < error_target) break;
    }
    SMDR_Eval_Mt_pole (Qt, 1, 4, 2, &SMDR_Mt_pole, &SMDR_Gammat_pole);
    SMDR_Eval_Mh_pole (Qh, 2.5, &SMDR_Mh_pole, &SMDR_Gammah_pole);
    SMDR_Eval_MZ_pole (QZ, 2.5, &SMDR_MZ_pole, &SMDR_GammaZ_pole, &SMDR_MZ_PDG, &SMDR_GammaZ_PDG);
    SMDR_Eval_MW_pole (QZ, 2.5, &SMDR_MW_pole, &SMDR_GammaW_pole, &SMDR_MW_PDG, &SMDR_GammaW_PDG);
    SMDR_GFermi = SMDR_Eval_GFermi (QG, 3);
    SMDR_Eval_Gauge (SMDR_Mt_pole, SMDR_Mh_pole, SMDR_MW_PDG);
    SMDR_Eval_QCDQED_at_MZ (SMDR_MZ_EXPT, Qdec, 5);   /* (Q_MZ, Q_dec_thZW, loopOrder): vary the decoupling scale */
    if (mbmb_target > SMDR_TOL) SMDR_mbmb = SMDR_Eval_mbmb(SMDR_MZ_EXPT, 5); else SMDR_mbmb = 0;
    if (mcmc_target > SMDR_TOL) SMDR_mcmc = SMDR_Eval_mcmc(SMDR_Mtau_EXPT, SMDR_mbmb_EXPT, SMDR_MZ_EXPT, 5);
    else SMDR_mcmc = 0;
    SMDR_Eval_mquarks_2GeV (SMDR_mbmb_EXPT, SMDR_MZ_EXPT, 5, &SMDR_ms_2GeV, &SMDR_mu_2GeV, &SMDR_md_2GeV);
    SMDR_Mtau_pole = SMDR_Eval_Mtau_pole (SMDR_Mtau_EXPT, SMDR_mbmb_EXPT, SMDR_MZ_EXPT, 3);
    SMDR_Mmuon_pole = SMDR_Eval_Mmuon_pole (SMDR_mcmc_EXPT, SMDR_mcmc_EXPT, SMDR_Mtau_EXPT, SMDR_mbmb_EXPT,
                                            SMDR_MZ_EXPT, 2);
    SMDR_Melectron_pole = SMDR_Eval_Melectron_pole (SMDR_mcmc_EXPT, SMDR_mcmc_EXPT, SMDR_Mtau_EXPT,
                                                    SMDR_mbmb_EXPT, SMDR_MZ_EXPT, 2);
    result_niters = j;
  }
  SMDR_Load_Inputs();
  SMDR_Lambda = 0;
  if (getenv("C148_VEFF")) {   /* vacuum energy only; irrelevant for the couplings -> skipped by default */
    SMDR_Lambda_in = SMDR_Lambda = -SMDR_Eval_Veffmin (-1, 3.5);
    SMDR_m2_in = SMDR_m2;
  }
  SMDR_RGeval_SM (Q_target, 5);
  SMDR_Save_Inputs();
  return(result_niters);
}

static void timing(void)
{
  double t;
  #include "AUTOrefmodel.h"
  t = now(); SMDR_Eval_Mt_pole (172.4, 1, 4, 2, &SMDR_Mt_pole, &SMDR_Gammat_pole);
  printf("Mt  %.10Lf  %.2fs\n", SMDR_Mt_pole, now() - t); fflush(stdout);
  t = now(); SMDR_Eval_MZ_pole (160., 2.5, &SMDR_MZ_pole, &SMDR_GammaZ_pole, &SMDR_MZ_PDG, &SMDR_GammaZ_PDG);
  printf("MZ  %.10Lf  %.2fs\n", SMDR_MZ_PDG, now() - t); fflush(stdout);
  t = now(); SMDR_Eval_MW_pole (160., 2.5, &SMDR_MW_pole, &SMDR_GammaW_pole, &SMDR_MW_PDG, &SMDR_GammaW_PDG);
  printf("MW  %.10Lf  %.2fs\n", SMDR_MW_PDG, now() - t); fflush(stdout);
  t = now(); SMDR_GFermi = SMDR_Eval_GFermi (172.4, 3);
  printf("GF  %.12Le  %.2fs\n", SMDR_GFermi, now() - t); fflush(stdout);
  t = now(); SMDR_Eval_Mh_pole (160., 2.5, &SMDR_Mh_pole, &SMDR_Gammah_pole);
  printf("Mh  %.10Lf  %.2fs\n", SMDR_Mh_pole, now() - t); fflush(stdout);
}

int main (int argc, char *argv[])
{
  if (argc > 1 && !strcmp(argv[1], "time")) { timing(); return 0; }
  if (argc < 13) { fprintf(stderr, "usage: see header\n"); return 1; }
  SMDR_REAL Mt = strtold(argv[2], 0), Mh = strtold(argv[3], 0), aS = strtold(argv[4], 0);
  SMDR_REAL xit = strtold(argv[5], 0), xih = strtold(argv[6], 0), xiZ = strtold(argv[7], 0);
  SMDR_REAL xiG = strtold(argv[8], 0), xid = strtold(argv[9], 0), Qout = strtold(argv[10], 0);
  SMDR_REAL tol = strtold(argv[11], 0);
  double t0 = now();
  int n = C148_Fit_Inputs (Qout, aS, SMDR_alpha_EXPT, SMDR_GFermi_EXPT, SMDR_MZ_EXPT, Mh, Mt, SMDR_mbmb_EXPT,
                           SMDR_mcmc_EXPT, SMDR_ms_2GeV_EXPT, SMDR_md_2GeV_EXPT, SMDR_mu_2GeV_EXPT, SMDR_Mtau_EXPT,
                           SMDR_Mmuon_EXPT, SMDR_Melectron_EXPT, SMDR_Delta_alpha_had_5_MZ_EXPT, tol,
                           xit * Mt, xih * 160., xiZ * 160., xiG * Mt, xid * SMDR_MZ_EXPT);
  FILE *f = fopen(argv[12], "w");
  fprintf(f, "niter %d\ntime %.2f\nMW_PDG %.12Le\n", n, now() - t0, SMDR_MW_PDG);
  fprintf(f, "Q %.12Le\ng3 %.18Le\ng %.18Le\ngp %.18Le\nyt %.18Le\nyb %.18Le\nytau %.18Le\nlambda %.18Le\nv %.18Le\n",
          SMDR_Q_in, SMDR_g3_in, SMDR_g_in, SMDR_gp_in, SMDR_yt_in, SMDR_yb_in, SMDR_ytau_in, SMDR_lambda_in, SMDR_v_in);
  {
    SMDR_REAL mub = 2.435e18L / 722.0L;
    SMDR_RGeval_SM (mub, 5);
    fprintf(f, "RG_mub %.12Le g3 %.18Le g %.18Le gp %.18Le yt %.18Le lambda %.18Le\n",
            mub, SMDR_g3, SMDR_g, SMDR_gp, SMDR_yt, SMDR_lambda);
  }
  fclose(f);
  printf("done niter=%d t=%.1fs\n", n, now() - t0);
  return 0;
}
