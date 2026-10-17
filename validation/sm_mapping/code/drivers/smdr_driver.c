/* sm-blind (R16 V1): driver for SMDR 1.3 (Martin & Robertson, arXiv:1907.02500), second (cross-check) matching.
   Commands on stdin, one JSON line per command on stdout:
     fit Q Mt Mh MZPDG asMZ alpha GF mbmb mcmc ms2 md2 mu2 Mtau Mmu Me dalhad
     run Qfinal loopOrder            (runs from the fitted *_in values)
     beta g3 g gp yt yb ytau lambda loopOrder   (Q d/dQ beta functions; light Yukawas = 0)          */
#include <stdio.h>
#include <string.h>
#include "smdr.h"
extern SMDR_REAL SMDR_beta_g3, SMDR_beta_g, SMDR_beta_gp, SMDR_beta_yt, SMDR_beta_yb, SMDR_beta_ytau, SMDR_beta_k;
int SMDR_Betas (int loopOrder);

static void pr_state (const char *tag)
{
  printf ("{\"cmd\":\"%s\",\"Q\":%.18Lg,\"g3\":%.18Lg,\"g\":%.18Lg,\"gp\":%.18Lg,\"yt\":%.18Lg,\"yb\":%.18Lg,"
          "\"ytau\":%.18Lg,\"lambda\":%.18Lg,\"v\":%.18Lg,\"m2\":%.18Lg,\"MW_pole\":%.18Lg,\"MW_PDG\":%.18Lg,"
          "\"MZ_pole\":%.18Lg,\"Q_in\":%.18Lg,\"g3_in\":%.18Lg,\"yt_in\":%.18Lg,\"lambda_in\":%.18Lg}\n",
          tag, SMDR_Q, SMDR_g3, SMDR_g, SMDR_gp, SMDR_yt, SMDR_yb, SMDR_ytau, SMDR_lambda, SMDR_v, SMDR_m2,
          SMDR_MW_pole, SMDR_MW_PDG, SMDR_MZ_pole, SMDR_Q_in, SMDR_g3_in, SMDR_yt_in, SMDR_lambda_in);
  fflush (stdout);
}

int main (void)
{
  char line[4096], cmd[16];
  while (fgets (line, sizeof line, stdin)) {
    if (sscanf (line, "%15s", cmd) != 1 || cmd[0] == '#') continue;
    if (!strcmp (cmd, "fit")) {
      SMDR_REAL Qt, Mt, Mh, MZ, as, al, GF, mb, mc, ms, md, mu, Mta, Mmu, Me, dah;
      if (sscanf (line + 3, "%Lf %Lf %Lf %Lf %Lf %Lf %Lf %Lf %Lf %Lf %Lf %Lf %Lf %Lf %Lf %Lf",
                  &Qt, &Mt, &Mh, &MZ, &as, &al, &GF, &mb, &mc, &ms, &md, &mu, &Mta, &Mmu, &Me, &dah) != 16) {
        printf ("{\"error\":\"fit args\"}\n"); continue; }
      SMDR_Fit_Inputs (Qt, as, al, GF, MZ, Mh, Mt, mb, mc, ms, md, mu, Mta, Mmu, Me, dah, 1.e-12L);
      pr_state ("fit");
    } else if (!strcmp (cmd, "run")) {
      SMDR_REAL Qf; int lo;
      if (sscanf (line + 3, "%Lf %d", &Qf, &lo) != 2) { printf ("{\"error\":\"run args\"}\n"); continue; }
      SMDR_RGeval_SM (Qf, lo);
      pr_state ("run");
    } else if (!strcmp (cmd, "beta")) {
      SMDR_REAL a[7]; int lo;
      if (sscanf (line + 4, "%Lf %Lf %Lf %Lf %Lf %Lf %Lf %d", &a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &lo) != 8) {
        printf ("{\"error\":\"beta args\"}\n"); continue; }
      SMDR_g3 = a[0]; SMDR_g = a[1]; SMDR_gp = a[2]; SMDR_yt = a[3]; SMDR_yb = a[4]; SMDR_ytau = a[5]; SMDR_lambda = a[6];
      SMDR_yc = SMDR_ys = SMDR_yu = SMDR_yd = SMDR_ymu = SMDR_ye = 0; SMDR_m2 = -8000; SMDR_v = 246; SMDR_Lambda = 0;
      SMDR_Betas (lo);
      printf ("{\"cmd\":\"beta\",\"loop\":%d,\"b_g3\":%.18Lg,\"b_g\":%.18Lg,\"b_gp\":%.18Lg,\"b_yt\":%.18Lg,"
              "\"b_yb\":%.18Lg,\"b_ytau\":%.18Lg,\"b_k\":%.18Lg}\n", lo, SMDR_beta_g3, SMDR_beta_g, SMDR_beta_gp,
              SMDR_beta_yt, SMDR_beta_yb, SMDR_beta_ytau, SMDR_beta_k);
      fflush (stdout);
    }
  }
  return 0;
}
