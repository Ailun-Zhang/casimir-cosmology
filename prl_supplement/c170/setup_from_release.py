"""Copy the unchanged production modules used by C170 from the code release v1.0.0 into ./c170_modules and verify
their SHA-256 against the list below (the same hashes as in the release's SHA256SUMS.txt).  Set DNEC_RELEASE to the
release folder first.
Usage: python -B setup_from_release.py
"""
import hashlib
import os
import shutil
import sys

MODULES = [
 [
  "dnec_common/__init__.py",
  "3fa54a646cefc4591582ab0915c2aa3cab16dde29650a48a17d28c4fbbfbc43e"
 ],
 [
  "dnec_common/io.py",
  "9a793d09177181edb5df13faf68ce0908d1812cea712e635965f4b1eb74b9feb"
 ],
 [
  "dnec_common/paths.py",
  "0302137cb9b33bf6328e086e5cae402a0e765351905203797e237ce0eadd9299"
 ],
 [
  "higgs/04_cross_terms/C149_coulomb.py",
  "2dc4e4e9a0678e61f56b09b114990b1614db040b8536047f45b320dbd3924e68"
 ],
 [
  "higgs/04_cross_terms/C149_direct4.py",
  "addebb35752e24908f982fe3df25adf3c339c775dd4d4455d2e6305aaa603a5f"
 ],
 [
  "higgs/04_cross_terms/C149_direct5.py",
  "d84afad13b04e9226d0e9683ebbc2ba495857ce143e693df71d906142a83a74f"
 ],
 [
  "higgs/04_cross_terms/C149_top.py",
  "09294058dd68c1ffcd126e9d96a1abab8f944aec3a712c60c33e8674b6643eb8"
 ],
 [
  "higgs/zmlib/_smdr_betas.py",
  "a5411d0a5a15ae22057f19c04f6657f37e3317dc29a0942fdf4133f00f3dbf0b"
 ],
 [
  "higgs/zmlib/_zpaths.py",
  "e820c98569ec89f8cb0a8d76a6a6c1bb57d5547219f38a983498f504e1fbec65"
 ],
 [
  "higgs/zmlib/C138_core.py",
  "947ce48e98460b449781e11625223b4586fed6f683ae8bcbef2630d351355dd1"
 ],
 [
  "higgs/zmlib/C140_beyond_bo.py",
  "7de15941f856493a7d4a6a40ee3caa8d2c205e344f0d2b7efc7c51bc6485023c"
 ],
 [
  "higgs/zmlib/C149_direct.py",
  "452700dcddc7de7b593d68c88fa66f3acf3183cb8582e6406a4982bce379daae"
 ],
 [
  "higgs/zmlib/c150_nnlo_central.py",
  "2a76947d2312e3f2aea42a43b2c20205e252d0a28fff3fff511077225f542fa4"
 ],
 [
  "higgs/zmlib/C150_twod.py",
  "9bb1372719028800ce4cb8805cfb37da66e8a287d9f130cc41f795d13e185446"
 ],
 [
  "higgs/zmlib/cross_2d.py",
  "0beb394324a489de380c8a7e823623741462850ee2d0c4ccf9ddcc8b8e4b8b2c"
 ],
 [
  "higgs/zmlib/cross_core.py",
  "4711622918de49565e4ff492f283d87134f6df88aaec5fe88594de13c5f7cf4d"
 ],
 [
  "higgs/zmlib/cross_kappa.py",
  "4778fa72626e3757a8e24304b477ec7243fb3579f7142f6638d192c38032aed4"
 ],
 [
  "higgs/zmlib/cross_var.py",
  "27c95f226b67819f209381e88521b6e8da9f37b5249edf1e1fa56bc04198566f"
 ],
 [
  "higgs/zmlib/hr_core.py",
  "845fda717f684cdf35bdfb2c076fe0328013bb27dcace84bdc16cbb91295c04b"
 ],
 [
  "higgs/zmlib/hr_part2.py",
  "622e59c1a6c629981d33dea07454aeacb59368dc85a03d399f9194ed1ae481e9"
 ],
 [
  "higgs/zmlib/hr_post.py",
  "7b276ffaab71ecc8943f0bde544f87a9c47624410375f90f1fedec075e5cadf1"
 ],
 [
  "higgs/zmlib/hr_potential.py",
  "09e09cac06748dd61cdd23ef1e7c365e545f25effd2a6a325b2a328df7b311bb"
 ],
 [
  "higgs/zmlib/PACKAGE_PATCHES.json",
  "986fa802953f9e802ac6054712c05dc2aab9c582e13bb86c4325abc09dc78b7c"
 ],
 [
  "higgs/zmlib/qcd.py",
  "5ccd826bac49f3f335a909d0729c2bbedeac37ce0830cc585f3f9bc4b694f4ba"
 ],
 [
  "higgs/zmlib/R14_fv.py",
  "18b4db54989160c70da362a3fb1ac52405d69ac2c5fde03695852244b04d9a62"
 ],
 [
  "higgs/zmlib/R14_res.py",
  "62b355e673b5f9470b762d8b0367a3b9e071dca3486110df5595a6902a23ae06"
 ],
 [
  "higgs/zmlib/R14_rge.py",
  "d909eadc9a0099b5017afef4595467766027f290d87bb4d2f0c2482674eecd12"
 ],
 [
  "higgs/zmlib/README.md",
  "2e0654a25cd3f25a5eed200d6180fbc97de528fda5fab30911da7e6e5aa3c7cd"
 ],
 [
  "higgs/zmlib/sm_provider.py",
  "03ec7b4278987722f0f0f9ed57d9726af01f927c242f987f4d582a545ca863c0"
 ],
 [
  "higgs/zmlib/sm_rge.py",
  "c2c77b7d87a878cd2cb961e38f8d447748bb0db8ee9d12aebd56d4b6af5c6003"
 ]
]
HERE = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(os.path.dirname(HERE), "c170_modules")
REL = os.environ.get("DNEC_RELEASE", "")
if not REL or not os.path.isdir(REL):
    sys.exit("set DNEC_RELEASE to the folder of the code release v1.0.0 (Zenodo 10.5281/zenodo.23434068)")
bad = 0
for rel, h in MODULES:
    src = os.path.join(REL, *rel.split("/"))
    dst = os.path.join(DST, *rel.split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    got = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    ok = got == h
    bad += not ok
    print(("ok  " if ok else "BAD ") + rel)
sys.exit(1 if bad else 0)
