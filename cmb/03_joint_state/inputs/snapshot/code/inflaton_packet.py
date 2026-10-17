"""Constrained semiclassical propagation of an explicitly prepared inflaton packet.

This is a positive Wigner sampling / Gaussian quadrature calculation, not a
solution of the joint Wheeler-DeWitt equation or a derivation of its measure.
The reference Gaussian width is an assumption (an inverted oscillator has no
normalizable instantaneous ground state). No source-project module is imported.
"""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import json, time, math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from numpy.polynomial.hermite import hermgauss

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'results'/'P2_inflaton'; OUT.mkdir(parents=True,exist_ok=True)
H=6.1159966044e-6
D=29.92
A=math.sqrt(2/3)

def U(phi): return 3*(1-np.exp(-A*phi))**2
def Up(phi):
    e=np.exp(-A*phi); return 6*A*e*(1-e)
def Upp(phi):
    e=np.exp(-A*phi); return 4*e*(2*e-1)

def trajectory(phi, momentum, refphi, rtol=2e-10):
    ur=U(refphi)
    lr=(D/(H*H*ur))**.25
    # Turning constraint: U(phi) w^3 - U(refphi) w + P^2/(2 h^2 Lref^6)=0,
    # w=(L/Lref)^2. The largest positive root is the expanding bounce.
    pp=momentum/(H*lr**3)
    roots=np.roots([U(phi),0,-ur,.5*pp**2])
    positive=[r.real for r in roots if abs(r.imag)<1e-9 and r.real>0]
    if not positive: raise ValueError('No positive turning volume')
    w=max(positive); n0=.5*math.log(w); v0=pp/w**1.5
    if v0*v0>=2*U(phi): raise ValueError('Turning point is not a bounce')
    def rhs(t,y):
        n,hhat,p,v=y
        c=ur*np.exp(-4*n)
        return [hhat, -.5*v*v+2*c/3, v, -3*hhat*v-Up(p)]
    def end(t,y):
        n,hh,p,v=y
        return .5*v*v-2*ur*np.exp(-4*n)/3-hh*hh
    end.terminal=True; end.direction=1
    sol=solve_ivp(rhs,[0,500],[n0,0,phi,v0],method='DOP853',rtol=rtol,
                  atol=rtol/100,events=end,dense_output=True,max_step=1.0)
    if not sol.success or not sol.t_events[0].size:
        raise RuntimeError('Inflation end absent: '+sol.message)
    tend=float(sol.t_events[0][0]); final=sol.sol(tend)
    y=sol.sol(np.linspace(0,tend,400))
    constraint=3*y[1]**2-.5*y[3]**2-U(y[2])+ur*np.exp(-4*y[0])
    rhoend=3*final[1]**2
    # At fixed h, fixed g*, and instantaneous reheating: log L0 =
    # log Lend + (1/4)log rhoend + constants. Includes correlated turning volume.
    logsize=float(final[0]+.25*np.log(rhoend))
    return {'Npost':float(final[0]-n0),'logsize_offset':logsize,
            'log_turn_volume_ratio':n0,'initial_v':float(v0),'rhoend_hat':float(rhoend),
            'constraint_max_abs':float(np.max(np.abs(constraint))),
            'initial_constraint_abs':float(abs(.5*v0*v0+U(phi)-ur/w**2)),
            't_end_in_h_units':tend,'evaluations':sol.nfev}

def packet(refphi,order=9,widthfactor=1.,rtol=2e-10):
    lr=(D/(H*H*U(refphi)))**.25
    omega=math.sqrt(abs(Upp(refphi)))
    sigma=widthfactor/math.sqrt(2*H*lr**3*omega)
    sigmap=1/(2*sigma)
    points,weights=hermgauss(order); weights=weights/math.sqrt(math.pi)
    rows=[]; wgt=[]
    center=trajectory(refphi,0,refphi,rtol)
    for i,x in enumerate(points):
        for j,z in enumerate(points):
            ph=refphi+math.sqrt(2)*sigma*x
            p=math.sqrt(2)*sigmap*z
            r=trajectory(ph,p,refphi,rtol)
            rows.append([ph,p,r['Npost'],r['logsize_offset']-center['logsize_offset'],
                         r['log_turn_volume_ratio'],r['constraint_max_abs']])
            wgt.append(weights[i]*weights[j])
    data=np.array(rows); ws=np.array(wgt)
    avg=ws@data; cov=(data-avg).T@(ws[:,None]*(data-avg))
    step=1e-4
    dnphi=(trajectory(refphi+step,0,refphi,rtol)['Npost']-
           trajectory(refphi-step,0,refphi,rtol)['Npost'])/(2*step)
    dp=.01
    dnp=(trajectory(refphi,dp,refphi,rtol)['Npost']-
         trajectory(refphi,-dp,refphi,rtol)['Npost'])/(2*dp)
    return {'refphi':refphi,'order':order,'widthfactor':widthfactor,'Lref_MP':lr,
            'hLb':H*lr,'delta':math.sqrt(2)/(3*H*lr**3),
            'Upp_at_center':Upp(refphi),'sigma_phi':sigma,'sigma_P':sigmap,
            'center':center,'means':avg.tolist(),'covariance':cov.tolist(),
            'sigma_N':math.sqrt(cov[2,2]),'sigma_logL0':math.sqrt(cov[3,3]),
            'dN_dphi':dnphi,'dN_dP':dnp,
            'sigmaN_linear_with_momentum':math.hypot(dnphi*sigma,dnp*sigmap),
            'width_to_classical_drift_ratio':sigma/(Up(refphi)/3),
            'constraint_max_abs':float(np.max(data[:,-1])),
            'points':data.tolist(),'weights':ws.tolist()}

def main():
    started=time.perf_counter()
    ref=brentq(lambda phi: trajectory(phi,0,phi)['Npost']-68.25,5.3,5.9,xtol=1e-11)
    cases={}
    specifications=[('reference_9',ref,9,1.,2e-10),
                    ('reference_13',ref,13,1.,2e-10),
                    ('reference_tight',ref,9,1.,2e-12),
                    ('half_width',ref,9,.5,2e-10),
                    ('double_width',ref,9,2.,2e-10),
                    ('larger_N',ref+.32,9,1.,2e-10),
                    ('smaller_N',ref-.20,9,1.,2e-10)]
    for label,p,n,f,rtol in specifications:
        r=packet(p,n,f,rtol); cases[label]=r
        (OUT/(label+'.json')).write_text(json.dumps(r,indent=2),encoding='utf-8')
        (OUT/'STATUS.json').write_text(json.dumps({'stage':'running','completed':list(cases),
                 'pid':os.getpid(),'elapsed_seconds':time.perf_counter()-started},indent=2))
        print(json.dumps({'label':label,'Ncenter':r['center']['Npost'],
               'sigmaN':r['sigma_N'],'sigmaLogL0':r['sigma_logL0'],
               'width_drift_ratio':r['width_to_classical_drift_ratio'],
               'constraint':r['constraint_max_abs']}),flush=True)
    reduced={k:{kk:vv for kk,vv in v.items() if kk not in ('points','weights')} for k,v in cases.items()}
    r9=cases['reference_9'];r13=cases['reference_13'];rt=cases['reference_tight']
    reduced['verification']={'quadrature_sigmaN_change':r13['sigma_N']-r9['sigma_N'],
        'quadrature_meanN_change':r13['means'][2]-r9['means'][2],
        'tight_solver_sigmaN_change':rt['sigma_N']-r9['sigma_N'],
        'scope':'Prepared Gaussian Wigner distribution, constraint projected at H=0, classical nonlinear propagation. No unique state origin, quantum gravity, decoherence proof or primordial inhomogeneous scalar quantization.'}
    (OUT/'summary.json').write_text(json.dumps(reduced,indent=2),encoding='utf-8')
    (OUT/'STATUS.json').write_text(json.dumps({'stage':'complete','pid':os.getpid(),
               'elapsed_seconds':time.perf_counter()-started,'cases':list(cases)},indent=2))

if __name__=='__main__':main()
