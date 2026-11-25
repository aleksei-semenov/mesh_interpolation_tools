"""Run smooth-field, disk, and robustness experiments and save reproducible data."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from geometry import mesh,area,validate
from bench import (field,gauss_triangle,target_average,reconstruct,intersection_polygons,
                   overlap_integral,deposit,metrics,disk_integral_triangle,disk_rectangle)


def write_csv(path,rows):
    if not rows: raise ValueError('No rows to write')
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def target_edges(n,shift=False):
    if not shift: return np.linspace(0,1,n+1)
    # Shift interior boundaries by half a nominal cell, keep exterior fixed.
    return np.r_[0.,(np.arange(1,n)+.5)/n,1.]

def geometry(ns,nt,seed=1729,jitter=.17,shift=False,mask=False):
    nodes,tri=mesh(ns,jitter=jitter,seed=seed)
    if mask:
        c=nodes[tri].mean(axis=1); tri=tri[~((abs(c[:,0]-.5)<.13)&(abs(c[:,1]-.5)<.13))]
    x=target_edges(nt,shift);y=target_edges(nt,shift); source=validate(nodes,tri,x,y)
    A=np.outer(np.diff(y),np.diff(x)).ravel()
    polys,cover,parent=intersection_polygons(nodes,tri,x,y)
    if not np.allclose(parent,source,rtol=1e-9,atol=1e-12): raise ValueError('Source is not fully within target domain')
    if not mask and not np.allclose(cover,A,rtol=1e-9,atol=1e-12): raise ValueError('Source and target coverage disagree')
    if np.any(cover>A+1e-10): raise ValueError('Source mesh overlaps itself')
    return nodes,tri,x,y,source,A,polys,cover

def norm(S,W,A,method):
    return S/A if method=='A' else np.divide(S,W,out=np.full_like(S,np.nan),where=W>0)

def smooth_group(ns,nt,levels,fieldname,seed,quad_order,sweep):
    nodes,tri,x,y,src,A,polys,cover=geometry(ns,nt,seed)
    fun=lambda X,Y:field(fieldname,X,Y)
    values=np.array([gauss_triangle(t,fun,quad_order) for t in nodes[tri]])
    check=np.array([gauss_triangle(t,fun,quad_order+4) for t in nodes[tri]])
    delta=float(np.max(abs(values-check)))
    if delta>1e-10: raise ValueError(f'Source quadrature verification failed: {delta}; increase --quad-order')
    target=target_average(fieldname,x,y,quad_order)
    if np.max(abs(target-target_average(fieldname,x,y,quad_order+4)))>1e-10: raise ValueError('Target quadrature verification failed; increase --quad-order')
    rows=[]
    for recname in ('constant','linear','quadratic'):
        rec=reconstruct(nodes,tri,values,recname); exact=overlap_integral(polys,len(A),values,rec)/A
        degree=2 if recname=='quadratic' else 1
        for k in levels:
            begin=time.perf_counter();S,W,dropped=deposit(nodes,tri,x,y,k,values,rec,degree);elapsed=time.perf_counter()-begin
            missing=A[W==0].sum()/A.sum(); mass=np.dot(values,src);B=np.dot(abs(values),src)
            for method in ('W','A'):
                q=norm(S,W,A,method); e=metrics(q,target,A); ge=metrics(q,exact,A); rep=metrics(exact,target,A)
                rows.append(dict(sweep=sweep,field=fieldname,source_n=ns,target_n=nt,Ns=len(tri),Nt=len(A),hS=np.sqrt(1/len(tri)),hT=np.sqrt(1/len(A)),
                     reconstruction=recname,normalization=method,quadrature_degree=degree,k=k,samples=len(tri)*4**k*(3 if degree==2 else 1),
                     E1=e[0],E2=e[1],Einf=e[2],geometric_E2=ge[1],representation_E2=rep[1],missing_fraction=missing,
                     physical_residual=(np.dot(q,A)-mass)/B if np.isfinite(q).all() else np.nan,
                     sampled_residual=(np.dot(q[W>0],W[W>0])-mass)/B,
                     dropped_area=dropped,deposit_seconds=elapsed,source_quadrature_check=delta,pk=np.nan))
    # Adjacent total-error rates only within a fixed mesh pair/method/field.
    for recname in ('constant','linear','quadratic'):
        for method in ('W','A'):
            rr=[r for r in rows if r['reconstruction']==recname and r['normalization']==method]
            for a,b in zip(rr[:-1],rr[1:]):
                if a['missing_fraction']==0 and b['missing_fraction']==0 and a['E2']>0 and b['E2']>0 and np.isfinite(a['E2']) and np.isfinite(b['E2']):
                    b['pk']=np.log(a['E2']/b['E2'])/((b['k']-a['k'])*np.log(2))
    return rows

def disk_case(ns,nt,k,seed,out,tol):
    nodes,tri,x,y,src,A,polys,cover=geometry(ns,nt,seed)
    v=np.array([disk_integral_triangle(t,tol=tol)/a for t,a in zip(nodes[tri],src)])
    ref=np.array([disk_rectangle(x[i],x[i+1],y[j],y[j+1],tol=tol)/A[j*nt+i] for j in range(nt) for i in range(nt)])
    rec=reconstruct(nodes,tri,v,'constant'); over=overlap_integral(polys,len(A),v,rec)/A
    S,W,drop=deposit(nodes,tri,x,y,k,v,rec,1); q=norm(S,W,A,'A'); geo=metrics(q,over,A); total=metrics(q,ref,A)
    negative=np.dot(A,np.maximum(-q,0)); above=np.dot(A,np.maximum(q-1,0))
    np.savez_compressed(out/'disk_fields.npz',x=x,y=y,reference=ref.reshape(nt,nt),output=q.reshape(nt,nt),error=(q-ref).reshape(nt,nt),
                        overlap=over.reshape(nt,nt),W=W.reshape(nt,nt))
    return dict(source_n=ns,target_n=nt,k=k,method='constant',normalization='A',quadrature_nodes=len(tri)*4**k,
                disk_source_integral=float(np.dot(v,src)),reference_disk_integral=float(np.dot(ref,A)),
                analytic_disk_integral=float(np.pi*.2**2),disk_integral_tolerance=tol,minimum=float(q.min()),maximum=float(q.max()),
                undershoot=float(max(0,-q.min())),overshoot=float(max(0,q.max()-1)),integrated_negative_part=negative,
                integrated_above_one=above,bound_count_tolerance=1e-12,area_fraction_below_zero=float(A[q < -1e-12].sum()/A.sum()),area_fraction_above_one=float(A[q > 1+1e-12].sum()/A.sum()),physical_mass_residual=float((np.dot(q,A)-np.dot(v,src))/np.dot(abs(v),src)),L1=total[0],L2=total[1],Linf=total[2],geometric_E2=geo[1],missing_fraction=float(A[W==0].sum()/A.sum()),dropped_area=drop)

def robustness_case(case,ns,nt,levels,seed):
    if case=='baseline': sn,tn,jit,shift,mask=ns,nt,.17,False,False
    elif case=='coarse_to_fine': sn,tn,jit,shift,mask=max(3,ns//2),2*nt,.17,False,False
    elif case=='fine_to_coarse': sn,tn,jit,shift,mask=2*ns,max(2,nt//2),.17,False,False
    elif case=='shifted': sn,tn,jit,shift,mask=ns,nt,.17,True,False
    elif case=='distorted': sn,tn,jit,shift,mask=ns,nt,.24,False,False
    elif case=='mask': sn,tn,jit,shift,mask=ns,nt,.17,False,True
    else: raise ValueError(case)
    nodes,tri,x,y,src,A,polys,cover=geometry(sn,tn,seed,jit,shift,mask)
    c=nodes[tri].mean(axis=1);v=field('sine',c[:,0],c[:,1]);rec=reconstruct(nodes,tri,v,'constant')
    ref=overlap_integral(polys,len(A),v,rec)/A;active=cover>1e-14
    rows=[]
    for k in levels:
        S,W,drop=deposit(nodes,tri,x,y,k,v,rec,1);q=S/A
        err=metrics(q,ref,A,active)
        # Valid physical source-covered area, not full rectangular cell area.
        f0=cover[(W==0)&active].sum()/cover[active].sum()
        rows.append(dict(case=case,k=k,source_n=sn,target_n=tn,mask=mask,normalization='A',
                         geometric_E2=err[1],missing_valid_area_fraction=f0,unassigned_area=drop,
                         active_area_fraction=cover.sum()/A.sum()))
    return rows

def latex_table(rows,out):
    def fmt(x): return '--' if not np.isfinite(x) else f'{x:.3e}'
    for fieldname in sorted({r['field'] for r in rows}):
        data=[r for r in rows if r['sweep']=='subdivision' and r['field']==fieldname and r['missing_fraction']==0]
        lines=[r'\begin{tabular}{llllllllll}',r'\toprule',r'Field & $n_S/n_T$ & Recon. & Norm. & $k$ & $E_1$ & $E_2$ & $E_\infty$ & $p_k$ & $f_0$ \\',r'\midrule']
        for r in data:
            lines.append(f"{r['field']} & {r['source_n']}/{r['target_n']} & {r['reconstruction']} & {r['normalization']} & {r['k']} & "+' & '.join(fmt(r[q]) for q in ('E1','E2','Einf','pk','missing_fraction'))+r' \\')
        if not data: lines.append(r'\multicolumn{10}{c}{No fully covered runs} \\')
        lines += [r'\bottomrule',r'\end{tabular}'];(out/f'convergence_{fieldname}.tex').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-n',type=int,default=8);p.add_argument('--target-n',type=int,default=11)
    p.add_argument('--source-sweep',nargs='+',type=int,default=[5,8,12]);p.add_argument('--target-sweep',nargs='+',type=int,default=[6,11,16])
    p.add_argument('--k',nargs='+',type=int,default=[0,1,2,3,4]);p.add_argument('--refine-k',type=int,default=4)
    p.add_argument('--fields',nargs='+',choices=['sine','quadratic'],default=['sine','quadratic'])
    p.add_argument('--disk-k',type=int,default=3);p.add_argument('--disk-tol',type=float,default=1e-10);p.add_argument('--quad-order',type=int,default=10)
    p.add_argument('--seed',type=int,default=1729);p.add_argument('--out',default='results')
    a=p.parse_args();levels=sorted(set(a.k))
    if min([a.source_n,*a.source_sweep])<3 or min([a.target_n,*a.target_sweep])<1 or min(levels+[a.refine_k,a.disk_k])<0 or a.quad_order<4 or a.disk_tol<=0: p.error('Invalid mesh sizes, subdivision depths, quadrature order or tolerance')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[]
    for name in a.fields:
        print('smooth',name,'subdivision',flush=True);rows += smooth_group(a.source_n,a.target_n,levels,name,a.seed,a.quad_order,'subdivision')
        for n in sorted(set(a.source_sweep)):
            print('smooth',name,'source',n,flush=True);rows+=smooth_group(n,a.target_n,[a.refine_k],name,a.seed,a.quad_order,'source')
        for n in sorted(set(a.target_sweep)):
            print('smooth',name,'target',n,flush=True);rows+=smooth_group(a.source_n,n,[a.refine_k],name,a.seed,a.quad_order,'target')
    write_csv(out/'smooth_diagnostics.csv',rows);latex_table(rows,out)
    disk=disk_case(a.source_n,a.target_n,a.disk_k,a.seed,out,a.disk_tol); (out/'disk_diagnostics.json').write_text(json.dumps(disk,indent=2))
    print('disk',json.dumps(disk),flush=True)
    rr=[]
    for case in ('baseline','coarse_to_fine','fine_to_coarse','shifted','distorted','mask'):
        print('robustness',case,flush=True);rr+=robustness_case(case,a.source_n,a.target_n,levels,a.seed)
    write_csv(out/'robustness_diagnostics.csv',rr)
    (out/'metadata.json').write_text(json.dumps(vars(a),indent=2)); print('Saved diagnostics; NaN means unavailable.')

if __name__=='__main__':main()
