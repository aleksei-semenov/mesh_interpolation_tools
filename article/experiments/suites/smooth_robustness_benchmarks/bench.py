"""Planar cell-average remapping with independent clipped-polygon references."""
import numpy as np
from scipy.integrate import quad
from scipy.special import roots_legendre
from geometry import area, clip, mesh, errors


def children(triangle, k):
    t = np.asarray(triangle, float)[None, :, :]
    for _ in range(k):
        a,b,c=t[:,0],t[:,1],t[:,2]
        ab,bc,ca=(a+b)/2,(b+c)/2,(c+a)/2
        t=np.stack([np.stack(v,axis=1) for v in [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]],axis=1).reshape(-1,3,2)
    return t


def quad_nodes(triangles, degree):
    """Centroid degree 1 or symmetric three-node degree 2; weights sum to one."""
    b = np.array([[1/3,1/3,1/3]]) if degree==1 else np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    if degree not in (1,2): raise ValueError('quadrature degree must be 1 or 2')
    return np.einsum('qv,nvd->nqd',b,triangles)


def gauss_triangle(t, fun, n=10):
    """Independent Duffy-product Gaussian integration, returns cell mean."""
    z,w=roots_legendre(n); z=(z+1)/2; w=w/2
    u,v=np.meshgrid(z,z,indexing='ij'); wu,wv=np.meshgrid(w,w,indexing='ij')
    p=t[0]+u[...,None]*(t[1]-t[0])+(1-u[...,None])*v[...,None]*(t[2]-t[0])
    return float(np.sum(2*wu*wv*(1-u)*fun(p[...,0],p[...,1])))


def field(name,x,y):
    if name=='sine': return 1+0.5*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
    if name=='quadratic': return 1+0.3*x+0.2*y+0.4*x*x-0.2*x*y+0.1*y*y
    raise ValueError(name)


def target_average(name,x,y,n=10):
    """Independent tensor Gauss rule on each rectangle."""
    z,w=roots_legendre(n); z=(z+1)/2; w=w/2
    xx=x[:-1,None]+np.diff(x)[:,None]*z; yy=y[:-1,None]+np.diff(y)[:,None]*z
    return np.array([[np.sum(w[:,None]*w[None,:]*field(name,X[:,None],Y[None,:])) for X in xx] for Y in yy]).ravel()


def moments(t):
    c=t.mean(axis=0); d=t-c
    # Uniform triangle covariance = sum(vertex offsets outer products)/12.
    cov=np.einsum('ni,nj->ij',d,d)/12
    return c,cov


def reconstruct(nodes, triangles, values, method):
    """Unweighted least-squares fit to cell averages; mean-preserving basis.
    Full two-ring vertex neighborhood, expanding if underdetermined. Fail on
    rank deficiency instead of silently falling back to a lower-order fit.
    """
    ts=nodes[triangles]; cc=np.array([moments(t)[0] for t in ts]); cv=np.array([moments(t)[1] for t in ts]); n=len(ts)
    if method=='constant': return cc,cv,np.zeros((n,5))
    if method not in ('linear','quadratic'): raise ValueError(method)
    vertex=[set() for _ in nodes]
    for i,tri in enumerate(triangles):
        for v in tri: vertex[v].add(i)
    neigh=[set().union(*(vertex[v] for v in tri))-{i} for i,tri in enumerate(triangles)]
    coef=np.zeros((n,5)); dim=2 if method=='linear' else 5
    for i in range(n):
        pool=set(neigh[i]); frontier=set(pool)
        for _ in range(3):
            if len(pool)>=max(2*dim,dim+3): break
            frontier=set().union(*(neigh[j] for j in frontier))-pool-{i} if frontier else set()
            pool |= frontier
        ids=np.array(sorted(pool),dtype=int)
        d=cc[ids]-cc[i]; h=np.sqrt(area(ts[i])); dx,dy=d[:,0],d[:,1]
        # Average of parent's centered basis over neighboring triangles.
        X=np.column_stack([dx/h,dy/h,(dx*dx+cv[ids,0,0]-cv[i,0,0])/h**2,
                           (dx*dy+cv[ids,0,1]-cv[i,0,1])/h**2,
                           (dy*dy+cv[ids,1,1]-cv[i,1,1])/h**2])[:,:dim]
        fit,_,rank,s=np.linalg.lstsq(X,values[ids]-values[i],rcond=None)
        if rank<dim or s[-1]/s[0] < 1e-12: raise ValueError(f'Rank-deficient {method} fit for triangle {i}; use a larger/well-shaped mesh')
        coef[i,:dim]=fit/np.array([h,h,h*h,h*h,h*h])[:dim]
    return cc,cv,coef


def evaluate(p,i,values,cc,cv,coef):
    d=p-cc[i]; x,y=d[...,0],d[...,1]; c=coef[i]
    return values[i]+c[0]*x+c[1]*y+c[2]*(x*x-cv[i,0,0])+c[3]*(x*y-cv[i,0,1])+c[4]*(y*y-cv[i,1,1])


def intersection_polygons(nodes,triangles,x,y):
    """Triangle/rectangle polygons via deterministic half-plane clipping."""
    nx=len(x)-1; ny=len(y)-1
    polys=[]; cover=np.zeros(nx*ny); source_cover=np.zeros(len(triangles))
    for i,t in enumerate(nodes[triangles]):
        lo=t.min(axis=0);hi=t.max(axis=0)
        x0=max(0,np.searchsorted(x,lo[0],side='right')-1); x1=min(nx-1,np.searchsorted(x,hi[0],side='left'))
        y0=max(0,np.searchsorted(y,lo[1],side='right')-1); y1=min(ny-1,np.searchsorted(y,hi[1],side='left'))
        for iy in range(y0,y1+1):
            for ix in range(x0,x1+1):
                poly=t.copy()
                for axis,edge,greater in ((0,x[ix],True),(0,x[ix+1],False),(1,y[iy],True),(1,y[iy+1],False)):
                    poly=clip(poly,axis,edge,greater)
                a=area(poly)
                if a>1e-18:
                    j=iy*nx+ix; polys.append((i,j,poly));cover[j]+=a;source_cover[i]+=a
    return polys,cover,source_cover


def overlap_integral(polys, n_target, values, rec):
    """Exact degree-2 polynomial integral over clipped polygons (floating point)."""
    total=np.zeros(n_target); cc,cv,co=rec
    for i,j,p in polys:
        for v in range(1,len(p)-1):
            tri=np.array([p[0],p[v],p[v+1]]); a=area(tri)
            if a: total[j]+=a*np.mean(evaluate(quad_nodes(tri[None,:,:],2)[0],i,values,cc,cv,co))
    return total


def deposit(nodes,triangles,x,y,k,values,rec,degree):
    """Assign each quadrature node independently to its target rectangle."""
    nx=len(x)-1; ny=len(y)-1; S=np.zeros(nx*ny); W=np.zeros(nx*ny); dropped=0.
    cc,cv,co=rec
    for i,t in enumerate(nodes[triangles]):
        child=children(t,k); pts=quad_nodes(child,degree).reshape(-1,2)
        weights=np.full(len(pts),area(t)/len(pts)); q=evaluate(pts,i,values,cc,cv,co)
        ix=np.searchsorted(x,pts[:,0],side='right')-1; iy=np.searchsorted(y,pts[:,1],side='right')-1
        ix[pts[:,0]==x[-1]]=nx-1; iy[pts[:,1]==y[-1]]=ny-1
        valid=(ix>=0)&(ix<nx)&(iy>=0)&(iy<ny); ind=iy[valid]*nx+ix[valid]
        np.add.at(S,ind,weights[valid]*q[valid]); np.add.at(W,ind,weights[valid]); dropped+=weights[~valid].sum()
    return S,W,dropped


def metrics(q,ref,a,active=None):
    if active is None: active=np.ones(len(a),bool)
    if not np.any(active) or not np.isfinite(q[active]).all(): return (np.nan,np.nan,np.nan)
    return errors(q[active],ref[active],a[active])


def disk_integral_triangle(t,c=(0.5,0.5),r=0.2,tol=1e-10):
    """Adaptive x integration of vertical intersection of triangle and disk."""
    lo=max(t[:,0].min(),c[0]-r); hi=min(t[:,0].max(),c[0]+r)
    if hi<=lo: return 0.
    def section(x):
        ys=[]
        for a,b in zip(t,np.roll(t,-1,axis=0)):
            if abs(b[0]-a[0])<1e-15: continue
            if min(a[0],b[0])-1e-14<=x<=max(a[0],b[0])+1e-14:
                ys.append(a[1]+(x-a[0])*(b[1]-a[1])/(b[0]-a[0]))
        if len(ys)<2: return 0.
        dy=np.sqrt(max(0.,r*r-(x-c[0])**2))
        return max(0.,min(max(ys),c[1]+dy)-max(min(ys),c[1]-dy))
    breaks=sorted(set([lo,hi]+[float(z) for z in t[:,0] if lo<z<hi]))
    # Explicit circle/edge crossings provide breakpoints for nonsmooth changes.
    for a,b in zip(t,np.roll(t,-1,axis=0)):
        d=b-a; v=a-np.array(c); A=np.dot(d,d); B=2*np.dot(v,d); C=np.dot(v,v)-r*r; disc=B*B-4*A*C
        if disc>=0:
            for s in ((-B-np.sqrt(disc))/(2*A),(-B+np.sqrt(disc))/(2*A)):
                z=a[0]+s*d[0]
                if 0<s<1 and lo<z<hi: breaks.append(float(z))
    breaks=sorted(set(breaks)); return sum(quad(section,a,b,epsabs=tol,epsrel=tol,limit=100)[0] for a,b in zip(breaks[:-1],breaks[1:]))


def disk_rectangle(x0,x1,y0,y1,c=(.5,.5),r=.2,tol=1e-10):
    lo=max(x0,c[0]-r); hi=min(x1,c[0]+r)
    if hi<=lo: return 0.
    def f(x):
        d=np.sqrt(max(0.,r*r-(x-c[0])**2)); return max(0.,min(y1,c[1]+d)-max(y0,c[1]-d))
    breaks=[lo,hi]
    for yy in (y0,y1):
        d2=r*r-(yy-c[1])**2
        if d2>=0:
            for xx in (c[0]-np.sqrt(d2),c[0]+np.sqrt(d2)):
                if lo<xx<hi: breaks.append(xx)
    return sum(quad(f,a,b,epsabs=tol,epsrel=tol,limit=100)[0] for a,b in zip(sorted(set(breaks))[:-1],sorted(set(breaks))[1:]))
