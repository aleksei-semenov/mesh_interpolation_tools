"""Plot saved smooth-refinement and disk/coverage diagnostics."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter


def load(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))

def num(row,key):return float(row[key])

def curve(ax,x,y,color,style,label=None,mark='o',status=0):
    x=np.array(x,float); y=np.array(y,float); ok=np.isfinite(y)&(y>0)
    # NaNs preserve gaps. Exact zeros/NA shown in an axes-relative strip: not fake log values.
    ax.plot(x,np.where(ok,y,np.nan),color=color,linestyle=style,marker=mark,markersize=3,label=label)
    trans=ax.get_xaxis_transform(); z=np.isfinite(y)&(y==0);na=~np.isfinite(y)
    if z.any(): ax.scatter(x[z],np.full(z.sum(),-.10-.018*(status%3)),marker='v',color=color,s=13,transform=trans,clip_on=False)
    if na.any(): ax.scatter(x[na],np.full(na.sum(),-.10-.018*(status%3)),marker='x',color=color,s=13,transform=trans,clip_on=False)

def smooth_plot(out,field,fit_k=None):
    rows=load(out/'smooth_diagnostics.csv');rows=[r for r in rows if r['field']==field]
    if not rows:raise ValueError('field not present in diagnostics')
    fig,ax=plt.subplots(1,3,figsize=(16,5.5)); colors={'constant':'#0072B2','linear':'#D55E00','quadratic':'#009E73'}
    styles={'geometric_E2':'-','representation_E2':':','E2':'--'}
    for i,rec in enumerate(colors):
        sub=[r for r in rows if r['sweep']=='subdivision' and r['reconstruction']==rec]
        for method in ('W','A'):
            rr=sorted([r for r in sub if r['normalization']==method],key=lambda r:int(r['k']))
            for key in ('geometric_E2','E2'):
                curve(ax[0],[num(r,'k') for r in rr],[num(r,key) for r in rr],colors[rec],styles[key],mark='o' if method=='W' else 's',status=i)
        rr=sorted([r for r in sub if r['normalization']=='A'],key=lambda r:int(r['k']))
        curve(ax[0],[num(r,'k') for r in rr],[num(r,'representation_E2') for r in rr],colors[rec],':',mark='',status=i)
        for method in ('W','A'):
            for panel,sweep,xkey,ykey in [(1,'source','hS','E2'),(2,'target','hT','geometric_E2')]:
                rr=sorted([r for r in rows if r['sweep']==sweep and r['reconstruction']==rec and r['normalization']==method],key=lambda r:num(r,xkey))
                curve(ax[panel],[num(r,xkey) for r in rr],[num(r,ykey) for r in rr],colors[rec],'-' if method=='W' else '--',mark='o' if method=='W' else 's',status=i)
                # Flag source points where geometry is not small relative to total error.
                if panel==1:
                    flagged=[r for r in rr if np.isfinite(num(r,'geometric_E2')) and np.isfinite(num(r,'E2')) and num(r,'E2')>0 and num(r,'geometric_E2')>=.1*num(r,'E2')]
                    if flagged: ax[1].scatter([num(r,xkey) for r in flagged],[num(r,ykey) for r in flagged],facecolors='none',edgecolors='black',s=70,linewidths=.8,zorder=5)
    for a in ax: a.set_yscale('log');a.grid(True,which='both',alpha=.22)
    for panel,sweep,xkey in [(1,'source','hS'),(2,'target','hT')]:
        a=ax[panel];a.set_xscale('log')
        ticks=sorted({num(r,xkey) for r in rows if r['sweep']==sweep})
        a.xaxis.set_major_locator(FixedLocator(ticks));a.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        a.xaxis.set_minor_formatter(NullFormatter())
    if fit_k is not None:
        left,right=fit_k
        if left>=right: raise ValueError('--fit-k requires start < end')
        ax[0].axvspan(left,right,color='gray',alpha=.10,label='fit interval')
        fits=[]
        for rec in colors:
            for method in ('W','A'):
                rr=sorted([r for r in rows if r['sweep']=='subdivision' and r['reconstruction']==rec and r['normalization']==method and left<=int(r['k'])<=right],key=lambda r:int(r['k']))
                rr=[r for r in rr if float(r['missing_fraction'])==0 and np.isfinite(float(r['geometric_E2'])) and float(r['geometric_E2'])>0]
                # Require 3 observed valid depths and retain the actual used interval.
                slope=float(np.polyfit([-int(r['k'])*np.log(2) for r in rr],[np.log(float(r['geometric_E2'])) for r in rr],1)[0]) if len(rr)>=3 else np.nan
                fits.append(dict(field=field,reconstruction=rec,normalization=method,requested_k_start=left,requested_k_end=right,used_depths=';'.join(r['k'] for r in rr),n=len(rr),fitted_geometric_E2_order=slope))
        with open(out/f'fitted_rates_{field}.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(fits[0]));w.writeheader();w.writerows(fits)
        ax[0].text(.02,.02,f'Fit range k={left}–{right}; rates in CSV',transform=ax[0].transAxes,fontsize=8,va='bottom')
    ax[0].set(xlabel='Subdivision depth $k$',ylabel='Area-weighted $E_2$',title='Fixed meshes: error components')
    ax[1].set(xlabel=r'Source size $h_S=\sqrt{1/N_S}$',ylabel=r'Total $E_2$',title='Source refinement')
    ax[2].set(xlabel=r'Target size $h_T=\sqrt{1/N_T}$',ylabel=r'Geometric $E_2$',title='Target refinement')
    handles=[Line2D([0],[0],color=colors[r],label=r) for r in colors]+[Line2D([0],[0],color='black',ls=styles[k],label=lab) for k,lab in [('geometric_E2','geometry'),('representation_E2','representation'),('E2','total')]]+[Line2D([0],[0],color='black',marker='o',ls='None',label='W sampled'),Line2D([0],[0],color='black',marker='s',ls='None',label='A physical')]
    fig.legend(handles=handles,loc='lower center',ncol=8,fontsize=8,bbox_to_anchor=(.5,.02))
    fig.suptitle(f'Smooth field: {field} | triangle centroid or degree-two quadrature')
    fig.text(.5,.075,'▼ exact zero, × unavailable (status strips); circled source point: geometric error ≥ 10% of total',ha='center',fontsize=8)
    fig.subplots_adjust(left=.06,right=.99,top=.87,bottom=.28,wspace=.32)
    for ext in ('pdf','png'):fig.savefig(out/f'smooth_refinement_{field}.{ext}',dpi=220)
    plt.close(fig)

def robust_plot(out):
    z=np.load(out/'disk_fields.npz');x=z['x'];y=z['y'];ref=z['reference'];q=z['output'];e=z['error'];nt=ref.shape[1]
    rows=load(out/'robustness_diagnostics.csv');cases=list(dict.fromkeys(r['case'] for r in rows));ks=sorted(set(int(r['k']) for r in rows))
    fig,axs=plt.subplots(2,3,figsize=(15,9));limits=max(float(np.max(abs(e))),1e-16)
    vmax=max(1.,float(np.max(q)));vmin=min(0.,float(np.min(q)))
    for ax,v,title in [(axs[0,0],ref,'Disk: target cell-average reference'),(axs[0,1],q,'Disk: output (physical normalization)'),(axs[0,2],e,'Disk: signed error')]:
        im=ax.pcolormesh(x,y,v,shading='flat',cmap='RdBu_r' if v is e else 'viridis',vmin=-limits if v is e else vmin,vmax=limits if v is e else vmax)
        fig.colorbar(im,ax=ax,shrink=.82);ax.set(xlabel='x',ylabel='y',title=title,aspect='equal')
    mid=int(np.argmin(abs((y[:-1]+y[1:])/2-.5)));center=(y[mid]+y[mid+1])/2
    axs[1,0].stairs(ref[mid],x,label='reference',linewidth=1.7);axs[1,0].stairs(q[mid],x,label='output',linewidth=1.2)
    axs[1,0].set(xlabel='x (cell edges)',ylabel='Cell-average value',title=f'Disk profile, target row centered at y={center:.3f}');axs[1,0].legend();axs[1,0].grid(alpha=.2)
    for ax,key,title in [(axs[1,1],'geometric_E2','Fixed-field geometric $E_2$ (A)'),(axs[1,2],'missing_valid_area_fraction','Missing valid-area fraction')]:
        values=np.full((len(cases),len(ks)),np.nan)
        for r in rows:values[cases.index(r['case']),ks.index(int(r['k']))]=float(r[key])
        positive=values[np.isfinite(values)&(values>0)]
        if len(positive):
            vmin=positive.min();vmax=positive.max(); vmax=vmax if vmax>vmin else vmin*10
            im=ax.imshow(np.ma.masked_where(~np.isfinite(values)|(values<=0),values),aspect='auto',cmap='viridis',norm=LogNorm(vmin=vmin,vmax=vmax))
            fig.colorbar(im,ax=ax,shrink=.8,label='positive values (log scale)')
        for i in range(len(cases)):
            for j in range(len(ks)):
                if values[i,j]==0: ax.text(j,i,'0',ha='center',va='center',fontsize=8)
                elif not np.isfinite(values[i,j]):ax.text(j,i,'×',ha='center',va='center',fontsize=8)
        ax.set_xticks(range(len(ks)),ks);ax.set_yticks(range(len(cases)),[s.replace('_to_', ' → ').replace('_',' ') for s in cases],fontsize=8)
        ax.set(xlabel='Subdivision depth k',title=title);ax.set_facecolor('#eeeeee')
    fig.suptitle('Disk discontinuity and mesh/coverage robustness');fig.tight_layout(rect=(0,0,1,.96))
    for ext in ('pdf','png'):fig.savefig(out/f'robustness.{ext}',dpi=220)
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default='results');p.add_argument('--field',choices=['sine','quadratic','all'],default='all');p.add_argument('--fit-k',nargs=2,type=int,metavar=('START','END'));a=p.parse_args();out=Path(a.out)
    fields=sorted(set(r['field'] for r in load(out/'smooth_diagnostics.csv'))) if a.field=='all' else [a.field]
    for f in fields:smooth_plot(out,f,a.fit_k)
    robust_plot(out);print('Saved PDF and PNG figures.')
if __name__=='__main__':main()
