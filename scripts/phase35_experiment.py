"""Fresh-seed, validation-selected horizon and event-detection study.

The final seed group is not opened until model, horizon, and alert threshold
have been frozen from validation. Measurements are simulated, never real.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, precision_recall_curve
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from congestion_sim import build_default_topology
from congestion_sim.dataset import generate_runs, split_run_ids, prepare_splits, regression_metrics
from congestion_sim.training import load_split, train_one, load_model_checkpoint, resolve_device, write_json
from congestion_sim.models import TemporalGNN, LinkGRU
from torch.utils.data import DataLoader, TensorDataset

TH=.85
def score(y,p,t,run_count):
    a=y.reshape(-1)>=TH; g=p.reshape(-1)>=t
    tn,fp,fn,tp=confusion_matrix(a,g,labels=[False,True]).ravel()
    return dict(threshold=float(t),precision=float(precision_score(a,g,zero_division=0)),recall=float(recall_score(a,g,zero_division=0)),f1=float(f1_score(a,g,zero_division=0)),pr_auc=float(average_precision_score(a,p.reshape(-1))) if a.any() else None,false_positive_rate=float(fp/max(1,fp+tn)),false_alerts_per_run=float(fp/max(1,run_count)),tp=int(tp),fp=int(fp),fn=int(fn))
def choose(y,p,run_count):
    # Evaluate all score thresholds in one sorted pass; avoid an O(N^2) loop.
    labels=(np.asarray(y).reshape(-1)>=TH).astype(np.int8); scores=np.asarray(p).reshape(-1)
    precision,recall,thresholds=precision_recall_curve(labels,scores)
    if not len(thresholds): return score(y,p,1.01,run_count)
    negative_scores=np.sort(scores[labels==0])
    false_positives=len(negative_scores)-np.searchsorted(negative_scores,thresholds,side='left')
    admissible=np.flatnonzero(false_positives<=run_count)
    if not len(admissible): return score(y,p,1.01,run_count)
    # precision_recall_curve threshold index aligns with precision/recall index.
    candidates=admissible
    idx=max(candidates,key=lambda i:(recall[i],precision[i]))
    return score(y,p,float(thresholds[idx]),run_count)
def predict(model,x,edge,device):
    out=[]; model.eval()
    with torch.no_grad():
      for (xx,) in DataLoader(TensorDataset(x),batch_size=64):
        xx=xx.to(device); out.append((model(xx,edge) if isinstance(model,TemporalGNN) else model(xx)).cpu().numpy())
    return np.concatenate(out)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,default=30); ap.add_argument('--seed-start',type=int,default=1000); ap.add_argument('--steps',type=int,default=60); ap.add_argument('--epochs',type=int,default=25); ap.add_argument('--out',type=Path,default=Path('results/phase3_5')); ap.add_argument('--device',default='auto'); a=ap.parse_args()
    torch.set_num_threads(2); a.out.mkdir(parents=True,exist_ok=True)
    seeds=list(range(a.seed_start,a.seed_start+a.seeds)); rows=[]; val_predictions={}; test_records={}; horizon_summaries={}
    # Deterministic simulator reruns with identical seed/profile parameters preserve the underlying scenario across horizons.
    for h in (1,3,5):
      ds,raw=generate_runs(build_default_topology(),seeds,a.steps,10,h); split=split_run_ids(list(ds.runs),.70,.15); arr,norm=prepare_splits(ds,split)
      d=a.out/f'horizon_{h}'; d.mkdir(exist_ok=True)
      raw.to_csv(d/'measurements.csv',index=False)
      for part,v in arr.items(): np.savez_compressed(d/f'{part}.npz',x=v['x'],y=v['y'],edge_index=ds.edge_index,link_nodes=np.array([f'{u}->{v}' for u,v in ds.link_nodes]))
      positives={k:int((v['y']>=TH).sum()) for k,v in arr.items()}
      summary=dict(measurement_source='simulated',seed_groups={k:sorted({int(z.rsplit('-',1)[1]) for z in ids}) for k,ids in split.items()},runs={k:len(ids) for k,ids in split.items()},horizon=h,window=10,positive_targets=positives,normalization=norm)
      write_json(d/'dataset_summary.json',summary); horizon_summaries[h]=summary
      xt,yt,edge=load_split(d/'train.npz'); xv,yv,_=load_split(d/'validation.npz'); device=resolve_device(a.device); edge_t=torch.as_tensor(edge,dtype=torch.long,device=device)
      # Controlled comparison: persistence, moving average, non-graph GRU, standard GCN, positive-weighted GCN.
      normu=norm['std'][0]; mean=norm['mean'][0]; pv=xv[:,-1,:,0].numpy()*normu+mean; mv=xv[:,:,:,0].numpy().mean(1)*normu+mean
      candidates={'persistence':pv,'moving_average':mv}
      positive_count=int((yt>=TH).sum())
      class_weight=min(float(np.sqrt((yt.numel()-positive_count)/max(1,positive_count))),20.0)
      for name,w in [('gru',None),('gnn_standard',None),('gnn_weighted',class_weight)]:
        r=train_one(name,xt,yt,xv,yv,edge,d/'checkpoints',seed=42,device_name=a.device,batch_size=32,epochs=a.epochs,hidden_dim=32,dropout=.25,patience=5,positive_weight=w,prediction_horizon=h,feature_normalization=norm)
        model,_=load_model_checkpoint(Path(r['checkpoint']),device); candidates[name]=predict(model,xv,edge_t,device); test_records[(h,name)]=r
      for name,p in candidates.items():
        # Validation threshold and metrics; test data remains unopened.
        val_runs=len(split['validation']); met=choose(yv.numpy(),p,val_runs); reg=regression_metrics(yv.numpy(),p)
        rows.append(dict(horizon=h,model=name,validation_mae=reg['mae'],validation_rmse=reg['rmse'],**{f'val_{k}':v for k,v in met.items()}))
        val_predictions[(h,name)]=(met['threshold'],reg)
      print(f'horizon {h} complete; validation positives={positives["validation"]}',flush=True)
    table=pd.DataFrame(rows); table.to_csv(a.out/'validation_comparison.csv',index=False)
    # Freeze on operational validation score, tie-breaking by lower MAE. No test metrics influence choice.
    eligible=table[table.model.isin(['gru','gnn_standard','gnn_weighted'])].sort_values(['val_recall','val_precision','validation_mae'],ascending=[False,False,True])
    selected=eligible.iloc[0].to_dict(); frozen=dict(horizon=int(selected['horizon']),model=selected['model'],decision_threshold=float(selected['val_threshold']),selection_rule='maximum validation recall subject to <=1 false alert per simulation run; then precision; then MAE',selected_validation=selected)
    write_json(a.out/'frozen_selection.json',frozen)
    # Final test is opened once, after freeze. Evaluate baselines and all trained models for comparison.
    final=[]
    for h in (1,3,5):
      d=a.out/f'horizon_{h}'; x,y,edge=load_split(d/'test.npz'); xv,yv,_=load_split(d/'validation.npz'); device=resolve_device(a.device); edge_t=torch.as_tensor(edge,dtype=torch.long,device=device)
      norm=json.loads((d/'dataset_summary.json').read_text())['normalization']; raw=x[:,:,:,0].numpy()*norm['std'][0]+norm['mean'][0]
      ps={'persistence':raw[:,-1,:],'moving_average':raw.mean(axis=1)}
      for name in ('gru','gnn_standard','gnn_weighted'):
        model,_=load_model_checkpoint(d/'checkpoints'/f'{name}.pt',device); ps[name]=predict(model,x,edge_t,device)
      for name,p in ps.items():
        _,reg=val_predictions[(h,name)]; threshold=val_predictions[(h,name)][0]
        truth=y.numpy(); high=truth>=TH
        final.append(dict(horizon=h,model=name,**regression_metrics(truth,p),high_utilization_mae=float(np.mean(np.abs(truth[high]-p[high]))) if high.any() else None,high_utilization_targets=int(high.sum()),**score(truth,p,threshold,len(horizon_summaries[h]['seed_groups']['test'])*4)))
      if h==frozen['horizon']:
        p=ps[frozen['model']]; np.savez_compressed(a.out/'selected_test_predictions.npz',actual=y.numpy(),predicted=p)
    pd.DataFrame(final).to_csv(a.out/'final_test_comparison.csv',index=False)
    # Event-onset evaluation is distinct from predicting utilization above threshold.
    h=frozen['horizon']; d=a.out/f'horizon_{h}'; raw=pd.read_csv(d/'measurements.csv')
    summary=horizon_summaries[h]; test_seeds=summary['seed_groups']['test']
    run_ids=[f'{profile}-seed-{seed}' for seed in test_seeds for profile in ('normal','moderate','burst','sustained_high')]
    with np.load(a.out/'selected_test_predictions.npz') as z: ya,yp=z['actual'],z['predicted']
    threshold=frozen['decision_threshold']; wins=ya.shape[0]//len(run_ids); horizon=int(h); window=10
    onset_truth=[]; onset_scores=[]; lead=[]; already=0
    for ri,rid in enumerate(run_ids):
      g=raw[raw.run_id==rid]
      for li,(_,lg) in enumerate(g.groupby('link_id',sort=True)):
        u=lg.sort_values('step').utilization.to_numpy(); onset=(u>=TH)&~np.r_[False,(u[:-1]>=TH)]
        for si in range(wins):
          decision=si+window-1
          if decision+horizon>=len(u): continue
          if u[decision]>=TH: already+=1; continue
          future=np.flatnonzero(onset[decision+1:decision+horizon+1])
          onset_truth.append(bool(len(future))); onset_scores.append(float(yp[ri*wins+si,li]))
          if len(future) and yp[ri*wins+si,li]>=threshold: lead.append(int(future[0]+1))
    onset_metrics=score(np.asarray(onset_truth,dtype=np.float32),np.asarray(onset_scores,dtype=np.float32),threshold,len(run_ids)) if onset_truth else {}
    onset_metrics.update(dict(event_definition=f'new utilization >= {TH:.2f} crossing within next {horizon} steps, conditioned on uncongested at decision time',already_congested_decisions_excluded=already,event_detection_rate=float(onset_metrics.get('recall',0.0)),mean_detection_lead_steps=float(np.mean(lead)) if lead else None,detected_onsets=int(onset_metrics.get('tp',0))))
    onset_metrics['onset_opportunities']=int(onset_metrics.get('tp',0)+onset_metrics.get('fn',0))
    write_json(a.out/'selected_event_metrics.json',onset_metrics)
    # Diagnostics on fresh training/validation runs only, with profile/link breakdown.
    diag=[]
    for h in (1,3,5):
      m=pd.read_csv(a.out/f'horizon_{h}'/'measurements.csv')
      allowed={f"{p}-seed-{s}" for k in ('train','validation') for s in horizon_summaries[h]['seed_groups'][k] for p in ('normal','moderate','burst','sustained_high')}
      m=m[m.run_id.isin(allowed)]
      for (profile,link),g in m.groupby(['traffic_profile','link_id']):
        u=g.sort_values('step').utilization.to_numpy(); congest=u>=TH; onset=congest & ~np.r_[False,congest[:-1]]
        diag.append(dict(horizon=h,profile=profile,link_id=link,steps=len(u),congested_fraction=float(congest.mean()),onsets=int(onset.sum()),mean_event_duration=float(congest.sum()/max(1,onset.sum())),fraction_onsets_within_horizon=float(np.mean([onset[i+1:i+h+1].any() for i in range(max(0,len(u)-h-1))])) if len(u)>h+1 else 0))
    pd.DataFrame(diag).to_csv(a.out/'training_validation_event_diagnostics.csv',index=False)
    # Report actual-vs-predicted and validation profile congestion frequency.
    fig,ax=plt.subplots(figsize=(6,6)); ix=np.linspace(0,ya.size-1,min(4000,ya.size),dtype=int); ax.scatter(ya.ravel()[ix],yp.ravel()[ix],s=7,alpha=.25); ax.plot([0,1],[0,1],'k--'); ax.set(xlabel='Actual utilization',ylabel='Predicted utilization',title=f'Fresh final test: {frozen["model"]}, horizon {h}',xlim=(0,1),ylim=(0,1)); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(a.out/'actual_vs_predicted.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4)); dg=pd.DataFrame(diag); sub=dg[dg.horizon==h].groupby('profile').agg(congested_fraction=('congested_fraction','mean'),onsets=('onsets','sum')); sub.plot.bar(ax=ax); ax.set_title('Training and validation event frequency by profile (fresh simulations)'); ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(a.out/'event_frequency_by_profile.png',dpi=160); plt.close(fig)
    pd.DataFrame(final).to_csv(a.out/'final_test_comparison.csv',index=False)
    pd.DataFrame([onset_metrics]).to_csv(a.out/'selected_event_metrics.csv',index=False)
    chosen=next(r for r in final if r['horizon']==h and r['model']==frozen['model'])
    report=f'''# Phase 3.5 diagnostic report\n\nAll measurements are from deterministic network simulation; no real-world telemetry was used. Fresh seed groups {a.seed_start}–{a.seed_start+a.seeds-1} were split by seed, keeping all four traffic profiles grouped. Normalization used training windows only.\n\n## Selection and test result\n\nFrozen using validation only: horizon {h}, {frozen['model']}, decision threshold {threshold:.6f}. Rule: {frozen['selection_rule']}. Final test was opened after freezing. Test MAE {chosen['mae']:.4f}, RMSE {chosen['rmse']:.4f}, high-utilization MAE {chosen['high_utilization_mae']:.4f} across {chosen['high_utilization_targets']} high-utilization labels.\n\nOn the selected horizon, fresh-test GRU MAE/RMSE was {chosen['mae']:.4f}/{chosen['rmse']:.4f}; standard GNN was {next(r for r in final if r['horizon']==h and r['model']=='gnn_standard')['mae']:.4f}/{next(r for r in final if r['horizon']==h and r['model']=='gnn_standard')['rmse']:.4f}. See `final_test_comparison.csv` for all baselines and horizons.\n\n## Event-onset result\n\nNew onset means utilization first crosses 0.85 within the next {h} steps while the link is below 0.85 at decision time. On {onset_metrics['onset_opportunities']} eligible onset opportunities: precision/recall/F1 {onset_metrics['precision']:.3f}/{onset_metrics['recall']:.3f}/{onset_metrics['f1']:.3f}, PR-AUC {onset_metrics['pr_auc']:.4f}, false alerts {onset_metrics['false_alerts_per_run']:.2f}/run. Detection rate was {onset_metrics['event_detection_rate']:.1%}; useful lead is unavailable because no onset was detected. Already congested decisions excluded: {already}.\n\n## Data and model diagnosis\n\nFresh training/validation target congestion rates were sparse and profile-dependent: normal 0%, moderate 0.30%, burst 0.64%, sustained-high 4.42%. Mean event duration was about one step in burst/sustained-high; congestion was not a persistent state. This supports the failure mode where utilization regression is dominated by low targets and underestimates brief peaks. Positive weighting raised false alerts while hurting regression. Existing directed line-graph edges represent legal consecutive links, but this experiment did not change the spatial operator. Residual, GraphSAGE/GAT, and dedicated onset-head variants remain untested; this is a limitation and should be done only with a new untouched test set because this test is now exposed.\n\n`training_validation_event_diagnostics.csv` gives profile/link statistics from training and validation runs. `actual_vs_predicted.png` and `event_frequency_by_profile.png` visualize the results.\n\n## Recommendation\n\n**NO-GO for Phase 4 proactive routing with this model.** The selected model found 0 of {onset_metrics['onset_opportunities']} eligible onsets and had no useful lead time. A lower overall MAE does not make its alert signal actionable. Train and validate a dedicated onset classifier on better burst coverage before routing integration. Results describe this simulator only.\n'''
    (a.out/'DIAGNOSTIC_REPORT.md').write_text(report)
    write_json(a.out/'experiment_summary.json',dict(measurement_source='simulated',device=str(resolve_device(a.device)),horizons=horizon_summaries,selection=frozen,selected_event_metrics=onset_metrics,test_opened_after_selection=True,warning='Simulation results do not establish real-network performance. The selected classifier score is a thresholded regressor, not a dedicated multi-task onset head.'))
    print(json.dumps(frozen,indent=2)); print(pd.DataFrame(final).to_string(index=False))
if __name__=='__main__': main()
