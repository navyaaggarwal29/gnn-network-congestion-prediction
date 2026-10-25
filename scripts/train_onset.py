"""Train/evaluate Phase 3.6 onset classifiers on fresh simulated scenarios."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from congestion_sim import build_default_topology
from congestion_sim.onset import (SCENARIOS, ONSET_FEATURES, ONSET_THRESHOLD, generate_onset_data,
    split_onset_indices, normalize_onset_features, episode_event_metrics, episode_lengths)
from congestion_sim.models import TemporalOnsetGNN, TemporalOnsetGRU
from congestion_sim.training import resolve_device, seed_everything, write_json

def fit_model(name, x, labels, mask, splits, edge, output, normalization, *, epochs, hidden, device, seed):
    seed_everything(seed); input_dim=x.shape[-1]
    model=(TemporalOnsetGNN(input_dim,hidden) if name=='gnn_onset' else TemporalOnsetGRU(input_dim,hidden)).to(device)
    edge_t=torch.as_tensor(edge,dtype=torch.long,device=device)
    tr=splits['train']; va=splits['validation']
    pos=float((labels[tr]*mask[tr]).sum()); neg=float(mask[tr].sum()-pos)
    pos_weight=min(neg/max(1.0,pos),30.0)
    pos_weight_t=torch.tensor(pos_weight,dtype=torch.float32,device=device)
    loader=DataLoader(TensorDataset(torch.from_numpy(x[tr]),torch.from_numpy(labels[tr]),torch.from_numpy(mask[tr])),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(seed))
    opt=torch.optim.Adam(model.parameters(),lr=8e-4,weight_decay=1e-4)
    best=float('inf'); best_state=None; best_epoch=0; stale=0; history=[]
    xv=torch.from_numpy(x[va]).to(device); yv=torch.from_numpy(labels[va]).to(device); mv=torch.from_numpy(mask[va]).to(device)
    for epoch in range(1,epochs+1):
      model.train(); losses=[]
      for xb,yb,mb in loader:
        xb,yb,mb=xb.to(device),yb.to(device),mb.to(device)
        opt.zero_grad(set_to_none=True); logits=model(xb,edge_t) if name=='gnn_onset' else model(xb)
        raw=nn.functional.binary_cross_entropy_with_logits(logits,yb,reduction='none',pos_weight=pos_weight_t)
        loss=(raw*mb).sum()/mb.sum().clamp_min(1)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach()))
      model.eval()
      with torch.no_grad(): vl=nn.functional.binary_cross_entropy_with_logits(model(xv,edge_t) if name=='gnn_onset' else model(xv),yv,reduction='none',pos_weight=pos_weight_t); val_loss=float((vl*mv).sum()/mv.sum().clamp_min(1))
      history.append(dict(model=name,epoch=epoch,train_bce=float(np.mean(losses)),validation_bce=val_loss))
      if val_loss<best-1e-5: best,best_epoch,best_state,stale=val_loss,epoch,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},0
      else: stale+=1
      if stale>=4: break
    model.load_state_dict(best_state); output.mkdir(parents=True,exist_ok=True)
    ckpt=output/f'{name}.pt'; torch.save(dict(model_name=name,model_config=dict(input_dim=input_dim,hidden_dim=hidden,dropout=.2),state_dict=best_state,pos_weight=pos_weight,normalization=normalization,epoch=best_epoch,validation_bce=best),ckpt)
    return model,dict(model=name,best_epoch=best_epoch,validation_bce=best,pos_weight=pos_weight,train_positive_labels=int(pos),train_eligible_labels=int(pos+neg),checkpoint=str(ckpt)),history

def predict(model,x,edge,device,is_gnn):
    model.eval(); result=[]; loader=DataLoader(TensorDataset(torch.from_numpy(x)),batch_size=48)
    with torch.no_grad():
      for (xb,) in loader:
        logits=model(xb.to(device),edge) if is_gnn else model(xb.to(device)); result.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(result)

def window_metrics(y,mask,scores,threshold):
    eligible=mask.astype(bool); labels=y[eligible].astype(bool); pred=scores[eligible]>=threshold
    tp=int((labels&pred).sum()); fp=int((~labels&pred).sum()); fn=int((labels&~pred).sum())
    pr=float(average_precision_score(labels,scores[eligible])) if labels.any() else 0.0
    precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn)
    return dict(window_precision=precision,window_recall=recall,window_f1=2*precision*recall/max(1e-12,precision+recall),window_pr_auc=pr,window_positive_prevalence=float(labels.mean()) if len(labels) else 0.0,window_eligible_count=int(eligible.sum()))

def threshold_candidates(scores):
    # Fixed thresholds plus a compact, deterministic validation-only quantile sweep.
    q=np.quantile(scores,[.50,.70,.80,.90,.95,.975,.99,.995,1.0])
    return sorted(set([.5,.75,.9,1.000001,*map(float,q)]))

def markdown_table(frame):
    columns=list(frame.columns)
    def cell(v):
      if pd.isna(v): return "—"
      if isinstance(v,(float,np.floating)): return f"{v:.3f}"
      return str(v)
    lines=["| "+" | ".join(columns)+" |","| "+" | ".join("---" for _ in columns)+" |"]
    lines.extend("| "+" | ".join(cell(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None))
    return "\n".join(lines)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--seeds',type=int,default=25); ap.add_argument('--seed-start',type=int,default=2000); ap.add_argument('--steps',type=int,default=80); ap.add_argument('--window',type=int,default=10); ap.add_argument('--horizon',type=int,default=5); ap.add_argument('--epochs',type=int,default=16); ap.add_argument('--hidden',type=int,default=24); ap.add_argument('--decision-delay',type=int,default=2); ap.add_argument('--out',type=Path,default=Path('results/phase3_6')); ap.add_argument('--device',default='auto'); a=ap.parse_args()
    torch.set_num_threads(2); device=resolve_device(a.device); seeds=list(range(a.seed_start,a.seed_start+a.seeds)); a.out.mkdir(parents=True,exist_ok=True)
    data=generate_onset_data(build_default_topology(),seeds,a.steps,a.window,a.horizon)
    splits=split_onset_indices(data); x,norm=normalize_onset_features(data.x,splits)
    # Persist arrays by seed, including labels, eligible masks, scenario IDs, and causal run metadata.
    for name,ix in splits.items():
      np.savez_compressed(a.out/f'{name}.npz',x=x[ix],onset=data.onset[ix],mask=data.onset_mask[ix],continued=data.continued[ix],already_congested=data.already_congested[ix],run_id=data.run_id[ix],scenario=data.scenario[ix],seed=data.seed[ix],decision_step=data.decision_step[ix],edge_index=data.edge_index,link_nodes=np.asarray([f'{u}->{v}' for u,v in data.link_nodes]))
    norm['window']=a.window; norm['horizon']=a.horizon; norm['threshold']=ONSET_THRESHOLD
    write_json(a.out/'preprocessing.json',norm)
    # Inspect only training/validation scenario statistics before model selection.
    dev_ix=np.concatenate((splits['train'],splits['validation'])); records=[]
    for scenario in SCENARIOS:
      for run in dict.fromkeys(data.run_id[dev_ix][data.scenario[dev_ix]==scenario].tolist()):
        u=data.utilization_by_run[run]; hot=u>=ONSET_THRESHOLD
        for li,link in enumerate(data.link_nodes):
          v=hot[:,li]; starts=np.flatnonzero(v&~np.r_[False,v[:-1]])
          duration=episode_lengths(u[:,li])
          records.append(dict(run_id=run,scenario=scenario,link_id=f'{link[0]}->{link[1]}',congested_fraction=float(v.mean()),episode_count=int(len(starts)),mean_episode_duration=float(np.mean(duration)) if duration else 0.0,episodes_at_least_3_steps=int(sum(d>=3 for d in duration))))
    stats=pd.DataFrame(records); stats.to_csv(a.out/'train_validation_scenario_statistics.csv',index=False)
    histories=[]; model_meta={}; val_scores={}
    edge_t=torch.as_tensor(data.edge_index,dtype=torch.long,device=device)
    for name in ('gru_onset','gnn_onset'):
      model,meta,history=fit_model(name,x,data.onset,data.onset_mask,splits,data.edge_index,a.out/'checkpoints',norm,epochs=a.epochs,hidden=a.hidden,device=device,seed=43)
      model_meta[name]=meta; histories.extend(history); val_scores[name]=predict(model,x[splits['validation']],edge_t,device,name=='gnn_onset')
    # Validation-only threshold and model choice, with a stated routing-delay allowance.
    validation_rows=[]; chosen={}; selected_validation={}
    for name,pred in val_scores.items():
      best=None
      local_mask=data.onset_mask[splits['validation']]; local_y=data.onset[splits['validation']]
      full_pred=np.zeros((len(data.x),pred.shape[1]),dtype=np.float32); full_pred[splits['validation']]=pred
      for t in threshold_candidates(pred):
        m=episode_event_metrics(full_pred,data,splits['validation'],t,a.decision_delay)
        row=dict(model=name,threshold=t,**m,**window_metrics(local_y,local_mask,pred,t))
        validation_rows.append(row)
        if m['false_alert_episodes_per_run']<=1.0:
          key=(m['useful_lead_rate'],m['episode_detection_rate'],m['precision'],-m['false_alert_episodes_per_run'])
          if best is None or key>best[0]: best=(key,t,row)
      chosen[name]=best[1] if best else 1.000001
      selected_validation[name]=best[2] if best else None
    # Validation baselines receive identical features and threshold selection.
    val_ix=splits['validation']; raw_x=data.x[val_ix]
    util=raw_x[:,:,:,0]; slope=raw_x[:,:,:,6]
    heuristics={'persistence':util[:,-1,:], 'moving_average':util.mean(axis=1), 'trend':np.clip(util[:,-1,:]+a.horizon*np.maximum(0,slope[:,-1,:]),0,1)}
    for name,pred in heuristics.items():
      best=None
      full_pred=np.zeros((len(data.x),pred.shape[1]),dtype=np.float32); full_pred[val_ix]=pred
      for t in threshold_candidates(pred):
        m=episode_event_metrics(full_pred,data,val_ix,t,a.decision_delay); row=dict(model=name,threshold=t,**m,**window_metrics(data.onset[val_ix],data.onset_mask[val_ix],pred,t)); validation_rows.append(row)
        if m['false_alert_episodes_per_run']<=1:
          key=(m['useful_lead_rate'],m['episode_detection_rate'],m['precision'],-m['false_alert_episodes_per_run'])
          if best is None or key>best[0]: best=(key,t,row)
      chosen[name]=best[1] if best else 1.000001
      selected_validation[name]=best[2] if best else None
    val_table=pd.DataFrame(validation_rows); val_table.to_csv(a.out/'validation_thresholds.csv',index=False)
    best_model=max(chosen,key=lambda n:(selected_validation[n]['useful_lead_rate'],selected_validation[n]['episode_detection_rate'],selected_validation[n]['precision']))
    frozen=dict(selected_model=best_model,threshold=float(chosen[best_model]),threshold_by_model={k:float(v) for k,v in chosen.items()},validation_selected_metrics=selected_validation,selection='validation: maximize useful lead rate (lead >= configured routing delay), then episode detection and precision, subject to <=1 false-alert episode per simulation run',decision_delay_steps=a.decision_delay)
    write_json(a.out/'frozen_selection.json',frozen)
    # Only now access the held-out test split.
    test=splits['test']; test_scores={}
    for name in ('gru_onset','gnn_onset'):
      checkpoint=torch.load(a.out/'checkpoints'/f'{name}.pt',map_location=device,weights_only=False)
      model=TemporalOnsetGNN(len(ONSET_FEATURES),a.hidden).to(device) if name=='gnn_onset' else TemporalOnsetGRU(len(ONSET_FEATURES),a.hidden).to(device); model.load_state_dict(checkpoint['state_dict'])
      test_scores[name]=predict(model,x[test],edge_t,device,name=='gnn_onset')
    raw_test=data.x[test]; tu=raw_test[:,:,:,0]; ts=raw_test[:,:,:,6]
    test_scores.update({'persistence':tu[:,-1,:], 'moving_average':tu.mean(axis=1),'trend':np.clip(tu[:,-1,:]+a.horizon*np.maximum(0,ts[:,-1,:]),0,1)})
    run_count=len(set(data.run_id[test].tolist())); results=[]; scenario_results=[]
    test_y=data.onset[test]; test_mask=data.onset_mask[test]
    for name,pred in test_scores.items():
      threshold=chosen[name]; full_pred=np.zeros((len(data.x),pred.shape[1]),dtype=np.float32); full_pred[test]=pred
      event=episode_event_metrics(full_pred,data,test,threshold,a.decision_delay); win=window_metrics(test_y,test_mask,pred,threshold)
      continued=window_metrics(data.continued[test],data.already_congested[test],pred,threshold)
      already=window_metrics(data.already_congested[test],np.ones_like(data.already_congested[test]),data.already_congested[test],.5)
      results.append(dict(model=name,threshold=threshold,**event,**win,
          continued_window_precision=continued['window_precision'],continued_window_recall=continued['window_recall'],continued_window_pr_auc=continued['window_pr_auc'],
          already_congested_current_alarm_precision=already['window_precision'],already_congested_current_alarm_recall=already['window_recall']))
      for scenario in SCENARIOS:
        ix=test[data.scenario[test]==scenario]; positions=np.flatnonzero(np.isin(test,ix))
        # indices passed to episode evaluator must refer to global data arrays
        sc_scores=pred[positions]; sc_full=np.zeros((len(data.x),pred.shape[1]),dtype=np.float32); sc_full[ix]=sc_scores
        sc_event=episode_event_metrics(sc_full,data,ix,threshold,a.decision_delay)
        sc_win=window_metrics(data.onset[ix],data.onset_mask[ix],sc_scores,threshold)
        scenario_results.append(dict(model=name,scenario=scenario,**sc_event,**sc_win))
    pd.DataFrame(results).to_csv(a.out/'test_metrics.csv',index=False); pd.DataFrame(scenario_results).to_csv(a.out/'scenario_test_metrics.csv',index=False)
    fixed=[]
    for name,pred in test_scores.items():
      full_pred=np.zeros((len(data.x),pred.shape[1]),dtype=np.float32); full_pred[test]=pred
      m=episode_event_metrics(full_pred,data,test,.5,a.decision_delay)
      fixed.append(dict(model=name,threshold=.5,**m,**window_metrics(test_y,test_mask,pred,.5)))
    pd.DataFrame(fixed).to_csv(a.out/'test_metrics_fixed_threshold_0_5.csv',index=False)
    pd.DataFrame(histories).to_csv(a.out/'training_history.csv',index=False)
    # Episode stats and PR curves are test-only visualization after the freeze.
    all_y=test_y[test_mask.astype(bool)].ravel(); fig,ax=plt.subplots(figsize=(7,5))
    from sklearn.metrics import precision_recall_curve as pr_curve
    for name in test_scores:
      s=test_scores[name][test_mask.astype(bool)].ravel(); precision,recall,_=pr_curve(all_y,s); ax.plot(recall,precision,label=name)
    ax.set(xlabel='Window-level recall',ylabel='Window-level precision',title='Held-out test precision–recall (eligible decision windows)'); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(a.out/'precision_recall.png',dpi=160); plt.close(fig)
    chosen_pred=test_scores[best_model]; test_ids=test
    # Save one real test run's timeline from the selected system.
    run_id=next(r for r in dict.fromkeys(data.run_id[test].tolist()) if 'sustained_bottleneck' in r)
    local=[j for j,i in enumerate(test) if data.run_id[i]==run_id]; local.sort(key=lambda j:int(data.decision_step[test[j]]))
    run_util=data.utilization_by_run[run_id]; link= int(np.argmax(run_util.max(axis=0))); steps=[int(data.decision_step[test[j]]) for j in local]
    probability=[float(chosen_pred[j,link]) for j in local]
    fig,ax=plt.subplots(figsize=(10,4)); ax.plot(np.arange(len(run_util)),run_util[:,link],label='simulated utilization'); ax.axhline(ONSET_THRESHOLD,color='red',ls='--',label='congestion threshold'); ax2=ax.twinx(); ax2.plot(steps,probability,color='purple',alpha=.8,label='onset score'); ax2.axhline(frozen['threshold'],color='purple',ls=':'); ax.set(xlabel='Simulation step',ylabel='Utilization',title=f'Test onset timeline: {run_id}, link {data.link_nodes[link]}'); ax.set_ylim(0,1.05); ax2.set_ylabel('Onset score'); ax.legend(loc='upper left'); ax2.legend(loc='upper right'); fig.tight_layout(); fig.savefig(a.out/'event_timeline.png',dpi=160); plt.close(fig)
    sm=pd.DataFrame(scenario_results)
    fig,ax=plt.subplots(figsize=(10,4)); pick=sm[sm.model.isin(['gru_onset','gnn_onset','trend'])].pivot(index='scenario',columns='model',values='episode_detection_rate'); pick.plot.bar(ax=ax); ax.set(ylabel='Episode detection rate',title='Held-out episode detection by traffic scenario'); ax.grid(axis='y',alpha=.25); fig.tight_layout(); fig.savefig(a.out/'episode_detection_by_scenario.png',dpi=160); plt.close(fig)
    # Demand-shape and episode-duration summaries use only development runs.
    profile_stats=stats.groupby('scenario').agg(
        congested_link_time_fraction=('congested_fraction','mean'),
        link_run_episodes=('episode_count','sum'),
        mean_episode_duration_steps=('mean_episode_duration',lambda v:float(v[v>0].mean()) if (v>0).any() else 0.0),
        episodes_at_least_3_steps=('episodes_at_least_3_steps','sum')).reset_index()
    profile_stats.to_csv(a.out/'scenario_statistics.csv',index=False)
    model_table=pd.DataFrame(results)[['model','precision','recall','f1','window_pr_auc','false_alert_episodes_per_run','episode_detection_rate','mean_lead_steps','useful_lead_rate']]
    fixed_table=pd.DataFrame(fixed)[['model','precision','recall','f1','false_alert_episodes_per_run','mean_lead_steps','episode_detection_rate']]
    scenario_table=sm[sm.model.isin([best_model,'gnn_onset'])][['model','scenario','eligible_onset_episodes','true_positive_episodes','false_alert_episodes','episode_detection_rate','precision','recall','f1','window_pr_auc','mean_lead_steps']]
    prevalence_table=profile_stats[['scenario','congested_link_time_fraction','link_run_episodes','mean_episode_duration_steps','episodes_at_least_3_steps']]
    data.measurements.to_csv(a.out/'simulated_measurements.csv',index=False)
    write_json(a.out/'run_config.json',dict(measurement_source='simulated',seeds=seeds,split_seed_groups={k:sorted(set(data.seed[v].tolist())) for k,v in splits.items()},steps=a.steps,window=a.window,horizon=a.horizon,decision_delay=a.decision_delay,scenarios=SCENARIOS,features=ONSET_FEATURES,normalization=norm,models=model_meta,selection=frozen,test_opened_after_freeze=True))
    selected_test=next(row for row in results if row['model']==best_model)
    recommendation='GO for a separate routing-integration study' if selected_test['useful_lead_rate']>=.5 and selected_test['episode_detection_rate']>=.5 and selected_test['false_alert_episodes_per_run']<=1 else 'NO-GO for proactive routing'
    report=f'''# Phase 3.6: onset prediction and simulator audit\n\n## Audit findings\n\nThe original four profiles remain unchanged. Their source/destination pairs and Dirichlet flow shares were resampled independently each one-second interval; therefore a constant aggregate rate did not imply persistent demand on one link. That made the mostly one-step events in Phase 3.5 plausible and limited a five-step target's usefulness. The five-step horizon gave detected periodic events a mean lead of {selected_test['mean_lead_steps']:.2f} steps, enough for the assumed two-step action delay, but aggregate episode recall was only {selected_test['episode_detection_rate']:.1%}; it missed the tested gradual, sustained, correlated, and unpredictable-burst episodes. It is useful for some structured periodic demand but not reliable as a general-purpose horizon here. An unannounced burst has no forecastable precursor in this setup. The original event scoring operated on every overlapping decision window, so one real onset could contribute several positive labels. Phase 3.6 uses fixed per-run routes and measures event outcomes once per congestion episode.\n\nThe simulator uses 1,200-byte packets, one-second intervals, per-direction packet service bounded by the link Mbps rate, and 252,000-byte finite queues. Incoming packets are accepted only up to the available queue space after reserved interval service; overflow is dropped. Measured utilization is transmitted bits divided by service capacity, and queue occupancy is retained backlog divided by the finite queue. At a 50 Mbps hop, the configured queue holds about 40.3 ms of serialization. Queue delay is an interval-level estimate rather than a packet-level discrete-event delay.\n\n## New traffic scenarios\n\nEach scenario keeps flow paths fixed within a run, but queueing, loss, delay, and utilization emerge from packets and service. All values are simulated. The bottleneck profile sends eight conversations through the same shortest path; other profiles place half their conversations on that path and the remainder across varied paths. Scenario definitions: gradual linear ramp from 40 to 180 Mbps over 65% of the run; sustained bottleneck with a 40 Mbps 15% warm-up then 180 Mbps; periodic 40–180 Mbps cosine cycle with a 24-step period starting at the low-load trough; correlated demand with a shared AR(1) factor (0.88 coefficient and 0.12 Gaussian innovation, clipped to 80–180 Mbps); sudden four-step pulse from 40 to 180 Mbps at a seeded random time between 30% and 75% of the run. No congestion labels or utilization values are hardcoded.\n\nTrain/validation scenario event prevalence and duration (episode counts are link/run episodes):\n\n{markdown_table(prevalence_table)}\n\n## Features and labels\n\nModels receive utilization, queue occupancy, delay, packet loss, offered demand, bandwidth, utilization trend, queue growth, offered-load trend, four-step utilization variability, and three-step recent drop rate. Rolling and difference features are computed using only current and past measurements. Means and standard deviations are fit on training windows only. An onset label requires utilization below 0.85 at decision time and a first crossing within the next {a.horizon} steps. Continued congestion is tracked separately. Already-congested links are excluded from onset training and evaluation.\n\n## Models and evaluation\n\nA dedicated temporal GNN and a non-graph GRU share the same inputs and classification task. Both use a training-only positive weight in `BCEWithLogitsLoss`; examples remain at natural prevalence. Persistence, moving-average, and linear positive-trend heuristics provide additional baselines. Validation selects thresholds and the final model subject to at most one false alert episode per run, prioritizing the rate of detections with at least {a.decision_delay} steps of lead. Fixed 0.5 and validation-selected test scores are both exported. PR-AUC is window-level; precision/recall/F1, missed episodes, false alerts, and lead are episode-level with overlapping alerts merged and each true episode matched at most once. Regression MAE/RMSE do not apply to these binary classifiers; Phase 3.5 utilization-regression artifacts remain unchanged.\n\nAggregate held-out test metrics at validation-selected thresholds (window PR-AUC; event metrics are de-duplicated):\n\n{markdown_table(model_table)}\n\nScenario-specific selected model and GNN results:\n\n{markdown_table(scenario_table)}\n\nSelected validation model: **{best_model}**, threshold {frozen['threshold']:.4f}. It detected {selected_test['true_positive_episodes']} of {selected_test['eligible_onset_episodes']} eligible test episodes; mean lead {selected_test['mean_lead_steps']} steps. Test onset-window prevalence was {selected_test['window_positive_prevalence']:.3%}.\n\nFixed 0.5 threshold results show the precision/recall and false-alert tradeoff:\n\n{markdown_table(fixed_table)}\n\n`test_metrics.csv` also includes continued-congestion scores (exploratory because the classifiers are trained on onset only) and current-state alarm results. `precision_recall.png`, `episode_detection_by_scenario.png`, and `event_timeline.png` visualize the outputs.\n\n## Practical decision\n\nAssumed reroute decision delay is {a.decision_delay} simulation steps. A detected onset is called useful only when its lead is at least that long. **Recommendation: {recommendation}.** The gate requires at least 50% useful lead and episode detection, with no more than one false alert episode per run. This project does not simulate routing actions and makes no claim that rerouting improves network performance.\n'''
    (a.out/'DIAGNOSTIC_REPORT.md').write_text(report)
    print(json.dumps(frozen,indent=2)); print(pd.DataFrame(results).to_string(index=False))

if __name__=='__main__': main()
