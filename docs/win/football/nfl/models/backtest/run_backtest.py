#!/usr/bin/env python3
"""Strict chronological NFL model -> calibration -> selection replay. Writes only this folder."""
from __future__ import annotations
import hashlib, importlib.util, json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np, pandas as pd, yaml

BACKTEST_DIR=Path(__file__).resolve().parent; NFL_ROOT=BACKTEST_DIR.parents[1]
STEP13_PATH=NFL_ROOT/'scripts/training/step13.py'; STEP14_PATH=NFL_ROOT/'scripts/training/step14_calibration.py'; SELECTIONS_PATH=NFL_ROOT/'scripts/02_select/selections.py'
SETTINGS_PATH=NFL_ROOT/'config/settings.yaml'; MARKETS_PATH=NFL_ROOT/'config/markets.yaml'; SCHEMA_PATH=NFL_ROOT/'models/step11_feature_schema.json'
PREDICTIONS_PATH=BACKTEST_DIR/'chronological_predictions.csv'; PROBABILITIES_PATH=BACKTEST_DIR/'walkforward_probabilities.csv'; MONEYLINE_PICKS_PATH=BACKTEST_DIR/'historical_moneyline_selected.csv'; SPREAD_PICKS_PATH=BACKTEST_DIR/'historical_spread_selected.csv'; TOTAL_PICKS_PATH=BACKTEST_DIR/'historical_total_selected.csv'; SUMMARY_SEASON_PATH=BACKTEST_DIR/'summary_by_season_market.csv'; SUMMARY_OVERALL_PATH=BACKTEST_DIR/'summary_overall_market.csv'; METADATA_PATH=BACKTEST_DIR/'run_metadata.json'
PREDICTION_OUTPUT_COLUMNS=['season','week','gameday','gametime','game_id','away_team','home_team','game_type','neutral_site_flag','roof','temp','wind','hist_temperature','hist_wind_speed','hist_precip_probability','hist_precip_type','away_moneyline','home_moneyline','spread_line','away_spread_odds','home_spread_odds','total_line','under_odds','over_odds','training_rows','predicted_margin','predicted_total','predicted_home_score','predicted_away_score','actual_margin','actual_total_points','actual_home_win','actual_home_ats_result','actual_total_result']
PROBABILITY_COLUMNS=['home_win_probability','away_win_probability','home_cover_probability','away_cover_probability','over_probability','under_probability']
CAL_COLS=['ml_calibration_rows','ml_calibration_intercept','ml_calibration_slope','ml_calibration_status','spread_calibration_rows','spread_calibration_intercept','spread_calibration_slope','spread_calibration_status','total_calibration_rows','total_calibration_intercept','total_calibration_slope','total_calibration_status']
MONEYLINE_PICK_COLUMNS=['season','week','game_id','away_team','home_team','ml_selection','ml_odds_american','ml_model_probability','ml_implied_probability','ml_edge','ml_ev','ml_kelly','ml_result','ml_flat_profit_units','ml_kelly_risk_units','ml_kelly_profit_units']
SPREAD_PICK_COLUMNS=['season','week','game_id','away_team','home_team','spread_selection','spread_line','spread_odds_american','spread_model_probability','spread_implied_probability','spread_edge','spread_ev','spread_kelly','spread_result','spread_flat_profit_units','spread_kelly_risk_units','spread_kelly_profit_units']
TOTAL_PICK_COLUMNS=['season','week','game_id','away_team','home_team','total_selection','total_line','total_odds_american','total_model_probability','total_implied_probability','total_edge','total_ev','total_kelly','total_result','total_flat_profit_units','total_kelly_risk_units','total_kelly_profit_units']
SUMMARY_COLUMNS=['season','market','bets','wins','losses','pushes','decisions','win_rate_pct','flat_risk_units','flat_net_units','flat_roi_pct','kelly_risk_units','kelly_net_units','kelly_roi_pct']

def fail(m): raise RuntimeError(m)
def clean(v):
 s='' if v is None else str(v).strip(); return '' if s.casefold() in {'','nan','none','null','<na>','nat'} else s
def num(v):
 try:x=float(clean(v))
 except (TypeError,ValueError):return None
 return x if math.isfinite(x) else None
def gid(v):
 s=clean(v); return s[:-2] if s.endswith('.0') else s
def ensure_write_path(p):
 p=Path(p).resolve(); root=BACKTEST_DIR.resolve()
 try:p.relative_to(root)
 except ValueError:fail(f'WRITE BOUNDARY VIOLATION: {p}')
 return p
def atomic_write_csv(df,p):
 p=ensure_write_path(p); p.parent.mkdir(parents=True,exist_ok=True); t=ensure_write_path(p.with_suffix(p.suffix+'.tmp')); df.to_csv(t,index=False); os.replace(t,p)
def atomic_write_json(o,p):
 p=ensure_write_path(p); t=ensure_write_path(p.with_suffix(p.suffix+'.tmp')); t.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
def load_module(p,n):
 s=importlib.util.spec_from_file_location(n,p)
 if s is None or s.loader is None:fail(f'Cannot load {p}')
 m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def read_yaml(p):
 with Path(p).open(encoding='utf-8') as f:o=yaml.safe_load(f)
 if not isinstance(o,dict):fail(f'Expected YAML mapping: {p}')
 return o
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def pred_row(r,pm,pt,n):
 return {'season':int(float(r.season)),'week':int(float(r.week)),'gameday':clean(r.gameday),'gametime':clean(r.gametime),'game_id':gid(r.game_id),'away_team':clean(r.away_team),'home_team':clean(r.home_team),'game_type':clean(r.get('game_type','')),'neutral_site_flag':r.get('neutral_site_flag',''),'roof':r.get('roof',''),'temp':r.get('temp',''),'wind':r.get('wind',''),'hist_temperature':r.get('hist_temperature',''),'hist_wind_speed':r.get('hist_wind_speed',''),'hist_precip_probability':r.get('hist_precip_probability',''),'hist_precip_type':r.get('hist_precip_type',''),'away_moneyline':r.get('away_moneyline',''),'home_moneyline':r.get('home_moneyline',''),'spread_line':float(r.spread_line),'away_spread_odds':r.get('away_spread_odds',''),'home_spread_odds':r.get('home_spread_odds',''),'total_line':float(r.total_line),'under_odds':r.get('under_odds',''),'over_odds':r.get('over_odds',''),'training_rows':n,'predicted_margin':float(pm),'predicted_total':float(pt),'predicted_home_score':(pt+pm)/2,'predicted_away_score':(pt-pm)/2,'actual_margin':float(r.margin),'actual_total_points':float(r.total_points),'actual_home_win':int(float(r.home_win)),'actual_home_ats_result':clean(r.home_ats_result).upper(),'actual_total_result':clean(r.total_result).upper()}
def fit_cal(step14,x,y):
 x=pd.to_numeric(x,errors='coerce'); y=pd.to_numeric(y,errors='coerce'); m=x.notna()&y.notna(); xa=x[m].to_numpy(float); ya=y[m].to_numpy(float); n=len(ya)
 if n<2:return n,np.nan,np.nan,'INSUFFICIENT_PRIOR_ROWS'
 if set(np.unique(ya))!={0.,1.}:return n,np.nan,np.nan,'PRIOR_OUTCOMES_ONE_CLASS'
 try:a,b=step14.fit_logistic_1d(xa,ya)
 except Exception as e:return n,np.nan,np.nan,f'FIT_UNAVAILABLE:{type(e).__name__}'
 if not(math.isfinite(a) and math.isfinite(b) and b>0):return n,np.nan,np.nan,'NONPOSITIVE_OR_INVALID_SLOPE'
 return n,float(a),float(b),'AVAILABLE'
def cals(step14,p):
 if p.empty:return {k:(0,np.nan,np.nan,'INSUFFICIENT_PRIOR_ROWS') for k in ('ml','spread','total')}
 am=pd.to_numeric(p.actual_margin); pm=pd.to_numeric(p.predicted_margin); sl=pd.to_numeric(p.spread_line); pt=pd.to_numeric(p.predicted_total); tl=pd.to_numeric(p.total_line)
 m=am.ne(0); ml=fit_cal(step14,pm[m],(am[m]>0).astype(int)); a=p.actual_home_ats_result.astype(str).str.upper(); m=a.isin(['WIN','LOSS']); sp=fit_cal(step14,pm[m]-sl[m],(a[m]=='WIN').astype(int)); t=p.actual_total_result.astype(str).str.upper(); m=t.isin(['OVER','UNDER']); to=fit_cal(step14,pt[m]-tl[m],(t[m]=='OVER').astype(int)); return {'ml':ml,'spread':sp,'total':to}
def cp(step14,c,x):
 return None if c[3]!='AVAILABLE' else float(step14.sigmoid(np.asarray([c[1]+c[2]*x],float))[0])

def present(v):
 x=num(v); return x is not None and x!=0
def history_row(r,sel):
 sl=num(r.spread_line); tl=num(r.total_line); pt=clean(r.get('hist_precip_type','')).casefold(); odds=int((present(r.home_moneyline) and present(r.away_moneyline)) or (sl is not None and present(r.home_spread_odds) and present(r.away_spread_odds)) or (tl is not None and present(r.over_odds) and present(r.under_odds)))
 return pd.Series({'game_id':gid(r.game_id),'season_type':sel.normalize_season_type(r.game_type),'sched_neutral_site':r.get('neutral_site_flag',''),'sched_roof':r.get('roof',''),'sched_odds_available':odds,'sched_home_moneyline_american':r.home_moneyline,'sched_away_moneyline_american':r.away_moneyline,'sched_home_spread':-sl if sl is not None else '','sched_away_spread':sl if sl is not None else '','sched_home_spread_american':r.home_spread_odds,'sched_away_spread_american':r.away_spread_odds,'sched_total':tl if tl is not None else '','sched_over_american':r.over_odds,'sched_under_american':r.under_odds,'wx_temperature':r.get('hist_temperature',r.get('temp','')),'wx_wind_speed':r.get('hist_wind_speed',r.get('wind','')),'wx_wind_gust':'','wx_precip_probability':r.get('hist_precip_probability',''),'wx_rain_flag':1 if 'rain' in pt else 0 if pt else '','wx_snow_flag':1 if 'snow' in pt else 0 if pt else '',**{c:r.get(c,np.nan) for c in PROBABILITY_COLUMNS}})
def pair(r,a,b):
 x,y=num(r.get(a)),num(r.get(b)); return x is not None and y is not None and 0<=x<=1 and 0<=y<=1 and math.isclose(x+y,1,abs_tol=1e-9)
def evaluate_single_market(r,market,settings,doc,sel):
 cfg=doc.get('markets',{}).get(market); gf=settings.get('game_filters',{})
 if not isinstance(cfg,dict):fail(f'markets.yaml missing {market}')
 th=sel.resolve_thresholds(settings,market,cfg); w=history_row(r,sel); ok,reason=sel.global_eligibility(w,settings); pre={'moneyline':'ml','spread':'spread','total':'total'}[market]
 if not ok:
  z=sel.empty_market(pre,reason); z.update({f'{pre}_line':np.nan} if market!='moneyline' else {}); return z
 if (sel.parse_int(w.get('sched_odds_available','')) or 0)!=1:
  z=sel.empty_market(pre,'CURRENT_ODDS_UNAVAILABLE'); z.update({f'{pre}_line':np.nan} if market!='moneyline' else {}); return z
 if market=='moneyline':return sel.evaluate_moneyline(w,cfg,th) if pair(w,'home_win_probability','away_win_probability') else sel.empty_market('ml','CALIBRATION_NOT_AVAILABLE')
 if market=='spread':
  if not pair(w,'home_cover_probability','away_cover_probability'):z=sel.empty_market('spread','CALIBRATION_NOT_AVAILABLE'); z['spread_line']=np.nan; return z
  return sel.evaluate_spread(w,cfg,th)
 if not pair(w,'over_probability','under_probability'):z=sel.empty_market('total','CALIBRATION_NOT_AVAILABLE'); z['total_line']=num(r.total_line); return z
 return sel.evaluate_total(w,cfg,th,gf)

def profit(res,o,risk):
 if res=='PUSH':return 0.
 if res=='LOSS':return -risk
 return risk*(o/100 if o>0 else 100/abs(o))
def grade_moneyline(r,s):
 m=float(r.actual_margin); res='PUSH' if abs(m)<1e-12 else 'WIN' if s['ml_selection']==('HOME' if m>0 else 'AWAY') else 'LOSS'; o=float(s['ml_odds_american']); k=float(s['ml_kelly']); return {'ml_result':res,'ml_flat_profit_units':profit(res,o,1),'ml_kelly_risk_units':k,'ml_kelly_profit_units':profit(res,o,k)}
def grade_spread(r,s):
 home=clean(r.actual_home_ats_result).upper(); res=home if s['spread_selection']=='HOME' else {'WIN':'LOSS','LOSS':'WIN','PUSH':'PUSH'}[home]; exp=-float(r.spread_line) if s['spread_selection']=='HOME' else float(r.spread_line)
 if not math.isclose(float(s['spread_line']),exp,abs_tol=1e-9):fail(f'Spread line mismatch {r.game_id}')
 o=float(s['spread_odds_american']); k=float(s['spread_kelly']); return {'spread_result':res,'spread_flat_profit_units':profit(res,o,1),'spread_kelly_risk_units':k,'spread_kelly_profit_units':profit(res,o,k)}
def grade_total(r,s):
 a=clean(r.actual_total_result).upper(); res='PUSH' if a=='PUSH' else 'WIN' if s['total_selection']==a else 'LOSS'; o=float(s['total_odds_american']); k=float(s['total_kelly']); return {'total_result':res,'total_flat_profit_units':profit(res,o,1),'total_kelly_risk_units':k,'total_kelly_profit_units':profit(res,o,k)}
def ident(r):return {'season':int(float(r.season)),'week':int(float(r.week)),'game_id':gid(r.game_id),'away_team':clean(r.away_team),'home_team':clean(r.home_team)}

def summarize(df,market,season):
 pre={'moneyline':'ml','spread':'spread','total':'total'}[market]; rr=df[f'{pre}_result'].astype(str) if len(df) else pd.Series(dtype=str); w=int((rr=='WIN').sum()); l=int((rr=='LOSS').sum()); p=int((rr=='PUSH').sum()); b=len(df); d=w+l; fn=float(pd.to_numeric(df[f'{pre}_flat_profit_units'],errors='coerce').fillna(0).sum()) if b else 0.; kr=float(pd.to_numeric(df[f'{pre}_kelly_risk_units'],errors='coerce').fillna(0).sum()) if b else 0.; kn=float(pd.to_numeric(df[f'{pre}_kelly_profit_units'],errors='coerce').fillna(0).sum()) if b else 0.; return {'season':season,'market':market,'bets':b,'wins':w,'losses':l,'pushes':p,'decisions':d,'win_rate_pct':100*w/d if d else np.nan,'flat_risk_units':float(b),'flat_net_units':fn,'flat_roi_pct':100*fn/b if b else np.nan,'kelly_risk_units':kr,'kelly_net_units':kn,'kelly_roi_pct':100*kn/kr if kr>0 else np.nan}

def main():
 step13=load_module(STEP13_PATH,'bt13'); step14=load_module(STEP14_PATH,'bt14'); sel=load_module(SELECTIONS_PATH,'btsel'); settings=read_yaml(SETTINGS_PATH); doc=read_yaml(MARKETS_PATH); schema=step13.load_schema(); seasons=[int(x) for x in schema['training_seasons']]; backtest_schema=dict(schema); backtest_schema.pop('input_sha256',None); raw=step13.read_inputs(backtest_schema,seasons); step13.validate_results(raw); ordered=step13.build_chronology(raw); X=step13.prepare_feature_matrix(ordered,schema); ym=pd.to_numeric(ordered.margin).astype(float); yt=pd.to_numeric(ordered.total_points).astype(float); held=ordered[pd.to_numeric(ordered.season).astype(int).isin(seasons[1:])]; groups=list(held.groupby(['season','week','gameday','gametime'],sort=False,dropna=False)); cats=list(schema['categorical_feature_indices']); params=dict(schema['model_params']); prior=[]; preds=[]; probs=[]; ml=[]; sp=[]; to=[]
 for n,(_,g) in enumerate(groups,1):
  pos=g.index.to_numpy(int); train=np.arange(int(pos.min()),dtype=int); mm=step13.train_regressor(X.iloc[train],ym.iloc[train],cats,params); tm=step13.train_regressor(X.iloc[train],yt.iloc[train],cats,params); pms=step13.predict_regressor(mm,X.loc[pos],cats); pts=step13.predict_regressor(tm,X.loc[pos],cats); gf=pd.DataFrame([pred_row(r,float(pms[j]),float(pts[j]),len(train)) for j,(_,r) in enumerate(ordered.loc[pos].iterrows())],columns=PREDICTION_OUTPUT_COLUMNS); hist=pd.concat(prior,ignore_index=True) if prior else gf.iloc[:0].copy(); cc=cals(step14,hist)
  for _,r in gf.iterrows():
   d=r.to_dict(); hp=cp(step14,cc['ml'],float(r.predicted_margin)); cv=cp(step14,cc['spread'],float(r.predicted_margin)-float(r.spread_line)); ov=cp(step14,cc['total'],float(r.predicted_total)-float(r.total_line))
   for a,b,p in [('home_win_probability','away_win_probability',hp),('home_cover_probability','away_cover_probability',cv),('over_probability','under_probability',ov)]:d[a]=p if p is not None else np.nan; d[b]=1-p if p is not None else np.nan
   for pre,key in [('ml','ml'),('spread','spread'),('total','total')]:c=cc[key]; d[f'{pre}_calibration_rows']=c[0]; d[f'{pre}_calibration_intercept']=c[1]; d[f'{pre}_calibration_slope']=c[2]; d[f'{pre}_calibration_status']=c[3]
   pr=pd.Series(d); probs.append(d); i=ident(pr); s=evaluate_single_market(pr,'moneyline',settings,doc,sel)
   if int(s.get('ml_selected',0))==1:ml.append({**i,**{k:s[k] for k in ['ml_selection','ml_odds_american','ml_model_probability','ml_implied_probability','ml_edge','ml_ev','ml_kelly']},**grade_moneyline(pr,s)})
   s=evaluate_single_market(pr,'spread',settings,doc,sel)
   if int(s.get('spread_selected',0))==1:sp.append({**i,**{k:s[k] for k in ['spread_selection','spread_line','spread_odds_american','spread_model_probability','spread_implied_probability','spread_edge','spread_ev','spread_kelly']},**grade_spread(pr,s)})
   s=evaluate_single_market(pr,'total',settings,doc,sel)
   if int(s.get('total_selected',0))==1:to.append({**i,**{k:s[k] for k in ['total_selection','total_line','total_odds_american','total_model_probability','total_implied_probability','total_edge','total_ev','total_kelly']},**grade_total(pr,s)})
  prior.append(gf); preds.extend(gf.to_dict('records'))
  if n==1 or n%10==0:print(f'group {n}/{len(groups)} prior_oos={len(hist)}')
 pdf=pd.DataFrame(preds,columns=PREDICTION_OUTPUT_COLUMNS); qdf=pd.DataFrame(probs,columns=PREDICTION_OUTPUT_COLUMNS+PROBABILITY_COLUMNS+CAL_COLS); a=pd.DataFrame(ml,columns=MONEYLINE_PICK_COLUMNS); b=pd.DataFrame(sp,columns=SPREAD_PICK_COLUMNS); c=pd.DataFrame(to,columns=TOTAL_PICK_COLUMNS); atomic_write_csv(pdf,PREDICTIONS_PATH); atomic_write_csv(qdf,PROBABILITIES_PATH); atomic_write_csv(a,MONEYLINE_PICKS_PATH); atomic_write_csv(b,SPREAD_PICKS_PATH); atomic_write_csv(c,TOTAL_PICKS_PATH)
 frames={'moneyline':a,'spread':b,'total':c}; ys=sorted(set().union(*[set(pd.to_numeric(x.season).astype(int)) for x in frames.values() if len(x)])); sr=[summarize(x[pd.to_numeric(x.season)==y],m,y) for y in ys for m,x in frames.items()]; overall=[summarize(x,m,'ALL') for m,x in frames.items()]; atomic_write_csv(pd.DataFrame(sr,columns=SUMMARY_COLUMNS),SUMMARY_SEASON_PATH); atomic_write_csv(pd.DataFrame(overall,columns=SUMMARY_COLUMNS),SUMMARY_OVERALL_PATH)
 histsha={f'historical_core_{s}.csv':sha(NFL_ROOT/'training'/f'historical_core_{s}.csv') for s in seasons}; meta={'created_utc':datetime.now(timezone.utc).isoformat(),'backtest_type':'CURRENT_STORED_PIPELINE_REPLAY','leakage_warning':'Static enrichment-rule features contain multi-season outcome-derived statistics; ROI is diagnostic replay, not fully leakage-free validation.','write_boundary':'models/backtest','training_seasons':seasons,'source_sha256':{'step13.py':sha(STEP13_PATH),'step14_calibration.py':sha(STEP14_PATH),'selections.py':sha(SELECTIONS_PATH),'settings.yaml':sha(SETTINGS_PATH),'markets.yaml':sha(MARKETS_PATH),'step11_feature_schema.json':sha(SCHEMA_PATH),**histsha},'rows':{'predictions':len(pdf),'moneyline_picks':len(a),'spread_picks':len(b),'total_picks':len(c)}}; atomic_write_json(meta,METADATA_PATH); print('Backtest complete')
if __name__=='__main__':
 try:main()
 except Exception as e:print(f'ERROR: {e}',file=sys.stderr);raise
