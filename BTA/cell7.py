from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm.auto import tqdm
import time

def evaluate(scores, y, k):
    order = np.argsort(-scores)
    return dict(roc_auc=roc_auc_score(y, scores),
                pr_auc=average_precision_score(y, scores),
                p_at_k=float(y[order[:k]].mean()))

RUN_TABPFN_IN_BENCHMARK = False
TABPFN_SEEDS = (0,)

records = []
dets_per_seed = len(make_detectors(0))
pbar = tqdm(total=len(CFG.seeds) * dets_per_seed, desc="Baselines+AE", unit="fit")
for seed in CFG.seeds:
    te = inject_anomalies(test_clean, rate=CFG.injection_rate, seed=seed)
    Xtr, Xte, y = transform(train), transform(te), te.is_anomaly.values
    for det in make_detectors(seed):
        pbar.set_postfix_str(f"seed {seed} | {det.name}")
        t0 = time.perf_counter(); det.fit(Xtr); t_fit = time.perf_counter() - t0
        t0 = time.perf_counter(); s = det.score(Xte); t_score = time.perf_counter() - t0
        records.append(dict(seed=seed, model=det.name, fit_s=t_fit, score_s=t_score,
                            **evaluate(s, y, CFG.precision_at_k)))
        pbar.update(1)
pbar.close()

if TABPFN_OK and RUN_TABPFN_IN_BENCHMARK:
    for seed in tqdm(TABPFN_SEEDS, desc="TabPFN(global)", unit="seed"):
        te = inject_anomalies(test_clean, rate=CFG.injection_rate, seed=seed)
        Xtr, Xte, y = transform(train), transform(te), te.is_anomaly.values
        t0 = time.perf_counter()
        s = tabpfn_scores(Xtr[:CFG.context_cap], Xte)
        records.append(dict(seed=seed, model="TabPFN(global)", fit_s=0.0,
                            score_s=time.perf_counter() - t0,
                            **evaluate(s, y, CFG.precision_at_k)))



#--------------

if CFG.run_tabpfn:
    try:
        import time as _t
        def vlog(msg):
            print(f"[TabPFN {_t.strftime('%H:%M:%S')}] {msg}", flush=True)

        vlog("importing tabpfn packages...")
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        from tabpfn_extensions import unsupervised as tab_unsup
        import torch
        vlog(f"imports OK | torch {torch.__version__} | "
             f"device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU (expect slow runs)'}")

        vlog("loading V2 weights (first ever run downloads ~300MB from HuggingFace; later runs read local cache)...")
        t0 = _t.perf_counter()
        _clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
        _reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        vlog(f"V2 weights ready in {_t.perf_counter()-t0:.1f}s (cached locally - no data leaves this machine during inference)")

        def tabpfn_scores(ctx_X, test_X, tag=""):
            t0 = _t.perf_counter()
            vlog(f"{tag}fitting on context: {len(ctx_X)} rows x {ctx_X.shape[1]} features "
                 f"(fits one model per feature - {ctx_X.shape[1]} internal fits)")
            m = tab_unsup.TabPFNUnsupervisedModel(_clf, _reg)
            m.fit(torch.as_tensor(np.asarray(ctx_X), dtype=torch.float32))
            vlog(f"{tag}scoring {len(test_X)} rows...")
            s = m.outliers(torch.as_tensor(np.asarray(test_X), dtype=torch.float32))
            s = np.asarray(s, dtype=float).ravel()
            vlog(f"{tag}done in {_t.perf_counter()-t0:.1f}s")
            return np.nan_to_num(-s if np.nanmedian(s) > 0 else s, nan=0.0)  # higher = more anomalous; VERIFY sign

        vlog(f"STEP 1/3: threshold calibration on last {CFG.tabpfn_context_cap} train rows "
             f"(target flag rate ~{100-CFG.flag_percentile:.1f}%)")
        thr_sample = train.sort_values("txn_date").tail(CFG.tabpfn_context_cap)
        s_tr = tabpfn_scores(transform(thr_sample), transform(thr_sample), tag="calib | ")
        tab_thr = np.percentile(s_tr, CFG.flag_percentile)
        vlog(f"threshold = {tab_thr:.4f} (score above this -> anomaly)")

        n_days = test.txn_date.nunique()
        vlog(f"STEP 2/3: per-day scoring of {len(test)} test rows across {n_days} days "
             f"(context per day = history of that day's accounts, cap {CFG.tabpfn_context_cap})")
        scores_all = np.full(len(test), np.nan)
        day_times = []
        for d in tqdm(sorted(test.txn_date.unique()), desc="TabPFN per-day", unit="day"):
            day_rows = test[test.txn_date == d]
            ctx = build_context(day_rows, train, cap=CFG.tabpfn_context_cap)
            if len(ctx) < CFG.tabpfn_min_context:
                vlog(f"  {pd.Timestamp(d).date()}: SKIPPED (only {len(ctx)} context rows < min {CFG.tabpfn_min_context})")
                continue
            t0 = _t.perf_counter()
            scores_all[day_rows.index] = tabpfn_scores(transform(ctx), transform(day_rows),
                                                       tag=f"  {pd.Timestamp(d).date()} | ")
            day_times.append(_t.perf_counter() - t0)
            if len(day_times) == 1:
                vlog(f"first day took {day_times[0]:.0f}s -> rough ETA for remaining "
                     f"{n_days-1} days: ~{day_times[0]*(n_days-1)/60:.0f} min")

        vlog(f"STEP 3/3: flagging + type attribution")
        scored = ~np.isnan(scores_all)
        flagged_idx = test.index[scored & (scores_all > tab_thr)]
        vlog(f"scored {int(scored.sum())} rows | flagged {len(flagged_idx)} "
             f"({100*len(flagged_idx)/max(int(scored.sum()),1):.1f}%)")
        labels = attributor.label(test.loc[flagged_idx]) if len(flagged_idx) else None
        tab_rows = []
        for i in test.index[scored]:
            rec = dict(model="TabPFN(per_account)", row_id=int(i), score=float(scores_all[i]),
                       threshold=float(tab_thr), is_anomaly=bool(scores_all[i] > tab_thr))
            if rec["is_anomaly"]: rec.update(labels.loc[i].to_dict())
            tab_rows.append(rec)
        long_df = pd.concat([long_df, pd.DataFrame(tab_rows)], ignore_index=True)
        if len(flagged_idx):
            vlog("flagged types: " + str(pd.DataFrame(tab_rows).query("is_anomaly").anomaly_type.value_counts().to_dict()))
        if day_times:
            vlog(f"avg {np.mean(day_times):.0f}s/day over {len(day_times)} days")
        vlog("DONE. Re-run cell 6 to regenerate the analyst export including TabPFN.")
    except Exception as e:
        print("TabPFN failed/unavailable:", e)
else:
    print("TabPFN disabled (CFG.run_tabpfn=False)")

res = pd.DataFrame(records)
summary = res.groupby("model").agg(["mean","std"]).round(3)
summary.to_csv(f"{CFG.results_dir}/benchmark_summary.csv")
display(summary[["roc_auc","pr_auc","p_at_k","fit_s","score_s"]])
