if CFG.run_tabpfn:
    try:
        import time as _t, os, json as _json

        def vlog(msg):
            print(f"[TabPFN {_t.strftime('%H:%M:%S')}] {msg}", flush=True)

        # ---------- config (with fallbacks if not yet in Config) ----------
        CALIB_ROWS = getattr(CFG, "tabpfn_calib_rows", 3000)   # rows for threshold calibration
        CTX_CAP    = CFG.tabpfn_context_cap
        MIN_CTX    = CFG.tabpfn_min_context

        # ---------- imports & device ----------
        vlog("importing tabpfn packages...")
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        from tabpfn_extensions import unsupervised as tab_unsup
        import torch
        vlog(f"imports OK | torch {torch.__version__} | device: "
             f"{'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU (expect slow runs)'}")

        # ---------- V2 weights (commercial-permissive; local inference) ----------
        vlog("loading V2 weights (first ever run downloads ~300MB from HuggingFace; "
             "later runs read local cache; no data leaves this machine during inference)...")
        t0 = _t.perf_counter()
        _clf = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
        _reg = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        vlog(f"V2 weights ready in {_t.perf_counter()-t0:.1f}s")

        # ---------- water-filling context builder (overrides library version) ----------
        def build_context(day_df, history_df, cap=CTX_CAP):
            """History of today's accounts, capped. Over cap: equal share per account,
            unused share from short accounts redistributed to the rest. Newest first."""
            accts = day_df["BankAccountCode"].unique()
            ctx = history_df[history_df["BankAccountCode"].isin(accts)]
            if len(ctx) <= cap:
                return ctx.sort_values("txn_date", ascending=False)
            sizes = ctx["BankAccountCode"].value_counts()
            alloc, budget, left = {}, cap, len(sizes)
            for acct, size in sizes.sort_values().items():   # smallest history first
                take = min(int(size), budget // left)
                alloc[acct] = take
                budget -= take
                left -= 1
            parts = [g.sort_values("txn_date", ascending=False).head(alloc[a])
                     for a, g in ctx.groupby("BankAccountCode")]
            out = pd.concat(parts).sort_values("txn_date", ascending=False)
            vlog(f"  context over cap -> split: { {a: int(v) for a, v in alloc.items()} }")
            return out

        # ---------- scoring wrapper ----------
        def tabpfn_scores(ctx_X, test_X, tag=""):
            t0 = _t.perf_counter()
            vlog(f"{tag}fitting on context: {len(ctx_X)} rows x {ctx_X.shape[1]} features "
                 f"(one internal model per feature = {ctx_X.shape[1]} fits)")
            m = tab_unsup.TabPFNUnsupervisedModel(_clf, _reg)
            m.fit(torch.as_tensor(np.asarray(ctx_X), dtype=torch.float32))
            vlog(f"{tag}scoring {len(test_X)} rows...")
            s = m.outliers(torch.as_tensor(np.asarray(test_X), dtype=torch.float32))
            s = np.asarray(s, dtype=float).ravel()
            vlog(f"{tag}done in {_t.perf_counter()-t0:.1f}s")
            return np.nan_to_num(-s if np.nanmedian(s) > 0 else s, nan=0.0)  # higher = anomalous; VERIFY sign

        # ---------- STEP 1/3: threshold (test accounts' history only, cached) ----------
        thr_file = f"{CFG.results_dir}/tabpfn_threshold.json"
        if os.path.exists(thr_file):
            tab_thr = _json.load(open(thr_file))["threshold"]
            vlog(f"STEP 1/3: loaded cached threshold = {tab_thr:.4f} "
                 f"(delete {thr_file} to recalibrate, e.g. when train data or account mix changes)")
        else:
            test_accts = test["BankAccountCode"].unique()
            thr_sample = (train[train["BankAccountCode"].isin(test_accts)]
                          .sort_values("txn_date", ascending=False)
                          .head(CALIB_ROWS))
            vlog(f"STEP 1/3: calibrating threshold on {len(thr_sample)} rows from "
                 f"{len(test_accts)} test accounts (of {train['BankAccountCode'].nunique()} "
                 f"accounts in train) | target flag rate ~{100-CFG.flag_percentile:.1f}%")
            s_tr = tabpfn_scores(transform(thr_sample), transform(thr_sample), tag="calib | ")
            tab_thr = float(np.percentile(s_tr, CFG.flag_percentile))
            _json.dump({"threshold": tab_thr, "calib_rows": int(len(thr_sample)),
                        "n_test_accounts": int(len(test_accts))}, open(thr_file, "w"))
            vlog(f"threshold = {tab_thr:.4f} (cached to {thr_file})")

        # ---------- STEP 2/3: per-day scoring ----------
        n_days = test.txn_date.nunique()
        vlog(f"STEP 2/3: scoring {len(test)} test rows across {n_days} days "
             f"(context per day = history of that day's accounts, cap {CTX_CAP})")
        scores_all = np.full(len(test), np.nan)
        day_times = []
        for d in tqdm(sorted(test.txn_date.unique()), desc="TabPFN per-day", unit="day"):
            day_rows = test[test.txn_date == d]
            ctx = build_context(day_rows, train, cap=CTX_CAP)
            if len(ctx) < MIN_CTX:
                vlog(f"  {pd.Timestamp(d).date()}: SKIPPED "
                     f"(only {len(ctx)} context rows < min {MIN_CTX})")
                continue
            t0 = _t.perf_counter()
            scores_all[day_rows.index] = tabpfn_scores(
                transform(ctx), transform(day_rows),
                tag=f"  {pd.Timestamp(d).date()} | ")
            day_times.append(_t.perf_counter() - t0)
            if len(day_times) == 1:
                vlog(f"first day took {day_times[0]:.0f}s -> rough ETA for remaining "
                     f"{n_days-1} days: ~{day_times[0]*(n_days-1)/60:.0f} min")

        # ---------- STEP 3/3: flag + type attribution ----------
        vlog("STEP 3/3: flagging + type attribution")
        scored = ~np.isnan(scores_all)
        flagged_idx = test.index[scored & (scores_all > tab_thr)]
        vlog(f"scored {int(scored.sum())} rows | flagged {len(flagged_idx)} "
             f"({100*len(flagged_idx)/max(int(scored.sum()),1):.1f}%)")
        labels = attributor.label(test.loc[flagged_idx]) if len(flagged_idx) else None
        tab_rows = []
        for i in test.index[scored]:
            rec = dict(model="TabPFN(per_account)", row_id=int(i), score=float(scores_all[i]),
                       threshold=float(tab_thr), is_anomaly=bool(scores_all[i] > tab_thr))
            if rec["is_anomaly"]:
                rec.update(labels.loc[i].to_dict())
            tab_rows.append(rec)
        long_df = pd.concat([long_df, pd.DataFrame(tab_rows)], ignore_index=True)
        if len(flagged_idx):
            vlog("flagged types: " + str(pd.DataFrame(tab_rows)
                 .query("is_anomaly").anomaly_type.value_counts().to_dict()))
        if day_times:
            vlog(f"avg {np.mean(day_times):.0f}s/day over {len(day_times)} scored days")
        vlog("DONE. Re-run cell 6 to regenerate the analyst export including TabPFN.")
    except Exception as e:
        print("TabPFN failed/unavailable:", e)
else:
    print("TabPFN disabled (CFG.run_tabpfn=False)")
