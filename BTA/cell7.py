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

res = pd.DataFrame(records)
summary = res.groupby("model").agg(["mean","std"]).round(3)
summary.to_csv(f"{CFG.results_dir}/benchmark_summary.csv")
display(summary[["roc_auc","pr_auc","p_at_k","fit_s","score_s"]])
