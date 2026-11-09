import { useEffect, useState } from "react";
import { api, Status } from "./api";

const METRICS: [string, string, boolean][] = [
  ["rmse", "RMSE", false], ["mae", "MAE", false], ["spearman", "Rank corr.", true], ["auc", "AUC 4★+", true], ["hit_top25", "4★+ in top quarter", true],
];

const SRC: Record<string, string> = { model: "Ranker", llm: "Ranker + Claude", crowd: "Crowd average", popular: "Popularity", svd: "Standard CF" };

export default function Scoreboard({ status, onBuild, llm }: { status: Status | null; onBuild: () => void; llm: any }) {
  const [s, setS] = useState<any>(null);
  const [msg, setMsg] = useState("");
  const load = () => api.scoreboard().then(setS);
  useEffect(() => { load(); }, [status?.built_at]);

  const build = async (full: boolean) => { await api.build(full); onBuild(); };
  const refresh = async () => { const r = await api.refresh(); setMsg(`${r.new_rss_entries} new RSS entries${r.retraining ? " — retraining" : ""}`); onBuild(); };

  const off = s?.offline;
  const live = s?.live;
  return (
    <div className="score">
      <section className="ops">
        <button onClick={refresh}>Check Letterboxd RSS</button>
        <button onClick={() => build(false)} disabled={status?.job.running}>Retrain</button>
        <button onClick={() => build(true)} disabled={status?.job.running}>Full rebuild</button>
        <span className="muted">{msg}</span>
        <span className="muted">Claude: {llm?.ready ? `ready (${llm.backend})` : `not ready — ${llm?.detail ?? "?"}`}</span>
      </section>
      {status?.job.running || status?.job_log?.length ? <pre className="log">{status?.job_log.join("\n")}</pre> : null}

      <h2>Offline test</h2>
      {!off ? <div className="empty">No results yet.</div> : <>
        <p className="muted">Trained on your {off.n_train} earliest ratings, predicting the {off.n_test} most recent (from {off.test_from}). Run {off.ts}.
          Lower is better for RMSE/MAE; higher for the rest. Every baseline is fitted to your ratings the same way.</p>
        {Object.entries(off.table).map(([row, t]: any) => {
          const names = Object.keys(t);
          const best: Record<string, string> = {};
          for (const [m, , hi] of METRICS) best[m] = names.reduce((a, b) => (hi ? t[b][m] > t[a][m] : t[b][m] < t[a][m]) ? b : a);
          const bs = off.bootstrap?.[row];
          return (
            <section key={row} className="tablewrap">
              <h3>{row} <span className="muted">n={t[names[0]].n} · {Math.round(t[names[0]].base_rate * 100)}% rated 4★+</span></h3>
              <table className="metrics"><thead><tr><th></th>{METRICS.map(([, l]) => <th key={l}>{l}</th>)}</tr></thead>
                <tbody>{names.map((n) => (
                  <tr key={n} className={n.includes("ours") ? "ours" : ""}><td>{n}</td>
                    {METRICS.map(([m]) => <td key={m} className={best[m] === n ? "best" : ""}>{t[n][m].toFixed(m.startsWith("hit") ? 2 : 3)}</td>)}</tr>))}
                </tbody></table>
              {bs && <p className="muted">Beats the best baseline ({bs.best_baseline}) on RMSE in {Math.round(bs.p_better * 100)}% of 2,000 bootstrap resamples.</p>}
            </section>);
        })}
        {off.watchlist && <section className="tablewrap"><h3>Watchlist check <span className="muted">do films you flagged yourself rank high among {off.watchlist.n_candidates.toLocaleString()} unseen films? ({off.watchlist.n_watchlist} in catalog)</span></h3>
          <table className="metrics"><thead><tr><th></th><th>Median percentile</th><th>In top 500</th><th>In top 2,000</th></tr></thead>
            <tbody>{Object.entries(off.watchlist.methods).map(([n, v]: any) => <tr key={n} className={n.includes("ours") ? "ours" : ""}><td>{n}</td><td>{v.median_percentile.toFixed(1)}</td><td>{v.in_top_500}</td><td>{v.in_top_2000}</td></tr>)}</tbody></table></section>}
        <p className="muted">Blend chosen by cross-validation: {Object.entries(off.learners || {}).map(([k, v]) => `${k} → ${v}`).join(", ")}</p>
      </>}

      <h2>Live test</h2>
      <p className="muted">Every pick shown is logged, along with what the baselines would have shown. When a later RSS entry or export shows you watched one, its rating counts here.
        The bar to beat is your 4★+ rate since 2023: <b>{live?.bar ? Math.round(live.bar * 100) + "%" : "—"}</b>.</p>
      {!live || !Object.keys(live.sources).length ? <div className="empty">Nothing logged yet. Ask for recommendations, watch some, log them on Letterboxd.</div> :
        <table className="metrics"><thead><tr><th>Source</th><th>Recommended</th><th>Watched</th><th>Rated 4★+</th><th>Hit rate</th><th>Avg rating</th></tr></thead>
          <tbody>{Object.values(live.sources).map((v: any) => (
            <tr key={v.kind + v.source}><td>{v.kind === "tv" ? "Shows · " : "Films · "}{SRC[v.source] ?? v.source}</td><td>{v.recommended}</td><td>{v.watched}</td><td>{v.hits}</td>
              <td className={v.hit_rate != null && live.bar != null && v.hit_rate > live.bar ? "best" : ""}>{v.hit_rate != null ? Math.round(v.hit_rate * 100) + "%" : "—"}</td><td>{v.mean_rating ?? "—"}</td></tr>))}
          </tbody></table>}
      {live?.feedback && Object.keys(live.feedback).length > 0 && <p className="muted">Feedback so far: {Object.entries(live.feedback).map(([k, v]) => `${k.replace("_", " ")} ${v}`).join(" · ")}</p>}
      <Blocks />
    </div>
  );
}

function Blocks() {
  const [b, setB] = useState<{ value: string; label: string }[]>([]);
  useEffect(() => { api.blocks().then(setB); }, []);
  if (!b.length) return null;
  return <section><h2>Block list</h2><div className="chips">{b.map((x) => (
    <span className="chip" key={x.value}>{x.value.split(":")[0]}: {x.label}
      <button className="link" onClick={async () => { await api.unblock(x.value); setB(b.filter((y) => y.value !== x.value)); }}>×</button></span>))}</div></section>;
}
