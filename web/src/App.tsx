import { useEffect, useState } from "react";
import { api, Status } from "./api";
import Recommend from "./Recommend";
import Taste from "./Taste";
import Scoreboard from "./Scoreboard";

type Tab = "films" | "shows" | "taste" | "score";
const TABS: [Tab, string][] = [["films", "Films"], ["shows", "Shows"], ["taste", "Your taste"], ["score", "Scoreboard"]];

export default function App() {
  const [tab, setTab] = useState<Tab>(() => (localStorage.getItem("tab") as Tab) || "films");
  const [status, setStatus] = useState<Status | null>(null);
  const [llm, setLlm] = useState<{ ready: boolean; detail: string; backend: string } | null>(null);

  const refreshStatus = () => api.status().then(setStatus).catch(() => {});
  useEffect(() => {
    refreshStatus();
    api.llm().then(setLlm).catch(() => setLlm({ ready: false, detail: "unreachable", backend: "?" }));
    const t = setInterval(refreshStatus, 4000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => { try { localStorage.setItem("tab", tab); } catch {} }, [tab]);

  return (
    <div className="app">
      <header>
        <div className="brand">Reel&nbsp;Taste</div>
        <nav>
          {TABS.map(([k, label]) => (
            <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{label}</button>
          ))}
        </nav>
        <div className="status">
          <span className={`pill ${status?.tmdb_key ? "ok" : "bad"}`} title="TMDB key in .env">TMDB</span>
          <span className={`pill ${llm?.ready ? "ok" : "bad"}`} title={llm?.detail}>Claude</span>
          {status?.job.running ? <span className="pill busy">building…</span> :
            <span className="muted" title={`RSS checked ${status?.rss_last_fetch ?? "never"}`}>model {status?.built_at ?? "not built"}</span>}
        </div>
      </header>
      {status && !status.tmdb_key && (
        <div className="banner">Add your free TMDB key to <code>.env</code> as <code>TMDB_API_KEY=…</code>, then press <b>Full rebuild</b> on the Scoreboard tab.</div>
      )}
      {status?.job.error && <pre className="banner bad">{status.job.error}</pre>}
      <main>
        {tab === "films" && <Recommend kind="movie" llmReady={!!llm?.ready} status={status} />}
        {tab === "shows" && <Recommend kind="tv" llmReady={!!llm?.ready} status={status} />}
        {tab === "taste" && <Taste />}
        {tab === "score" && <Scoreboard status={status} onBuild={refreshStatus} llm={llm} />}
      </main>
    </div>
  );
}
