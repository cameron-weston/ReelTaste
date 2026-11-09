import { useEffect, useState } from "react";
import { api, stars } from "./api";

function Bars({ rows, label, value, max, fmt }: { rows: any[]; label: (r: any) => string; value: (r: any) => number; max?: number; fmt?: (r: any) => string }) {
  const m = max ?? Math.max(...rows.map(value), 1);
  return (
    <div className="bars">
      {rows.map((r, i) => (
        <div className="bar" key={i}>
          <span className="bl">{label(r)}</span>
          <span className="bt"><span style={{ width: `${(value(r) / m) * 100}%` }} /></span>
          <span className="bv">{fmt ? fmt(r) : value(r)}</span>
        </div>
      ))}
    </div>
  );
}

export default function Taste() {
  const [t, setT] = useState<any>(null);
  const [err, setErr] = useState("");
  useEffect(() => { api.taste().then(setT).catch((e) => setErr(String(e))); }, []);
  if (err) return <div className="banner bad">{err}</div>;
  if (!t) return <div className="empty">Loading…</div>;
  const hist = Object.entries(t.histogram).map(([k, v]) => ({ k: +k, v: v as number }));

  return (
    <div className="taste">
      <div className="stats">
        {[["Watched", t.counts.watched], ["Rated", t.counts.rated], ["Reviews", t.counts.reviews], ["Watchlist", t.counts.watchlist],
          ["Rewatched", t.counts.rewatched], ["Average", t.mean.toFixed(2) + "★"], ["4★ or higher", Math.round(t.share_4plus * 100) + "%"]]
          .map(([k, v]) => <div className="stat" key={k as string}><b>{v}</b><span>{k}</span></div>)}
      </div>

      <div className="cols">
        <section><h2>How you rate</h2>
          <Bars rows={hist} label={(r) => stars(r.k) || "½"} value={(r) => r.v} /></section>
        <section><h2>By year logged</h2>
          <p className="muted">{t.backfill_n} ratings entered on 4 Dec 2020 are left out (backfill).</p>
          <Bars rows={t.by_year} label={(r) => r.year} value={(r) => r.share_4plus} max={1}
            fmt={(r) => `${Math.round(r.share_4plus * 100)}% 4★+ · ${r.n}`} /></section>
        <section><h2>By decade released</h2>
          <Bars rows={t.by_decade} label={(r) => `${r.decade}s`} value={(r) => r.mean - 2} max={3}
            fmt={(r) => `${r.mean.toFixed(2)}★ · ${r.n}`} /></section>
      </div>

      {t.fingerprint && (
        <div className="cols two">
          <section><h2>Pulls your ratings up</h2>
            <div className="chips">{t.fingerprint.loves.map((x: any) => <span key={x.kind + x.name} className="chip up"><i>{x.kind}</i> {x.name}</span>)}</div></section>
          <section><h2>Pulls your ratings down</h2>
            <div className="chips">{t.fingerprint.avoids.map((x: any) => <span key={x.kind + x.name} className="chip down"><i>{x.kind}</i> {x.name}</span>)}</div></section>
        </div>
      )}

      {t.directors && (
        <div className="cols">
          <section><h2>Most-watched directors</h2>
            <Bars rows={t.directors} label={(r) => r.name} value={(r) => r.n} fmt={(r) => `${r.n} · ${r.mean.toFixed(1)}★`} /></section>
          <section><h2>Genres, best to worst</h2>
            <Bars rows={t.genres} label={(r) => r.name} value={(r) => r.mean - 2} max={3} fmt={(r) => `${r.mean.toFixed(2)}★ · ${r.n}`} /></section>
          <section><h2>Languages</h2>
            <Bars rows={t.languages} label={(r) => r.name.toUpperCase()} value={(r) => r.n} fmt={(r) => `${r.n} · ${r.mean.toFixed(1)}★`} /></section>
        </div>
      )}

      {t.above_crowd && (
        <div className="cols two">
          <section><h2>You rate above the crowd</h2>
            <p className="muted">Crowd = TMDB average out of 5. On average you sit {t.crowd_gap_mean > 0 ? "+" : ""}{t.crowd_gap_mean} from it.</p>
            <table><tbody>{t.above_crowd.map((g: any) => <tr key={g.name}><td>{g.name} <span className="muted">{g.year}</span></td><td>{g.yours}★</td><td className="muted">{g.crowd}</td><td className="up">+{g.gap.toFixed(1)}</td></tr>)}</tbody></table></section>
          <section><h2>You rate below the crowd</h2>
            <table><tbody>{t.below_crowd.map((g: any) => <tr key={g.name}><td>{g.name} <span className="muted">{g.year}</span></td><td>{g.yours}★</td><td className="muted">{g.crowd}</td><td className="down">{g.gap.toFixed(1)}</td></tr>)}</tbody></table></section>
        </div>
      )}
      {!t.directors && <div className="note">Directors, genres, crowd comparison and your taste fingerprint appear once TMDB data is in.</div>}
    </div>
  );
}
