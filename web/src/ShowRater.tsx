import { useEffect, useState } from "react";
import { api } from "./api";

const STEPS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];

export default function ShowRater() {
  const [q, setQ] = useState("");
  const [data, setData] = useState<any>(null);
  const [hover, setHover] = useState<Record<number, number>>({});

  const load = (query = q) => api.showQueue(query).then(setData);
  useEffect(() => { load(""); }, []);

  const rate = async (id: number, rating: number | null, status: string) => {
    await api.rateShow(id, rating, status);
    setData({ ...data, items: data.items.filter((x: any) => x.tmdb_id !== id),
      n_rated: data.n_rated + (status === "rated" ? 1 : 0) });
    if (data.items.length <= 4) load();
  };

  return (
    <section className="rater">
      <div className="rater-head">
        <b>Rate shows you've seen</b>
        <span className="muted">{data?.n_rated ?? 0} rated this way · retrain from the Scoreboard tab to use them</span>
        <input placeholder="Find a show…" value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && load()} />
      </div>
      <div className="rater-grid">
        {data?.items.map((s: any) => (
          <div className="rate-card" key={s.tmdb_id}>
            {s.poster ? <img src={s.poster} alt="" loading="lazy" /> : <div className="noimg">{s.title}</div>}
            <div className="t">{s.title} <span className="muted">{s.year}</span></div>
            <div className="starpick" onMouseLeave={() => setHover({ ...hover, [s.tmdb_id]: 0 })}>
              {STEPS.map((v) => (
                <span key={v} className={`half ${v % 1 ? "r" : "l"} ${(hover[s.tmdb_id] || 0) >= v ? "on" : ""}`}
                  onMouseEnter={() => setHover({ ...hover, [s.tmdb_id]: v })} onClick={() => rate(s.tmdb_id, v, "rated")} />
              ))}
              <span className="val">{hover[s.tmdb_id] ? hover[s.tmdb_id].toFixed(1) : ""}</span>
            </div>
            <button className="link" onClick={() => rate(s.tmdb_id, null, "not_seen")}>Haven't seen</button>
          </div>
        ))}
      </div>
    </section>
  );
}
