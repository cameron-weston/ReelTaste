import { useEffect, useRef, useState } from "react";
import { api } from "./api";

const STEPS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];

export default function ShowRater() {
  const [q, setQ] = useState("");
  const [data, setData] = useState<any>(null);
  const [hover, setHover] = useState<Record<number, number>>({});
  const [err, setErr] = useState("");

  const left = useRef<number | null>(null);
  const pending = useRef(new Set<number>()); // rated but maybe not saved yet: never let a refill bring them back
  const load = (query = q) => api.showQueue(query)
    .then((d) => setData({ ...d, items: d.items.filter((x: any) => !pending.current.has(x.tmdb_id)) }))
    .catch((e) => setErr(String(e.message || e)));
  useEffect(() => { load(""); }, []);
  useEffect(() => { left.current = data?.items?.length ?? null; }, [data]);

  const rate = async (id: number, rating: number | null, status: string) => {
    const bump = status === "rated" ? 1 : 0;
    const card = data?.items.find((x: any) => x.tmdb_id === id);
    pending.current.add(id);
    setData((d: any) => ({ ...d, items: d.items.filter((x: any) => x.tmdb_id !== id), n_rated: d.n_rated + bump }));
    try {
      await api.rateShow(id, rating, status);
      setErr("");
    } catch (e: any) {
      pending.current.delete(id);
      setData((d: any) => ({ ...d, n_rated: d.n_rated - bump,
        items: card && !d.items.some((x: any) => x.tmdb_id === id) ? [card, ...d.items] : d.items }));
      setErr(`Couldn't save that rating: ${e.message || e}`);
      return;
    }
    if ((left.current ?? 99) <= 5) load();
  };

  return (
    <section className="rater">
      <div className="rater-head">
        <b>Rate shows you've seen</b>
        <span className="muted">{data?.n_rated ?? 0} rated this way · the model retrains after 10 new ratings</span>
        <input placeholder="Find a show…" value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && load()} />
      </div>
      {err && <div className="banner bad">{err}</div>}
      <div className="rater-grid">
        {data?.items.map((s: any) => (
          <div className="rate-card" key={s.tmdb_id}>
            {s.poster ? <img src={s.poster} alt="" loading="lazy" /> : <div className="noimg">{s.title}</div>}
            <div className="t">{s.title} <span className="muted">{s.year}</span></div>
            <div className="starpick" onMouseLeave={() => setHover((h) => ({ ...h, [s.tmdb_id]: 0 }))}>
              {STEPS.map((v) => (
                <span key={v} className={`half ${v % 1 ? "l" : "r"} ${(hover[s.tmdb_id] || 0) >= v ? "on" : ""}`}
                  onMouseEnter={() => setHover((h) => ({ ...h, [s.tmdb_id]: v }))} onClick={() => rate(s.tmdb_id, v, "rated")} />
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
