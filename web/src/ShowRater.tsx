import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import StarPicker from "./StarPicker";

export default function ShowRater() {
  const [q, setQ] = useState("");
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState("");

  const left = useRef<number | null>(null);
  const pending = useRef(new Set<number>()); // rated but maybe not saved yet: never let a refill bring them back
  const submitted = useRef(""); // refills use the last search you ran, not unsubmitted typing
  const reqId = useRef(0);
  const load = (query = submitted.current) => {
    submitted.current = query;
    const id = ++reqId.current;
    return api.showQueue(query)
      .then((d) => { if (id === reqId.current) setData({ ...d, items: d.items.filter((x: any) => !pending.current.has(x.tmdb_id)) }); })
      .catch((e) => setErr(String(e.message || e)));
  };
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
        <input placeholder="Find a show…" value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={(e) => e.key === "Enter" && load(q)} />
      </div>
      {err && <div className="banner bad">{err}</div>}
      <div className="rater-grid">
        {data?.items.map((s: any) => (
          <div className="rate-card" key={s.tmdb_id}>
            {s.poster ? <img src={s.poster} alt="" loading="lazy" /> : <div className="noimg">{s.title}</div>}
            <div className="t">{s.title} <span className="muted">{s.year}</span></div>
            <StarPicker value={null} onChange={(v) => v && rate(s.tmdb_id, v, "rated")} size={20} label={`Rate ${s.title}`} showValue={false} commitOnEnter />
            <button className="link" onClick={() => rate(s.tmdb_id, null, "not_seen")}>Haven't seen</button>
          </div>
        ))}
      </div>
    </section>
  );
}
