import { useEffect, useRef, useState } from "react";
import { api, Card, stars } from "./api";

const STEPS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];
const today = () => new Date().toLocaleDateString("en-CA"); // YYYY-MM-DD in local time

export type Logged = { id: number; rating: number | null; liked: boolean; retraining: boolean; left: number };

type Props = { card: Card; queryId: string | null | undefined; onClose: () => void; onSaved: (l: Logged) => void };

export default function LogModal({ card, queryId, onClose, onSaved }: Props) {
  const [rating, setRating] = useState<number | null>(null);
  const [hover, setHover] = useState(0);
  const [liked, setLiked] = useState(false);
  const [dated, setDated] = useState(true);
  const [date, setDate] = useState(today());
  const [rewatch, setRewatch] = useState(false);
  const [review, setReview] = useState("");
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState("");
  const box = useRef<HTMLDivElement>(null);

  const save = async () => {
    setSaving(true); setErr("");
    try {
      const r: any = await api.log({
        kind: card.kind, tmdb_id: card.tmdb_id, rating, liked, rewatch, review,
        watched_date: dated ? date : null, query_id: queryId,
      });
      onSaved({ id: r.id, rating, liked, retraining: r.retraining, left: Math.max(0, r.retrain_every - r.new_since_train) });
    } catch (e: any) { setErr(String(e.message || e)); setSaving(false); }
  };

  useEffect(() => {
    box.current?.focus();
    const k = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) save();
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  });

  const shown = hover || rating || 0;
  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="logbox" ref={box} tabIndex={-1} role="dialog" aria-label={`Log ${card.title}`}>
        <button className="x" onClick={onClose} aria-label="Close">×</button>
        <div className="logposter">
          {card.poster ? <img src={card.poster} alt="" /> : <div className="noimg">{card.title}</div>}
        </div>
        <div className="logform">
          <div className="logkicker">I watched…</div>
          <h2>{card.title} <span className="muted">{card.year}</span></h2>

          <div className="logrow">
            <label className="check"><input type="checkbox" checked={dated} onChange={(e) => setDated(e.target.checked)} />
              {card.kind === "tv" ? "Finished on" : "Watched on"}</label>
            <input type="date" value={date} max={today()} disabled={!dated} onChange={(e) => setDate(e.target.value)} />
            <label className="check"><input type="checkbox" checked={rewatch} onChange={(e) => setRewatch(e.target.checked)} />
              I've {card.kind === "tv" ? "seen" : "watched"} this before</label>
          </div>

          <textarea placeholder="Add a review…" value={review} onChange={(e) => setReview(e.target.value)} rows={4} />

          <div className="logrow rate">
            <div>
              <div className="lbl">Rating</div>
              <div className="starpick big" onMouseLeave={() => setHover(0)}>
                {STEPS.map((v) => (
                  <span key={v} className={`half ${v % 1 ? "l" : "r"} ${shown >= v ? "on" : ""}`}
                    onMouseEnter={() => setHover(v)} onClick={() => setRating(rating === v ? null : v)} />
                ))}
                <span className="val">{shown ? shown.toFixed(1) : ""}</span>
              </div>
            </div>
            <div>
              <div className="lbl">Like</div>
              <button className={`heart ${liked ? "on" : ""}`} onClick={() => setLiked(!liked)} aria-pressed={liked}
                aria-label="Like">{liked ? "♥" : "♡"}</button>
            </div>
            <button className="primary save" onClick={save} disabled={saving}>{saving ? "Saving…" : "Save"}</button>
          </div>
          <div className="muted lognote">
            {rating ? `Predicted ${card.predicted.toFixed(1)}. Your rating trains the model and counts in the live test.`
              : "No rating: it's logged as watched and hidden from picks, but won't train the model."}
            {" "}Saved here only; Letterboxd has no way for apps to post.{" "}
            {card.letterboxd_url && <a href={card.letterboxd_url} target="_blank" rel="noreferrer">Open on Letterboxd ↗</a>}
          </div>
          {err && <div className="banner bad">{err}</div>}
        </div>
      </div>
    </div>
  );
}
