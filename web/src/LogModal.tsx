import { useEffect, useRef, useState } from "react";
import { api, Card } from "./api";
import StarPicker from "./StarPicker";
const today = () => new Date().toLocaleDateString("en-CA"); // YYYY-MM-DD in local time

export type Logged = { id: number; rating: number | null; liked: boolean; retraining: boolean; left: number };

type Props = { card: Card; queryId: string | null | undefined; onClose: () => void; onSaved: (l: Logged) => void };

export default function LogModal({ card, queryId, onClose, onSaved }: Props) {
  const [rating, setRating] = useState<number | null>(null);
  const [liked, setLiked] = useState(false);
  const [dated, setDated] = useState(true);
  const [date, setDate] = useState(today());
  const [rewatch, setRewatch] = useState(false);
  const [review, setReview] = useState("");
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState("");
  const box = useRef<HTMLDivElement>(null);
  const busy = useRef(false); // blocks a second save from key repeat or a double press

  const save = async () => {
    if (busy.current) return;
    busy.current = true; setSaving(true); setErr("");
    try {
      const r: any = await api.log({
        kind: card.kind, tmdb_id: card.tmdb_id, rating, liked, rewatch, review,
        watched_date: dated ? date : null, query_id: queryId,
      });
      onSaved({ id: r.id, rating, liked, retraining: r.retraining, left: Math.max(0, r.retrain_every - r.new_since_train) });
    } catch (e: any) {
      setErr(String(e.message || e)); setSaving(false); busy.current = false;
    }
  };
  const saveRef = useRef(save);
  saveRef.current = save;
  const close = () => { if (!busy.current) onClose(); }; // a save in flight finishes before closing
  const closeRef = useRef(close);
  closeRef.current = close;

  // Register once; refs keep the handlers current. Focus returns to the opener on close.
  useEffect(() => {
    // Safari doesn't focus clicked buttons, so the opener may be <body>; the card is the fallback.
    const a = document.activeElement as HTMLElement | null;
    const opener = a && a !== document.body ? a : null;
    box.current?.focus();
    const k = (e: KeyboardEvent) => {
      if (e.key === "Escape") closeRef.current();
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault(); e.stopPropagation();
        if (!e.repeat) saveRef.current();
      }
      if (e.key === "Tab" && box.current) { // keep focus inside the dialog
        const f = box.current.querySelectorAll<HTMLElement>(
          "button:not([disabled]), input:not([disabled]), textarea, a[href], [tabindex='0']");
        const first = f[0], last = f[f.length - 1];
        const a = document.activeElement;
        const atStart = a === first || a === box.current || !box.current.contains(a);
        if (e.shiftKey && atStart) { e.preventDefault(); last.focus(); }
        else if (!e.shiftKey && (a === last || !box.current.contains(a))) { e.preventDefault(); first.focus(); }
      }
    };
    window.addEventListener("keydown", k, true);
    return () => {
      window.removeEventListener("keydown", k, true);
      // After a save the "Watched it!" button is gone; land on the card instead.
      if (opener?.isConnected) opener.focus();
      else document.querySelector<HTMLElement>(`[data-key="${card.key}"]`)?.focus();
    };
  }, []);

  const meta = [card.directors[0], card.kind === "movie" && card.runtime ? `${card.runtime} min` : null]
    .filter(Boolean).join(" · ");
  const note = !rating ? "No rating: it's logged as watched and hidden from picks, but won't train the model."
    : !dated ? "No date: it's treated as an old watch, so it trains the model at the lower backfill weight."
    : rewatch ? "A rewatch: it trains the model but doesn't count as a live-test hit."
    : `Predicted ${card.predicted.toFixed(1)}. Your rating trains the model and counts in the live test.`;

  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="logbox" ref={box} tabIndex={-1} role="dialog" aria-modal="true" aria-label={`Log ${card.title}`}>
        <button className="x" onClick={close} aria-label="Close">×</button>
        <div className="logposter">
          {card.poster ? <img src={card.poster} alt="" /> : <div className="noimg">{card.title}</div>}
        </div>
        <div className="logform">
          <header>
            <div className="logkicker">I watched…</div>
            <h2>{card.title} <span className="year">{card.year}</span></h2>
            {meta && <div className="sub">{meta}</div>}
          </header>

          <div className="logrow">
            <label className="check"><input type="checkbox" checked={dated} onChange={(e) => setDated(e.target.checked)} />
              {card.kind === "tv" ? "Finished on" : "Watched on"}</label>
            <input type="date" value={date} max={today()} disabled={!dated} aria-label="Date"
              onChange={(e) => setDate(e.target.value || today())} />
            <label className="check rewatch"><input type="checkbox" checked={rewatch} onChange={(e) => setRewatch(e.target.checked)} />
              I've {card.kind === "tv" ? "seen" : "watched"} this before</label>
          </div>

          <textarea placeholder="Add a review…" value={review} onChange={(e) => setReview(e.target.value)} rows={4} aria-label="Review" />

          <div className="lograte">
            <div className="field">
              <span className="lbl">Rating</span>
              <StarPicker value={rating} onChange={setRating} size={30} />
            </div>
            <div className="field">
              <span className="lbl">Like</span>
              <button className={`heart ${liked ? "on" : ""}`} onClick={() => setLiked(!liked)} aria-pressed={liked}
                aria-label="Like">{liked ? "♥" : "♡"}</button>
            </div>
            <button className="primary save" onClick={save} disabled={saving}>{saving ? "Saving…" : "Save"}</button>
          </div>
          <p className="lognote">
            {note} Saved here only; Letterboxd has no way for apps to post.{" "}
            {card.letterboxd_url && <a href={card.letterboxd_url} target="_blank" rel="noreferrer">Open on Letterboxd ↗</a>}
          </p>
          {err && <div className="banner bad">{err}</div>}
        </div>
      </div>
    </div>
  );
}
