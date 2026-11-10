import { useEffect, useRef, useState } from "react";
import { api, Card, RecResponse, Status } from "./api";
import { Stars } from "./StarPicker";
import ShowRater from "./ShowRater";
import LogModal, { Logged } from "./LogModal";

type Props = { kind: "movie" | "tv"; llmReady: boolean; status: Status | null };

const PAGE_SIZES = [12, 24, 48, 96];
const PAGE_KEY = "perPage";

const PLACEHOLDER = {
  movie: "e.g. slow and strange, under two hours, not bleak",
  tv: "e.g. something I can finish in a weekend, funny but not a sitcom",
};

export default function Recommend({ kind, llmReady, status }: Props) {
  const [mood, setMood] = useState("");
  const [mode, setMode] = useState<"discover" | "watchlist">("discover");
  const [dial, setDial] = useState(0.5);
  const [useLlm, setUseLlm] = useState(true);
  const [showFilters, setShowFilters] = useState(false);
  const [f, setF] = useState<any>({ include_docs: true });
  const [res, setRes] = useState<RecResponse | null>(null);
  // Pages fetched for the query on screen, by page index (0-based). Revisiting one doesn't refetch.
  const [pages, setPages] = useState<Record<number, Card[]>>({});
  const [page, setPage] = useState(0);
  const [nPages, setNPages] = useState(0);
  const [pageInput, setPageInput] = useState("1");
  const [perPage, setPerPage] = useState<number>(() => {
    try { const v = Number(localStorage.getItem(PAGE_KEY)); return PAGE_SIZES.includes(v) ? v : 24; } catch { return 24; }
  });
  const items = pages[page] ?? [];
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [done, setDone] = useState<Record<number, string>>({});
  const [rater, setRater] = useState(false);
  const [logging, setLogging] = useState<Card | null>(null);
  const [logged, setLogged] = useState<Record<number, Logged>>({});

  const ready = kind === "movie" ? status?.has_films : status?.has_shows;
  const reqId = useRef(0); // drops a slow response that a newer request has already replaced
  const lastAsk = useRef<any>(null); // "More" continues the query on screen, not edits made since

  // A new query fetches page 0 (Claude may reorder its top). Other pages continue the model's order,
  // skipping what page 0 showed, so any page can be fetched directly.
  const ask = async (target = 0, size = perPage) => {
    const id = ++reqId.current;
    const fresh = target === 0;
    setLoading(true); setErr("");
    try {
      const filters: any = {};
      for (const [k, v] of Object.entries(f)) if (v !== "" && v !== null && v !== undefined) filters[k] = v;
      const q = !fresh && lastAsk.current ? lastAsk.current : { kind, mode, mood, filters, dial, n: size };
      const r = await api.recommend({
        ...q, page: target, use_llm: useLlm && llmReady && fresh,
        query_id: fresh ? null : res?.query_id, first_page: fresh ? [] : (pages[0] ?? []).map((c) => c.key),
      });
      if (id !== reqId.current) return;
      if (r.error) setErr(r.error);
      if (fresh) {
        lastAsk.current = q; // only once these results are the ones on screen
        setRes(r); setPages(r.items.length ? { 0: r.items } : {}); setPage(0); setPageInput("1");
        setNPages(r.n_pages ?? 0); setDone({}); setLogged({});
      } else if (r.items.length) {
        setPages((ps) => ({ ...ps, [target]: r.items })); setPage(target); setPageInput(String(target + 1));
      } else {
        setNPages(target); setPageInput(String(page + 1)); // ran past the end: the last page is the one before
      }
    } catch (e: any) { if (id === reqId.current) setErr(String(e.message || e)); }
    if (id === reqId.current) setLoading(false);
  };
  const goTo = (p: number) => {
    if (!Number.isFinite(p)) { setPageInput(String(page + 1)); return; }
    const t = Math.max(0, Math.min(p, Math.max(nPages - 1, 0)));
    window.scrollTo({ top: 0, behavior: "smooth" });
    if (pages[t]) { setPage(t); setPageInput(String(t + 1)); }
    else ask(t);
  };
  const changeSize = (n: number) => {
    setPerPage(n);
    try { localStorage.setItem(PAGE_KEY, String(n)); } catch {}
    if (ready) ask(0, n);
  };
  const shownSize = lastAsk.current?.n ?? perPage; // size of the query on screen, not a pending change
  const pageStart = page * shownSize;
  const pager = items.length > 0 && (
    <nav className="pager" aria-label="Pages">
      <button onClick={() => goTo(page - 1)} disabled={page === 0 || loading}>← Prev</button>
      <span className="pager-page">
        <input type="number" min={1} max={nPages || 1} value={pageInput} aria-label="Page number" disabled={loading}
          onChange={(e) => setPageInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") goTo(parseInt(pageInput, 10) - 1); }}
          onBlur={() => setPageInput(String(page + 1))} /* Enter navigates; leaving the box just resets it */
          onFocus={(e) => e.target.select()} />
        <span className="muted">of {nPages.toLocaleString()}</span>
      </span>
      <button onClick={() => goTo(page + 1)} disabled={loading || page >= nPages - 1}>
        {loading ? "Loading…" : "Next →"}</button>
      <span className="muted pager-info">
        {(pageStart + 1).toLocaleString()}–{(pageStart + items.length).toLocaleString()} of {res?.n_candidates?.toLocaleString()} eligible {kind === "movie" ? "films" : "shows"}
      </span>
      <label className="pager-size">Per page
        <select value={perPage} onChange={(e) => changeSize(+e.target.value)} disabled={loading}>
          {PAGE_SIZES.map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
      </label>
    </nav>
  );
  const act = async (fn: () => Promise<void>) => {
    try { await fn(); } catch (e: any) { setErr(String(e.message || e)); }
  };

  useEffect(() => { if (ready) ask(0); }, [kind, ready]);

  const fb = (c: Card, action: string) => {
    // Open before any await: Safari blocks pop-ups that don't come straight from the click.
    if (action === "watchlist" && c.letterboxd_url) window.open(c.letterboxd_url, "_blank");
    return act(async () => {
      await api.feedback({ kind: c.kind, tmdb_id: c.tmdb_id, action, query_id: res?.query_id });
      setDone((d) => ({ ...d, [c.tmdb_id]: action }));
    });
  };
  const onLogged = (c: Card, l: Logged) => {
    setLogged((m) => ({ ...m, [c.tmdb_id]: l }));
    setLogging(null);
  };
  const unlog = (c: Card) => act(async () => {
    await api.unlog(logged[c.tmdb_id].id);
    setLogged(({ [c.tmdb_id]: _, ...rest }) => rest);
  });
  const block = (c: Card, value: string, label: string) => act(async () => {
    await api.block(value, label);
    setDone((d) => ({ ...d, [c.tmdb_id]: `blocked ${label}` }));
  });

  return (
    <div>
      {kind === "tv" && (
        <div className="note">
          Shows is the weaker tab: Letterboxd barely tracks TV, so this runs on your film taste plus
          {" "}{status?.n_show_ratings ?? 0} show ratings. Rate more to sharpen it.{" "}
          <button className="link" onClick={() => setRater(!rater)}>{rater ? "Hide rating session" : "Rate shows"}</button>
        </div>
      )}
      {kind === "tv" && rater && <ShowRater />}

      <section className="ask">
        <textarea value={mood} onChange={(e) => setMood(e.target.value)} placeholder={PLACEHOLDER[kind]}
          onKeyDown={(e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !loading && ready) ask(0); }} rows={2} />
        <div className="controls">
          {kind === "movie" && (
            <div className="seg">
              <button className={mode === "discover" ? "on" : ""} onClick={() => setMode("discover")}>Discover</button>
              <button className={mode === "watchlist" ? "on" : ""} onClick={() => setMode("watchlist")}>My watchlist</button>
            </div>
          )}
          <label className="dial" title="How hard to push away from popular titles">
            <span>Crowd-pleasers</span>
            <input type="range" min={0} max={1} step={0.05} value={dial} onChange={(e) => setDial(+e.target.value)} disabled={mode === "watchlist"} />
            <span>Deep cuts</span>
          </label>
          <label className="check" title={llmReady ? "Claude reorders the shortlist for your mood and writes the why" : "Claude CLI not signed in — see Scoreboard"}>
            <input type="checkbox" checked={useLlm && llmReady} disabled={!llmReady} onChange={(e) => setUseLlm(e.target.checked)} /> Claude
          </label>
          <button className="link" onClick={() => setShowFilters(!showFilters)}>Filters</button>
          <button className="primary" onClick={() => ask(0)} disabled={loading || !ready}>{loading ? (useLlm && llmReady ? "Asking Claude…" : "Ranking…") : "Recommend"}</button>
        </div>
        {showFilters && (
          <div className="filters">
            {kind === "movie" && <label>Max runtime <input type="number" placeholder="min" value={f.max_runtime ?? ""} onChange={(e) => setF({ ...f, max_runtime: e.target.value ? +e.target.value : "" })} /></label>}
            <label>Years <input type="number" placeholder="from" value={f.year_min ?? ""} onChange={(e) => setF({ ...f, year_min: e.target.value ? +e.target.value : "" })} />
              – <input type="number" placeholder="to" value={f.year_max ?? ""} onChange={(e) => setF({ ...f, year_max: e.target.value ? +e.target.value : "" })} /></label>
            <label>Language <select value={f.languages?.[0] ?? ""} onChange={(e) => setF({ ...f, languages: e.target.value ? [e.target.value] : "" })}>
              <option value="">Any</option><option value="en">English</option><option value="fr">French</option><option value="ja">Japanese</option>
              <option value="ko">Korean</option><option value="es">Spanish</option><option value="it">Italian</option><option value="de">German</option></select></label>
            <label>Min votes <input type="number" placeholder={kind === "movie" ? "200" : "300"} value={f.min_votes ?? ""} onChange={(e) => setF({ ...f, min_votes: e.target.value ? +e.target.value : "" })} /></label>
            {kind === "movie" && <label className="check"><input type="checkbox" checked={f.include_docs} onChange={(e) => setF({ ...f, include_docs: e.target.checked })} /> Documentaries</label>}
            {kind === "tv" && <label className="check"><input type="checkbox" checked={!!f.finished_only} onChange={(e) => setF({ ...f, finished_only: e.target.checked })} /> Finished shows only</label>}
            {kind === "tv" && <label>Max seasons <input type="number" value={f.max_seasons ?? ""} onChange={(e) => setF({ ...f, max_seasons: e.target.value ? +e.target.value : "" })} /></label>}
          </div>
        )}
      </section>

      {!ready && <div className="empty">{status?.job.running ? "Building the model… this tab appears when it's done." : "Not built yet — see the Scoreboard tab."}</div>}
      {err && <div className="banner bad">{err}</div>}
      {res?.llm_error && <div className="banner">Claude step skipped ({res.llm_error}). Showing the model's own order.</div>}
      {res?.note && <div className="empty">{res.note}</div>}
      {res?.note_soft && <div className="note">{res.note_soft}</div>}

      {pager}
      <div className="grid">
        {items.map((c, i) => {
          const lg = logged[c.tmdb_id];
          return (
          <article key={c.key} data-key={c.key} tabIndex={-1} className={`card ${done[c.tmdb_id] ? "dim" : ""} ${lg ? "isLogged" : ""}`}>
            <div className="poster">{c.poster ? <img src={c.poster} alt="" loading="lazy" /> : <div className="noimg">{c.title}</div>}
              <span className="rank">{pageStart + i + 1}</span></div>
            <div className="body">
              <header className="card-head">
                <h3><a href={c.letterboxd_url || c.tmdb_url} target="_blank" rel="noreferrer">{c.title}</a> <span className="year">{c.year}</span></h3>
                <div className="sub">
                  {[
                    c.directors[0],
                    c.kind === "movie" ? (c.runtime ? `${c.runtime} min` : null)
                      : `${c.seasons ?? "?"} season${c.seasons === 1 ? "" : "s"}${c.episodes ? ` (${c.episodes} eps)` : ""}`,
                    c.kind === "tv" ? (c.status === "Ended" || c.status === "Canceled" ? "finished" : "ongoing") : null,
                    c.networks?.[0],
                    c.language !== "en" ? c.language?.toUpperCase() : null,
                  ].filter(Boolean).join(" · ")}
                </div>
              </header>
              <div className="pred">
                <Stars value={c.predicted} size={17} />
                <span className="num">{c.predicted.toFixed(1)}</span>
                <span className={`conf ${c.confidence}`}>{c.confidence} confidence</span>
              </div>
              <p className="why">{c.why}{c.why_source === "llm" && <span className="tag">Claude</span>}</p>
              {c.because.length > 0 && (
                <div className="because">{c.because.map((b) => (
                  <span key={b.name} title={b.name}><span className="bn">{b.name}</span><Stars value={b.rating} size={11} /></span>))}
                </div>
              )}
              <div className="actions">
                {lg ? <span className="logged">
                  <span>✓ Logged</span>
                  {lg.rating ? <Stars value={lg.rating} size={14} /> : <span>as watched</span>}
                  {lg.liked && <span className="heart on small" aria-label="liked">♥</span>}
                  <span className="muted">{lg.retraining ? "· retraining now" : lg.rating ? `· retrains in ${lg.left} more` : ""}</span>
                  <button className="link" onClick={() => unlog(c)}>Undo</button>
                </span> : done[c.tmdb_id] ? <span className="muted">✓ {done[c.tmdb_id].replace("_", " ")}</span> : <>
                  <button className="watched" onClick={() => setLogging(c)}>Watched it!</button>
                  <button onClick={() => fb(c, "not_interested")}>Not interested</button>
                  {c.kind === "movie" && <button onClick={() => fb(c, "watchlist")}>+ Watchlist</button>}
                  <details className="blockmenu"><summary>Block</summary>
                    <div className="blockpop">
                      <button onClick={() => block(c, c.key, c.title)}>This {c.kind === "movie" ? "film" : "show"}</button>
                      {c.directors[0] && <button onClick={() => block(c, `person:${c.directors[0]}`, c.directors[0])}>{c.directors[0]}</button>}
                      {c.genres.map((g) => <button key={g} onClick={() => block(c, `genre:${g}`, g)}>All {g}</button>)}
                    </div>
                  </details>
                </>}
              </div>
            </div>
          </article>
          );
        })}
      </div>
      {logging && <LogModal card={logging} queryId={res?.query_id} onClose={() => setLogging(null)} onSaved={(l) => onLogged(logging, l)} />}
      {items.length > 0 && pager}
    </div>
  );
}
