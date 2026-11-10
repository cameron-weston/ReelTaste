export type Card = {
  key: string; kind: "movie" | "tv"; tmdb_id: number; title: string; year: number | null;
  poster: string | null; overview: string; genres: string[]; directors: string[];
  runtime: number | null; language: string; vote_average: number; vote_count: number;
  predicted: number; confidence: "high" | "medium" | "low";
  because: { name: string; year: number | null; rating: number; sim: number }[];
  shared: string[]; why: string; why_source?: string; tmdb_url: string; letterboxd_url: string | null;
  seasons?: number; episodes?: number; status?: string; networks?: string[];
};

export type RecResponse = {
  items: Card[]; query_id: string | null; llm_used?: boolean; llm_error?: string | null;
  n_candidates?: number; error?: string; note?: string;
};

export type Status = {
  tmdb_key: boolean; built: boolean; built_at: string | null; has_films: boolean; has_shows: boolean;
  n_show_ratings: number; rss_last_fetch: string | null;
  job: { running: boolean; error: string | null; started: string | null }; job_log: string[];
};

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  status: () => req<Status>("/api/status"),
  llm: (force = false) => req<{ backend: string; ready: boolean; detail: string }>(`/api/llm?force=${force}`),
  recommend: (body: object) => req<RecResponse>("/api/recommend", { method: "POST", body: JSON.stringify(body) }),
  feedback: (body: object) => req("/api/feedback", { method: "POST", body: JSON.stringify(body) }),
  log: (body: object) => req("/api/log", { method: "POST", body: JSON.stringify(body) }),
  unlog: (id: number) => req(`/api/log?id=${id}`, { method: "DELETE" }),
  blocks: () => req<{ value: string; label: string }[]>("/api/blocklist"),
  block: (value: string, label: string) => req("/api/blocklist", { method: "POST", body: JSON.stringify({ value, label }) }),
  unblock: (value: string) => req(`/api/blocklist?value=${encodeURIComponent(value)}`, { method: "DELETE" }),
  taste: () => req<any>("/api/taste"),
  scoreboard: () => req<any>("/api/scoreboard"),
  build: (full: boolean) => req(`/api/build?full=${full}`, { method: "POST" }),
  refresh: () => req<{ new_rss_entries: number; retraining: boolean }>("/api/refresh", { method: "POST" }),
  showQueue: (q = "") => req<any>(`/api/shows/queue?q=${encodeURIComponent(q)}`),
  rateShow: (tmdb_id: number, rating: number | null, status: string) =>
    req("/api/shows/rate", { method: "POST", body: JSON.stringify({ tmdb_id, rating, status }) }),
};

export const stars = (r: number) => "★".repeat(Math.floor(r)) + (r % 1 >= 0.5 ? "½" : "");
