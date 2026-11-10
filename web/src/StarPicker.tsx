import { useId, useState } from "react";

// Five drawn stars, each split into a left and right half (half-star ratings like Letterboxd).
// Hover previews, click sets, clicking the current value clears it; arrow keys step by half a star.

const PATH = "M12 2.2l2.95 6.3 6.85.8-5.08 4.7 1.37 6.8L12 17.4l-6.09 3.4 1.37-6.8L2.2 9.3l6.85-.8z";

type Props = {
  value: number | null;
  onChange: (v: number | null) => void;
  size?: number;
  label?: string;
  showValue?: boolean;
  /** Arrow keys preview and Enter commits (for pickers whose every change saves at once). */
  commitOnEnter?: boolean;
};

export default function StarPicker({ value, onChange, size = 28, label = "Rating", showValue = true, commitOnEnter = false }: Props) {
  const [hover, setHover] = useState(0);
  const [draft, setDraft] = useState<number | null>(null); // keyboard preview when commitOnEnter
  const uid = useId().replace(/:/g, "");
  const cur = commitOnEnter ? draft ?? value : value;
  const shown = hover || cur || 0;

  const pick = (v: number) => { onChange(value === v ? null : v); setHover(0); setDraft(null); };
  const set = (v: number | null) => (commitOnEnter ? setDraft(v) : onChange(v));
  const key = (e: React.KeyboardEvent) => {
    const c = cur ?? 0;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") { e.preventDefault(); set(Math.min(5, c + 0.5)); }
    else if (e.key === "ArrowLeft" || e.key === "ArrowDown") { e.preventDefault(); set(c <= 0.5 ? null : c - 0.5); }
    else if (e.key === "Backspace" || e.key === "Delete") { e.preventDefault(); set(null); }
    else if (e.key === "Enter" && commitOnEnter && draft) { e.preventDefault(); pick(draft); }
  };

  return (
    <div className="stars-pick" role="slider" tabIndex={0} aria-label={label} aria-valuemin={0} aria-valuemax={5}
      aria-valuenow={cur ?? 0} aria-valuetext={cur ? `${cur} stars` : "no rating"}
      onKeyDown={key} onMouseLeave={() => setHover(0)} onBlur={() => setDraft(null)}>
      {[1, 2, 3, 4, 5].map((n) => {
        const fill = shown >= n ? 1 : shown >= n - 0.5 ? 0.5 : 0;
        return (
          <span key={n} className="star" style={{ width: size, height: size }}>
            <svg viewBox="0 0 24 24" width={size} height={size} aria-hidden="true">
              <defs><clipPath id={`${uid}-${n}`}><rect x="0" y="0" width="12" height="24" /></clipPath></defs>
              <path d={PATH} className="star-empty" />
              {fill > 0 && <path d={PATH} className="star-full" clipPath={fill === 0.5 ? `url(#${uid}-${n})` : undefined} />}
            </svg>
            <span className="hit l" onMouseEnter={() => setHover(n - 0.5)} onClick={() => pick(n - 0.5)} />
            <span className="hit r" onMouseEnter={() => setHover(n)} onClick={() => pick(n)} />
          </span>
        );
      })}
      {showValue && <span className="stars-val">{shown ? shown.toFixed(1) : "–"}</span>}
    </div>
  );
}

/** Read-only stars for display (predictions, logged ratings). */
export function Stars({ value, size = 16 }: { value: number; size?: number }) {
  const v = Math.round(value * 2) / 2;
  const uid = useId().replace(/:/g, "");
  return (
    <span className="stars-row" aria-label={`${v} stars`}>
      {[1, 2, 3, 4, 5].map((n) => {
        const fill = v >= n ? 1 : v >= n - 0.5 ? 0.5 : 0;
        return (
          <svg key={n} viewBox="0 0 24 24" width={size} height={size} aria-hidden="true">
            <defs><clipPath id={`${uid}-${n}`}><rect x="0" y="0" width="12" height="24" /></clipPath></defs>
            <path d={PATH} className="star-empty" />
            {fill > 0 && <path d={PATH} className="star-full" clipPath={fill === 0.5 ? `url(#${uid}-${n})` : undefined} />}
          </svg>
        );
      })}
    </span>
  );
}
