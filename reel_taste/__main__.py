"""Command line: python -m reel_taste {build|retrain|evaluate|rss|serve|cf}"""
from __future__ import annotations

import json
import sys


def main(argv: list[str]):
    cmd = argv[0] if argv else "help"
    if cmd == "cf":
        from .cf import CF
        CF.train()
    elif cmd == "build":
        from . import pipeline
        pipeline.build(full=True)
    elif cmd == "retrain":
        from . import pipeline
        pipeline.build(full=False)
    elif cmd == "evaluate":
        from . import pipeline
        from .evaluate import evaluate
        uf = pipeline.merged_user_films()
        res = evaluate(pipeline.build_context(uf), uf)
        for row, t in res["table"].items():
            print(f"== {row}")
            for k, v in t.items():
                print(f"  {k:26s} n={v['n']:3d} rmse={v['rmse']:.3f} mae={v['mae']:.3f} "
                      f"rho={v['spearman']:.3f} auc={v['auc']:.3f} top25%={v['hit_top25']:.2f}")
        print(json.dumps(res["bootstrap"], indent=1))
        print(json.dumps(res.get("watchlist"), indent=1))
    elif cmd == "rss":
        from . import pipeline
        print("new entries:", pipeline.refresh_rss())
    elif cmd == "serve":
        import uvicorn
        port = int(argv[1]) if len(argv) > 1 else 8765
        uvicorn.run("reel_taste.api:app", host="127.0.0.1", port=port)
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
