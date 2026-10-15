"""
live_detector.py
================
Reads Zeek's JSON conn.log, builds the model features, scores every flow and writes alerts.

Two modes
---------
replay (default): process a finished conn.log from start to end, then exit.
    python live_detector.py --conn-log conn.log --model xgb

follow: keep watching a conn.log that Zeek is still writing (live capture).
    python live_detector.py --conn-log conn.log --model xgb --follow

Outputs
-------
alerts.jsonl          one JSON line per flow flagged as an attack (the dashboard/API will read this)
scored_flows.jsonl    optional (--scored-out): every flow with its score
features_live.npy     optional (--dump-features): scaled feature rows for check_distribution.py

Note: Zeek writes a connection to conn.log only when the connection ENDS (or times out),
and it buffers writes, so alerts arrive a few seconds to minutes after the traffic started.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from nids_features import FeatureBuilder, parse_conn_record


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def load_predictor(name: str, model_dir: str):
    """Return a function X -> attack probability (1-D array)."""
    mdir = Path(model_dir)
    if name == "rf":
        import joblib
        model = joblib.load(mdir / "rf_binary.joblib")
        return lambda X: model.predict_proba(X)[:, 1]
    if name == "xgb":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(mdir / "xgb_binary.json")
        return lambda X: model.predict_proba(X)[:, 1]
    if name == "mlp":
        import keras
        model = keras.models.load_model(mdir / "mlp_binary.keras")
        return lambda X: model.predict(X, batch_size=1024, verbose=0).ravel()
    raise ValueError("--model must be rf, xgb or mlp")


# ----------------------------------------------------------------------------
# Reading conn.log
# ----------------------------------------------------------------------------
def read_lines(path: Path, follow: bool, from_start: bool):
    """
    Yield complete lines. Yields None whenever there is nothing new, so the caller can flush a batch.
    In replay mode it yields None once at the end of the file and stops.
    """
    while not path.exists():
        if not follow:
            raise FileNotFoundError(path)
        print(f"waiting for {path} ...", file=sys.stderr)
        yield None
        time.sleep(1.0)

    f = open(path, "r", encoding="utf-8", errors="replace")
    if follow and not from_start:
        f.seek(0, os.SEEK_END)
    buf = ""
    try:
        while True:
            chunk = f.readline()
            if chunk:
                buf += chunk
                if buf.endswith("\n"):
                    yield buf
                    buf = ""
                continue
            yield None
            if not follow:
                return
            time.sleep(0.3)
            try:                                      # log rotated or truncated: start over
                if os.stat(path).st_size < f.tell():
                    f.close()
                    f = open(path, "r", encoding="utf-8", errors="replace")
                    buf = ""
            except FileNotFoundError:
                pass
    finally:
        f.close()


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conn-log", required=True, help="path to Zeek conn.log (JSON)")
    ap.add_argument("--artifacts", default="processed", help="folder with scaler.joblib and meta.json")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--model", default="xgb", choices=["rf", "xgb", "mlp"])
    ap.add_argument("--threshold", type=float, default=0.5, help="attack probability needed to raise an alert")
    ap.add_argument("--follow", action="store_true", help="keep watching the log (live mode)")
    ap.add_argument("--from-start", action="store_true", help="in follow mode, read the file from the beginning")
    ap.add_argument("--alerts-out", default="alerts.jsonl")
    ap.add_argument("--scored-out", default=None, help="optional: write every scored flow here")
    ap.add_argument("--dump-features", default=None, help="optional: save scaled features as .npy for the drift check")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    builder = FeatureBuilder(args.artifacts)
    predict = load_predictor(args.model, args.model_dir)
    print(f"model: {args.model}   threshold: {args.threshold}   features: {len(builder.feature_names)}")

    alerts_f = open(args.alerts_out, "a", encoding="utf-8")
    scored_f = open(args.scored_out, "a", encoding="utf-8") if args.scored_out else None
    feature_chunks = [] if args.dump_features else None
    stats = Counter()
    pending = []

    def flush():
        if not pending:
            return
        X = builder.transform(pending)
        scores = predict(X)
        if feature_chunks is not None:
            feature_chunks.append(X)
        for flow, p in zip(pending, scores):
            p = float(p)
            is_attack = p >= args.threshold
            stats["flows"] += 1
            record = {
                "time": iso(flow["ltime"]), "ts": flow["ltime"],
                "src": flow["srcip"], "sport": flow["sport"],
                "dst": flow["dstip"], "dport": flow["dsport"],
                "proto": flow["proto"], "service": flow["service"], "state": flow["state"],
                "duration": round(flow["dur"], 6),
                "sbytes": int(flow["sbytes"]), "dbytes": int(flow["dbytes"]),
                "spkts": int(flow["spkts"]), "dpkts": int(flow["dpkts"]),
                "score": round(p, 4), "attack": bool(is_attack), "model": args.model,
            }
            if scored_f:
                scored_f.write(json.dumps(record) + "\n")
            if is_attack:
                stats["alerts"] += 1
                alerts_f.write(json.dumps(record) + "\n")
                print(f"[ALERT] {record['time']}  {record['src']}:{record['sport']} -> "
                      f"{record['dst']}:{record['dport']}  {record['proto']}  score={p:.3f}")
        alerts_f.flush()
        if scored_f:
            scored_f.flush()
        pending.clear()

    try:
        for line in read_lines(Path(args.conn_log), args.follow, args.from_start):
            if line is None:                      # nothing new: score what we have
                flush()
                continue
            line = line.strip()
            if not line or line.startswith("#"):  # skip blank lines and TSV headers
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_lines"] += 1
                continue
            flow = parse_conn_record(rec)
            if flow is None:
                stats["skipped"] += 1
                continue
            pending.append(flow)
            if len(pending) >= args.batch_size:
                flush()
        flush()
    except KeyboardInterrupt:
        flush()
        print("\nstopped by user")
    finally:
        alerts_f.close()
        if scored_f:
            scored_f.close()
        if feature_chunks:
            np.save(args.dump_features, np.vstack(feature_chunks))
            print(f"saved scaled features to {args.dump_features}")
        pct = 100 * stats["alerts"] / max(stats["flows"], 1)
        print(f"\nflows scored: {stats['flows']}   alerts: {stats['alerts']} ({pct:.1f}%)   "
              f"skipped: {stats['skipped']}   bad lines: {stats['bad_lines']}")


if __name__ == "__main__":
    main()
