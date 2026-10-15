"""
nids_features.py
================
Turns Zeek conn.log records into EXACTLY the feature vector the models were trained on.

It mirrors the preprocessing notebook:
  * same grouping of proto / service / state (state uses the Zeek map)
  * same recomputed columns: sload, dload, smean, dmean (+ the 6 extra features if they were used)
  * same log1p columns, same one-hot columns, same column order, same fitted scaler
  * ct_* features computed over a sliding window of the last 100 connections

The artifacts (scaler.joblib, meta.json) come from the `processed/` folder written by the
preprocessing notebook. If you change anything in that notebook (features, groupings),
change it here too and retrain.
"""

import json
from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Same constants as the preprocessing notebook
# ----------------------------------------------------------------------------
PROTO_CATS = ["tcp", "udp", "icmp", "other"]
SERVICE_CATS = ["-", "dns", "http", "smtp", "ftp", "ftp-data", "ssh",
                "pop3", "ssl", "irc", "dhcp", "snmp", "radius", "other"]
STATE_GROUPS = ["ok", "no_reply", "reset", "other"]
ZEEK_STATE_MAP = {"SF": "ok", "S1": "ok", "S2": "ok", "S3": "ok", "S0": "no_reply",
                  "REJ": "reset", "RSTO": "reset", "RSTR": "reset",
                  "RSTOS0": "reset", "RSTRH": "reset"}
CAT_SPEC = {"proto": PROTO_CATS, "service": SERVICE_CATS, "state": STATE_GROUPS}

CT_KEYS = {
    "ct_srv_src":       ("service", "srcip"),
    "ct_srv_dst":       ("service", "dstip"),
    "ct_dst_ltm":       ("dstip",),
    "ct_src_ltm":       ("srcip",),
    "ct_src_dport_ltm": ("srcip", "dsport"),
    "ct_dst_sport_ltm": ("dstip", "sport"),
    "ct_dst_src_ltm":   ("srcip", "dstip"),
}
CT_COLS = list(CT_KEYS)
BASE_NUMERIC_COLS = ["dur", "sbytes", "dbytes", "spkts", "dpkts",
                     "sload", "dload", "smean", "dmean", "is_sm_ips_ports", *CT_COLS]
EXTRA_COLS = ["rate", "sinpkt_approx", "dinpkt_approx", "byte_ratio", "pkt_ratio", "dur_zero"]
LOG_COLS = ["dur", "sbytes", "dbytes", "spkts", "dpkts", "sload", "dload", "smean", "dmean",
            "rate", "sinpkt_approx", "dinpkt_approx"]


# ----------------------------------------------------------------------------
# Zeek conn.log record -> raw flow
# ----------------------------------------------------------------------------
def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _get(rec: dict, dotted: str):
    """Zeek JSON flattens records ('id.orig_h'); also accept nested dicts."""
    if dotted in rec:
        return rec[dotted]
    cur = rec
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def group_proto(p):
    p = (p or "").strip().lower()
    return p if p in PROTO_CATS[:-1] else "other"


def group_service(s):
    if isinstance(s, (list, tuple)):
        s = s[0] if s else "-"
    s = (s or "-").strip().lower()
    s = s.split(",")[0] or "-"                      # keep the first service if several
    # Zeek recognises more services than Argus did. The training data never contains "other",
    # so a service the models don't know is treated as "-" (no recognised service) instead.
    return s if s in SERVICE_CATS[:-1] else "-"


def group_state(s):
    return ZEEK_STATE_MAP.get((s or "").strip().upper(), "other")


def parse_conn_record(rec: dict):
    """Return a raw flow dict from one parsed conn.log JSON line, or None if unusable."""
    try:
        ts = float(rec["ts"])
        srcip, dstip = _get(rec, "id.orig_h"), _get(rec, "id.resp_h")
        sport, dsport = int(_get(rec, "id.orig_p")), int(_get(rec, "id.resp_p"))
    except (KeyError, TypeError, ValueError):
        return None
    if not srcip or not dstip:
        return None

    dur = _num(rec.get("duration"))
    sbytes = rec.get("orig_ip_bytes", rec.get("orig_bytes"))   # IP-level bytes, closest to Argus
    dbytes = rec.get("resp_ip_bytes", rec.get("resp_bytes"))
    return {
        "ts": ts,
        "ltime": ts + dur,                          # conn.log is written in order of connection end
        "srcip": str(srcip), "sport": sport,
        "dstip": str(dstip), "dsport": dsport,
        "proto": group_proto(rec.get("proto")),
        "service": group_service(rec.get("service")),
        "state": group_state(rec.get("conn_state")),
        "dur": max(dur, 0.0),
        "sbytes": max(_num(sbytes), 0.0), "dbytes": max(_num(dbytes), 0.0),
        "spkts": max(_num(rec.get("orig_pkts")), 0.0), "dpkts": max(_num(rec.get("resp_pkts")), 0.0),
    }


# ----------------------------------------------------------------------------
# Streaming ct_* features
# ----------------------------------------------------------------------------
class StreamingCT:
    """
    For each flow, counts how many of the last `window` flows (including itself) share each key.
    Feed flows in the order Zeek wrote them. One O(1) update per flow.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self.buf = deque()
        self.counts = {name: Counter() for name in CT_KEYS}

    def update(self, flow: dict) -> dict:
        keys = {name: tuple(flow[c] for c in cols) for name, cols in CT_KEYS.items()}
        self.buf.append(keys)
        for name, k in keys.items():
            self.counts[name][k] += 1
        if len(self.buf) > self.window:
            old = self.buf.popleft()
            for name, k in old.items():
                c = self.counts[name]
                c[k] -= 1
                if c[k] <= 0:
                    del c[k]
        return {name: self.counts[name][k] for name, k in keys.items()}


# ----------------------------------------------------------------------------
# Flows -> scaled model input
# ----------------------------------------------------------------------------
def safe_div(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    out = np.zeros_like(a)
    np.divide(a, b, out=out, where=b > 0)
    return out


class FeatureBuilder:
    """Stateful (keeps the ct_* window). Create one per detector process."""

    def __init__(self, artifacts_dir: str = "processed", window: int = 100):
        art = Path(artifacts_dir)
        self.scaler = joblib.load(art / "scaler.joblib")
        self.meta = json.loads((art / "meta.json").read_text())
        self.feature_names = self.meta["feature_names"]
        self.scale_cols = self.meta["scale_cols"]
        # infer what the training notebook used, from the saved column list
        self.use_extra = "rate" in self.feature_names
        self.use_state = any(c.startswith("state_") for c in self.feature_names)
        self.ct = StreamingCT(window)

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sload"] = safe_div(df["sbytes"] * 8, df["dur"])
        df["dload"] = safe_div(df["dbytes"] * 8, df["dur"])
        df["smean"] = safe_div(df["sbytes"], df["spkts"])
        df["dmean"] = safe_div(df["dbytes"], df["dpkts"])
        if self.use_extra:
            df["rate"] = safe_div(df["spkts"] + df["dpkts"] - 1, df["dur"])
            df["sinpkt_approx"] = safe_div(df["dur"] * 1000, df["spkts"] - 1)
            df["dinpkt_approx"] = safe_div(df["dur"] * 1000, df["dpkts"] - 1)
            df["byte_ratio"] = np.log1p(df["sbytes"]) - np.log1p(df["dbytes"])
            df["pkt_ratio"] = np.log1p(df["spkts"]) - np.log1p(df["dpkts"])
            df["dur_zero"] = (df["dur"] <= 0).astype(int)
        return df

    def _model_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = BASE_NUMERIC_COLS + (EXTRA_COLS if self.use_extra else [])
        num = df[numeric].astype("float64").copy()
        for c in LOG_COLS:
            if c in num.columns:
                num[c] = np.log1p(num[c].clip(lower=0))
        parts = [num]
        for col, cats in CAT_SPEC.items():
            if col == "state" and not self.use_state:
                continue
            cat = pd.Series(pd.Categorical(df[col], categories=cats), index=df.index)
            parts.append(pd.get_dummies(cat, prefix=col).astype("float64"))
        return pd.concat(parts, axis=1)

    def transform(self, flows: list) -> np.ndarray:
        """flows: list of dicts from parse_conn_record, in Zeek's order. Returns scaled float32 array."""
        rows = []
        for f in flows:
            row = {k: f[k] for k in ("dur", "sbytes", "dbytes", "spkts", "dpkts",
                                     "proto", "service", "state")}
            row["is_sm_ips_ports"] = int(f["srcip"] == f["dstip"] and f["sport"] == f["dsport"])
            row.update(self.ct.update(f))
            rows.append(row)
        df = self._engineer(pd.DataFrame(rows))
        frame = self._model_frame(df)
        frame = frame.reindex(columns=self.feature_names, fill_value=0.0).copy()
        frame[self.scale_cols] = self.scaler.transform(frame[self.scale_cols])
        return frame.to_numpy(dtype="float32")