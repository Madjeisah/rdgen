"""Collect per-model runs into the population table used by the analysis,
adding agreement-on-the-line predictions (which need all models together)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..baselines.aotl import agreement_on_the_line


def aggregate(runs_dir: str | Path) -> pd.DataFrame:
    runs = Path(runs_dir)
    rows, preds = [], []
    for summ in sorted(runs.glob("*/*/summary.json")):
        rec = json.loads(summ.read_text())
        rec.pop("by_polarity", None)
        prov = rec.pop("provenance", {})
        rec["prereg_status"] = (prov.get("prereg") or {}).get("status", "")
        rec.update({f"x_{k}": v for k, v in (rec.pop("extra", {}) or {}).items()})
        rows.append(rec)
        p = summ.parent / "preds.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["domain"] = rec["domain"]
            preds.append(df)
    table = pd.DataFrame(rows)
    if preds and len(table):
        allp = pd.concat(preds, ignore_index=True)
        aotl = [agreement_on_the_line(g).assign(domain=d) for d, g in allp.groupby("domain")]
        aotl_df = pd.concat(aotl, ignore_index=True).rename(columns={"model": "name"})
        table = table.merge(aotl_df[["name", "domain", "aotl_pred", "aotl_r2"]],
                            on=["name", "domain"], how="left")
    return table
