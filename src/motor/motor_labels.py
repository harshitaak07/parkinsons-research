import os
import pandas as pd
from typing import List, Tuple


def _read_csv_safe(path: str, usecols: List[str] | None = None) -> pd.DataFrame:
    """
    Read a CSV if it exists. If missing, return empty DataFrame with requested columns.
    """
    if not os.path.exists(path):
        cols = usecols if usecols is not None else []
        return pd.DataFrame(columns=cols)
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize join keys to `PATNO` and `EVENT_ID` if present; coerce types.
    """
    if df.empty:
        return df
    out = df.copy()
    # Standardize typical variants
    if "patno" in out.columns and "PATNO" not in out.columns:
        out = out.rename(columns={"patno": "PATNO"})
    if "event_id" in out.columns and "EVENT_ID" not in out.columns:
        out = out.rename(columns={"event_id": "EVENT_ID"})
    # Coerce to string for safe joins
    for key in ["PATNO", "EVENT_ID"]:
        if key in out.columns:
            out[key] = out[key].astype(str).str.strip()
    return out


def _select_existing(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()


def load_gait_substudy(path: str) -> pd.DataFrame:
    df = _normalize_keys(_read_csv_safe(path))
    cols = [
        "PATNO", "EVENT_ID", "SUB_EVENT_ID", "PAG_NAME", "INFODT",
        "AXIVITYUSED", "AXIVITYDT", "AXIVITYUP", "AXIVITYUPDT",
        "OPALUSED", "GAITTUG1", "GAITTUG2", "GAITASTCMPLT", "GAITASTDUAL"
    ]
    return _select_existing(df, cols)


def load_mds_updrs_part3(path: str) -> pd.DataFrame:
    df = _normalize_keys(_read_csv_safe(path))
    # Core fields and total score
    cols = [
        "PATNO", "EVENT_ID", "PAG_NAME", "INFODT", "PDSTATE", "PDMEDYN",
        "NP3TOT"
    ]
    out = _select_existing(df, cols)
    # If NP3TOT missing, attempt to sum NP3 item columns
    if not out.empty and "NP3TOT" not in out.columns:
        np3_cols = [c for c in df.columns if c.startswith("NP3") and c != "NP3TOT"]
        if np3_cols:
            tmp = df[["PATNO", "EVENT_ID"] + np3_cols].copy()
            tmp[np3_cols] = tmp[np3_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            summed = tmp.groupby(["PATNO", "EVENT_ID"], as_index=False)[np3_cols].sum()
            summed["NP3TOT"] = summed[np3_cols].sum(axis=1)
            out = _normalize_keys(out.merge(summed[["PATNO", "EVENT_ID", "NP3TOT"],], on=["PATNO", "EVENT_ID"], how="left"))
    return out


def load_mds_updrs_part2(path: str) -> pd.DataFrame:
    df = _normalize_keys(_read_csv_safe(path))
    cols = ["PATNO", "EVENT_ID", "PAG_NAME", "INFODT", "NP2PTOT"]
    out = _select_existing(df, cols)
    # If total missing, compute from items
    if not out.empty and "NP2PTOT" not in out.columns:
        np2_cols = [c for c in df.columns if c.startswith("NP2")] 
        if np2_cols:
            tmp = df[["PATNO", "EVENT_ID"] + np2_cols].copy()
            tmp[np2_cols] = tmp[np2_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            agg = tmp.groupby(["PATNO", "EVENT_ID"], as_index=False)[np2_cols].sum()
            agg["NP2PTOT"] = agg[np2_cols].sum(axis=1)
            out = _normalize_keys(out.merge(agg[["PATNO", "EVENT_ID", "NP2PTOT"],], on=["PATNO", "EVENT_ID"], how="left"))
    return out


def load_schwab_england(path: str) -> pd.DataFrame:
    df = _normalize_keys(_read_csv_safe(path))
    cols = ["PATNO", "EVENT_ID", "PAG_NAME", "INFODT", "MSEADLG"]
    return _select_existing(df, cols)


def _compute_row_mean(df: pd.DataFrame, item_prefixes: Tuple[str, ...]) -> pd.Series:
    cols = [c for c in df.columns if any(c.startswith(p) for p in item_prefixes)]
    if not cols:
        return pd.Series(index=df.index, dtype=float)
    vals = df[cols].apply(pd.to_numeric, errors="coerce")
    return vals.mean(axis=1)


def load_neuroqol_ue(path: str) -> pd.DataFrame:
    df = _normalize_keys(_read_csv_safe(path))
    if df.empty:
        return df
    out_cols = ["PATNO", "EVENT_ID", "PAG_NAME", "INFODT"]
    out = _select_existing(df, out_cols)
    out["NQ_UEFS_MEAN"] = _compute_row_mean(df, ("NQUEX",))
    return out


def load_neuroqol_le(path: str) -> pd.DataFrame:
    df = _normalize_keys(_read_csv_safe(path))
    if df.empty:
        return df
    out_cols = ["PATNO", "EVENT_ID", "PAG_NAME", "INFODT"]
    out = _select_existing(df, out_cols)
    out["NQ_LEFS_MEAN"] = _compute_row_mean(df, ("NQMOB",))
    return out


def load_pquest(path: str) -> pd.DataFrame:
    df = _normalize_keys(_read_csv_safe(path))
    cols = [
        "PATNO", "EVENT_ID", "PAG_NAME", "INFODT",
        "TRBUPCHR", "WRTSMLR", "VOICSFTR", "POORBAL", "FTSTUCK",
        "LSSXPRSS", "ARMLGSHK", "TRBBUTTN", "SHUFFLE", "MVSLOW", "TOLDPD"
    ]
    out = _select_existing(df, cols)
    if out.empty:
        return out
    # Ensure binary ints
    symptom_cols = [c for c in out.columns if c not in ("PATNO", "EVENT_ID", "PAG_NAME", "INFODT")]
    out[symptom_cols] = out[symptom_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    out["PQUEST_SUM"] = out[symptom_cols].sum(axis=1)
    return out


def build_motor_labels(
    base_dir: str = "data/raw/motor",
    output_path: str = "data/processed/motor_labels.csv",
) -> pd.DataFrame:
    """
    Build a unified labels/metadata table keyed by PATNO + EVENT_ID.
    Includes: UPDRS Part III (primary), Part II, Schwab & England, NeuroQoL UE/LE,
    PQUEST, and Gait Substudy flags (Axivity/Opal usage and TUG times).
    """
    paths = {
        "gait_substudy": os.path.join(base_dir, "Gait_Substudy_Gait_Mobility_Assessment_and_Measurement_01Oct2025.csv"),
        "updrs3": os.path.join(base_dir, "MDS-UPDRS_Part_III_01Oct2025.csv"),
        "updrs2": os.path.join(base_dir, "MDS_UPDRS_Part_II__Patient_Questionnaire_01Oct2025.csv"),
        "schwab": os.path.join(base_dir, "Modified_Schwab___England_Activities_of_Daily_Living_01Oct2025.csv"),
        "nq_ue": os.path.join(base_dir, "Neuro_QoL__Upper_Extremity_Function_-_Short_Form_01Oct2025.csv"),
        "nq_le": os.path.join(base_dir, "Neuro_QoL__Lower_Extremity_Function__Mobility__-_Short_Form_01Oct2025.csv"),
        "pquest": os.path.join(base_dir, "Participant_Motor_Function_Questionnaire_01Oct2025.csv"),
    }

    df_gait = load_gait_substudy(paths["gait_substudy"])  # flags + TUG
    df_u3 = load_mds_updrs_part3(paths["updrs3"])         # NP3TOT + PDSTATE
    df_u2 = load_mds_updrs_part2(paths["updrs2"])         # NP2PTOT
    df_se = load_schwab_england(paths["schwab"])          # MSEADLG
    df_ue = load_neuroqol_ue(paths["nq_ue"])              # NQ_UEFS_MEAN
    df_le = load_neuroqol_le(paths["nq_le"])              # NQ_LEFS_MEAN
    df_pq = load_pquest(paths["pquest"])                  # PQUEST items + sum

    # Start from UPDRS3 as primary label table; if empty, start from gait_substudy
    if not df_u3.empty:
        labels = df_u3.copy()
    else:
        labels = df_gait.copy()

    for part in (df_u2, df_se, df_ue, df_le, df_pq, df_gait):
        if not part.empty:
            labels = labels.merge(part, on=["PATNO", "EVENT_ID"], how="outer", suffixes=("", "_dup"))
            # Drop accidental duplicate suffix columns
            dup_cols = [c for c in labels.columns if c.endswith("_dup")]
            if dup_cols:
                labels = labels.drop(columns=dup_cols)

    # Sort and tidy
    sort_cols = [c for c in ["PATNO", "EVENT_ID", "INFODT"] if c in labels.columns]
    if sort_cols:
        labels = labels.sort_values(by=sort_cols)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labels.to_csv(output_path, index=False)
    return labels


def main():
    build_motor_labels()
    print("Saved motor labels to data/processed/motor_labels.csv")


if __name__ == "__main__":
    main()




