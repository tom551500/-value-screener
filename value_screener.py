# -*- coding: utf-8 -*-
"""
估值河流選股器 (value_screener.py)

概念：
  把「估值河流圖」的位置（股價淨值比在自己歷史的第幾層）當成第一道篩選，
  再用四道品質過濾，避開「低 PB 但基本面一路惡化」的價值陷阱：
    1. ROE：近四季 EPS 不可連續下滑、單季年化 ROE 要有底
    2. 營收：近 3 月年增率
    3. 毛利率：最新一季不可明顯低於前四季平均、營益率為正
    4. 股價止跌：站上均線、離近期低點有反彈、量夠
  另外附上「同業 PB 比較」（判斷是產業性還是個股性低估）。

資料來源：FinMind API（免費註冊等級即可，不需要分K或籌碼資料）
  TaiwanStockInfo / TaiwanStockPER / TaiwanStockPrice
  TaiwanStockFinancialStatements / TaiwanStockMonthRevenue

為了節省 API 次數，採兩階段掃描：
  第一階段（每檔 1 次請求）：PER/PBR → PB 分位、河流層、ROE(TTM)
  第二階段（只對通過第一階段者，每檔 3 次請求）：股價、財報、月營收
結果會快取在本機資料夾，同一天重按掃描不會重複扣 API 次數，可以分批續掃。

本程式僅整理資料與篩選，不構成買賣建議。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "https://api.finmindtrade.com/api/v4/data"
CACHE_DIR = Path(".screener_cache")
TZ = ZoneInfo("Asia/Taipei")
LAYER_CUTS = (10, 30, 50, 70, 90)  # 河流線 = 自己歷史的 P10/P30/P50/P70/P90
NAN = float("nan")


# ----------------------------------------------------------------------------
# 設定
# ----------------------------------------------------------------------------
@dataclass
class Cfg:
    window_years: int = 5
    # 1 估值
    use_val: bool = True
    max_pct: float = 15.0
    # 2 ROE
    use_roe: bool = True
    min_roe: float = 6.0
    min_runrate: float = 4.0
    max_consec_down: int = 3
    # 3 營收
    use_rev: bool = True
    min_yoy: float = 0.0
    # 4 毛利率
    use_gm: bool = True
    gm_tol: float = 1.0
    need_op_pos: bool = True
    # 5 股價止跌
    use_px: bool = True
    above_ma20: bool = True
    above_ma60: bool = False
    min_bounce: float = 3.0
    min_vol: float = 300.0


GROUPS = ["估值", "ROE", "營收", "毛利率", "股價"]


def enabled_groups(cfg: Cfg) -> dict:
    return {
        "估值": cfg.use_val,
        "ROE": cfg.use_roe,
        "營收": cfg.use_rev,
        "毛利率": cfg.use_gm,
        "股價": cfg.use_px,
    }


# ----------------------------------------------------------------------------
# FinMind 客戶端（含檔案快取）
# ----------------------------------------------------------------------------
class RateLimitError(Exception):
    """FinMind 次數用完（免費註冊約 600 次/小時）。"""


class FinMindError(Exception):
    pass


class FinMindClient:
    def __init__(self, token: str = "", cache_dir: Path = CACHE_DIR):
        self.token = (token or "").strip()
        self.cache_dir = Path(cache_dir)
        self.today = datetime.now(TZ).date()
        self.requests_made = 0
        self._cleanup()

    # -- cache helpers
    def _path(self, dataset: str, data_id: str | None, start: str | None) -> Path:
        key = f"{dataset}_{data_id or 'ALL'}_{start or 'NA'}_{self.today.isoformat()}.pkl"
        return self.cache_dir / key

    def is_cached(self, dataset: str, data_id: str | None = None, start: str | None = None) -> bool:
        return self._path(dataset, data_id, start).exists()

    def _cleanup(self) -> None:
        try:
            if not self.cache_dir.exists():
                return
            limit = time.time() - 3 * 86400
            for f in self.cache_dir.glob("*.pkl"):
                if f.stat().st_mtime < limit:
                    f.unlink(missing_ok=True)
        except Exception:
            pass

    # -- main entry
    def get(self, dataset: str, data_id: str | None = None, start: str | None = None) -> pd.DataFrame:
        path = self._path(dataset, data_id, start)
        if path.exists():
            try:
                return pd.read_pickle(path)
            except Exception:
                pass
        df = self._request(dataset, data_id, start)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_pickle(path)
        except Exception:
            pass
        return df

    def _request(self, dataset: str, data_id: str | None, start: str | None) -> pd.DataFrame:
        params = {"dataset": dataset}
        if data_id:
            params["data_id"] = data_id
        if start:
            params["start_date"] = start
        if self.token:
            params["token"] = self.token
        last_err = ""
        for attempt in range(3):
            try:
                r = requests.get(API_URL, params=params, timeout=30)
            except requests.RequestException as e:  # 不顯示 URL，避免洩漏 token
                last_err = type(e).__name__
                time.sleep(1.5 * (attempt + 1))
                continue
            self.requests_made += 1
            if r.status_code in (402, 429):
                raise RateLimitError("API 次數已達上限")
            try:
                j = r.json()
            except ValueError:
                last_err = f"HTTP {r.status_code}"
                time.sleep(1.5 * (attempt + 1))
                continue
            status = j.get("status", r.status_code)
            msg = str(j.get("msg", ""))
            if status in (402, 429) or "upper limit" in msg.lower() or "reach" in msg.lower():
                raise RateLimitError(msg or "API 次數已達上限")
            if status != 200:
                raise FinMindError(f"{dataset} {data_id or ''}: {msg}")
            return pd.DataFrame(j.get("data", []))
        raise FinMindError(f"{dataset} {data_id or ''}: 連線失敗 ({last_err})")


def ago(today, days: int) -> str:
    return (today - timedelta(days=days)).isoformat()


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


# ----------------------------------------------------------------------------
# 股票池
# ----------------------------------------------------------------------------
def load_universe(fx) -> pd.DataFrame:
    """上市櫃 4 碼普通股（排除 ETF、存託憑證等）。"""
    df = fx.get("TaiwanStockInfo")
    if df is None or df.empty:
        return pd.DataFrame(columns=["stock_id", "name", "industry"])
    d = df.copy()
    d["stock_id"] = d["stock_id"].astype(str)
    d = d[d["stock_id"].str.fullmatch(r"[1-9]\d{3}")]
    if "type" in d.columns:
        d = d[d["type"].isin(["twse", "tpex"])]
    if "industry_category" not in d.columns:
        d["industry_category"] = ""
    d["industry_category"] = d["industry_category"].fillna("").astype(str)
    d = d[~d["industry_category"].str.contains("ETF|Index|大盤|指數|存託憑證|受益證券|ETN", regex=True)]
    if "date" in d.columns:
        d = d.sort_values("date")
    d = d.drop_duplicates("stock_id", keep="last")
    out = pd.DataFrame(
        {
            "stock_id": d["stock_id"].values,
            "name": (d["stock_name"].fillna("") if "stock_name" in d.columns else pd.Series([""] * len(d), index=d.index)).values,
            "industry": d["industry_category"].values,
        }
    )
    return out.sort_values("stock_id").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 資料載入
# ----------------------------------------------------------------------------
def per_frame(fx, sid: str) -> pd.DataFrame:
    df = fx.get("TaiwanStockPER", sid, ago(fx.today, 1830))
    if df is None or df.empty or "PBR" not in df.columns or "PER" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    for c in ("PER", "PBR", "dividend_yield"):
        if c in d.columns:
            d[c] = to_num(d[c])
    return d.sort_values("date").reset_index(drop=True)


def price_frame(fx, sid: str) -> pd.DataFrame:
    # 一次抓 5 年：第二階段取尾端、河流圖用全部
    df = fx.get("TaiwanStockPrice", sid, ago(fx.today, 1830))
    if df is None or df.empty or "close" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["close"] = to_num(d["close"])
    if "Trading_Volume" in d.columns:
        d["Trading_Volume"] = to_num(d["Trading_Volume"])
    d = d[d["close"] > 0]
    return d.sort_values("date").reset_index(drop=True)


def fin_frame(fx, sid: str) -> pd.DataFrame:
    df = fx.get("TaiwanStockFinancialStatements", sid, ago(fx.today, 1100))
    return df if df is not None else pd.DataFrame()


def rev_frame(fx, sid: str) -> pd.DataFrame:
    df = fx.get("TaiwanStockMonthRevenue", sid, ago(fx.today, 620))
    return df if df is not None else pd.DataFrame()


# ----------------------------------------------------------------------------
# 指標計算
# ----------------------------------------------------------------------------
def stage1_metrics(d: pd.DataFrame, window_years: int = 5) -> dict | None:
    """PB 在自己歷史的位置 + ROE(TTM) 代理值。ROE_TTM = PBR / PER。"""
    if d is None or d.empty:
        return None
    last_date = d["date"].iloc[-1]
    w = d[d["date"] >= last_date - pd.DateOffset(years=window_years)]
    pbr_hist = w["PBR"].where(w["PBR"] > 0).dropna()
    cur = d["PBR"].iloc[-1]
    if len(pbr_hist) < 200 or not (cur > 0):
        return None
    pct = float((pbr_hist <= cur).mean() * 100)
    layer = int(sum(pct > c for c in LAYER_CUTS))

    per_cur = d["PER"].iloc[-1]
    per_hist = w["PER"].where(w["PER"] > 0).dropna()
    per_pct = float((per_hist <= per_cur).mean() * 100) if (per_cur > 0 and len(per_hist) >= 200) else NAN

    # FinMind 的 PER 在虧損時為 0 → 視為 TTM 虧損，ROE 設為 NaN
    roe_ttm = float(cur / per_cur * 100) if per_cur > 0 else NAN
    return {
        "pbr": float(cur),
        "pb_pct": pct,
        "layer": layer,
        "pb_vs_min": float(cur / pbr_hist.min()),
        "per": float(per_cur) if per_cur > 0 else NAN,
        "per_pct": per_pct,
        "roe_ttm": roe_ttm,
        "div_yield": float(d["dividend_yield"].iloc[-1]) if "dividend_yield" in d.columns else NAN,
        "data_date": last_date.strftime("%Y-%m-%d"),
    }


def price_metrics(p: pd.DataFrame) -> dict:
    if p is None or p.empty or len(p) < 65:
        return {}
    c = p["close"]
    close = float(c.iloc[-1])
    ma20 = float(c.tail(20).mean())
    ma60 = float(c.tail(60).mean())
    low60 = float(c.tail(60).min())
    vol20 = float(p["Trading_Volume"].tail(20).mean() / 1000) if "Trading_Volume" in p.columns else NAN
    return {
        "close": close,
        "vs_ma20": (close / ma20 - 1) * 100,
        "vs_ma60": (close / ma60 - 1) * 100,
        "bounce60": (close / low60 - 1) * 100,
        "vol20": vol20,
    }


def fin_metrics(f: pd.DataFrame, roe_ttm: float) -> dict:
    """毛利率、營益率、單季 EPS 走勢、單季年化 ROE（= ROE_TTM × 4×單季EPS / TTM EPS）。"""
    if f is None or f.empty or not {"date", "type", "value"} <= set(f.columns):
        return {}
    g = f.copy()
    g["value"] = to_num(g["value"])
    pv = g.pivot_table(index="date", columns="type", values="value", aggfunc="last").sort_index()

    def col(*names):
        for n in names:
            if n in pv.columns:
                return pv[n]
        return None

    rev, gp, op, eps = col("Revenue", "OperatingRevenue"), col("GrossProfit"), col("OperatingIncome"), col("EPS")
    if gp is None and rev is not None and col("CostOfGoodsSold") is not None:
        gp = rev - col("CostOfGoodsSold")

    out: dict = {"fin_date": str(pv.index[-1])}
    if rev is not None and gp is not None:
        gm = (gp / rev * 100).replace([np.inf, -np.inf], np.nan).dropna()
        if len(gm) >= 4:
            prev = gm.iloc[-5:-1]
            out["gm_now"] = float(gm.iloc[-1])
            out["gm_prev_avg"] = float(prev.mean())
    if rev is not None and op is not None:
        opm = (op / rev * 100).replace([np.inf, -np.inf], np.nan).dropna()
        if len(opm) >= 1:
            out["opm_now"] = float(opm.iloc[-1])
    if eps is not None:
        e = eps.dropna().values
        if len(e) >= 4:
            last = e[-5:] if len(e) >= 5 else e[-4:]
            consec = 0
            for i in range(len(last) - 1, 0, -1):
                if last[i] < last[i - 1]:
                    consec += 1
                else:
                    break
            ttm = float(e[-4:].sum())
            eps_q = float(e[-1])
            out["eps_q"] = eps_q
            out["eps_ttm"] = ttm
            out["eps_down_q"] = consec
            out["runrate_roe"] = float(roe_ttm * 4 * eps_q / ttm) if (ttm > 0 and roe_ttm == roe_ttm) else NAN
    return out


def rev_metrics(r: pd.DataFrame) -> dict:
    if r is None or r.empty or "revenue" not in r.columns:
        return {}
    d = r.copy()
    if {"revenue_year", "revenue_month"} <= set(d.columns):
        y, m = to_num(d["revenue_year"]), to_num(d["revenue_month"])
    else:  # 沒有年月欄位時，用公布日往前推一個月
        dt = pd.to_datetime(d["date"]) - pd.DateOffset(months=1)
        y, m = dt.dt.year, dt.dt.month
    d["ym"] = y * 12 + (m - 1)
    d["revenue"] = to_num(d["revenue"])
    d = d.dropna(subset=["ym", "revenue"])
    if d.empty:
        return {}
    s = d.groupby(d["ym"].astype(int))["revenue"].last()
    latest = int(s.index.max())

    def yoy(keys):
        cur = [s.get(k) for k in keys]
        prev = [s.get(k - 12) for k in keys]
        if any(v is None for v in cur + prev):
            return NAN
        pv = float(sum(prev))
        return (float(sum(cur)) / pv - 1) * 100 if pv > 0 else NAN

    return {
        "rev_latest_yoy": yoy([latest]),
        "rev_3m_yoy": yoy([latest - 2, latest - 1, latest]),
        "rev_month": f"{latest // 12}-{latest % 12 + 1:02d}",
    }


# ----------------------------------------------------------------------------
# 過濾：單一來源（表格與單檔診斷共用）
# ----------------------------------------------------------------------------
def _v(x):
    try:
        x = float(x)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _fmt(x, nd=2, suffix=""):
    x = _v(x)
    return "—" if x is None else f"{x:.{nd}f}{suffix}"


def evaluate_row(row: dict, cfg: Cfg) -> list[tuple]:
    """回傳 [(群組, 項目, 目前值, 門檻, 是否通過)]。缺資料一律視為未通過。"""
    r = row
    c: list[tuple] = []

    pct = _v(r.get("pb_pct"))
    c.append(("估值", f"PB 歷史分位（近 {cfg.window_years} 年）", _fmt(pct, 1, "%"), f"≤ {cfg.max_pct:g}%",
              pct is not None and pct <= cfg.max_pct))

    roe = _v(r.get("roe_ttm"))
    c.append(("ROE", "ROE(TTM) = PBR/PER", _fmt(roe, 1, "%"), f"≥ {cfg.min_roe:g}%", roe is not None and roe >= cfg.min_roe))
    rr = _v(r.get("runrate_roe"))
    c.append(("ROE", "最新單季年化 ROE", _fmt(rr, 1, "%"), f"≥ {cfg.min_runrate:g}%", rr is not None and rr >= cfg.min_runrate))
    dn = _v(r.get("eps_down_q"))
    c.append(("ROE", "單季 EPS 連續下滑季數", _fmt(dn, 0), f"< {cfg.max_consec_down}", dn is not None and dn < cfg.max_consec_down))

    y3 = _v(r.get("rev_3m_yoy"))
    c.append(("營收", "近 3 月營收年增率", _fmt(y3, 1, "%"), f"≥ {cfg.min_yoy:g}%", y3 is not None and y3 >= cfg.min_yoy))

    gm, gp = _v(r.get("gm_now")), _v(r.get("gm_prev_avg"))
    c.append(("毛利率", "最新季毛利率 vs 前四季平均", f"{_fmt(gm,1,'%')} vs {_fmt(gp,1,'%')}", f"≥ 前四季均 − {cfg.gm_tol:g}pt",
              gm is not None and gp is not None and gm >= gp - cfg.gm_tol))
    if cfg.need_op_pos:
        op = _v(r.get("opm_now"))
        c.append(("毛利率", "最新季營業利益率", _fmt(op, 1, "%"), "> 0%", op is not None and op > 0))

    if cfg.above_ma20:
        x = _v(r.get("vs_ma20"))
        c.append(("股價", "收盤相對 MA20", _fmt(x, 1, "%"), "≥ 0%", x is not None and x >= 0))
    if cfg.above_ma60:
        x = _v(r.get("vs_ma60"))
        c.append(("股價", "收盤相對 MA60", _fmt(x, 1, "%"), "≥ 0%", x is not None and x >= 0))
    b = _v(r.get("bounce60"))
    c.append(("股價", "距近 60 日低點反彈", _fmt(b, 1, "%"), f"≥ {cfg.min_bounce:g}%", b is not None and b >= cfg.min_bounce))
    vol = _v(r.get("vol20"))
    c.append(("股價", "20 日均量（張）", _fmt(vol, 0), f"≥ {cfg.min_vol:g}", vol is not None and vol >= cfg.min_vol))
    return c


def group_status(checks: list[tuple], cfg: Cfg) -> dict:
    en = enabled_groups(cfg)
    out = {}
    for g in GROUPS:
        items = [ok for (gg, _, _, _, ok) in checks if gg == g]
        out[g] = True if not en[g] else all(items)
    return out


def apply_filters(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    if df.empty:
        return df
    en = enabled_groups(cfg)
    n_en = sum(en.values())
    res = []
    for row in df.to_dict("records"):
        st_ = group_status(evaluate_row(row, cfg), cfg)
        fails = [g for g in GROUPS if en[g] and not st_[g]]
        res.append({"n_fail": len(fails), "n_pass": n_en - len(fails), "fail_txt": "、".join(fails)})
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(res)], axis=1)
    out["pass_txt"] = out["n_pass"].astype(str) + "/" + str(n_en)
    return out


def add_industry_context(df: pd.DataFrame) -> pd.DataFrame:
    """用本次掃描到的同業，算 PB 相對同業中位數；同業整體都在低層 = 產業性。"""
    if df.empty:
        return df
    d = df.copy()
    ok = d[d["pbr"].notna() & (d["industry"].fillna("") != "")]
    st_ = ok.groupby("industry").agg(ind_n=("pbr", "size"), ind_med_pbr=("pbr", "median"), ind_med_pct=("pb_pct", "median"))
    d = d.merge(st_, left_on="industry", right_index=True, how="left")
    d["pb_vs_ind"] = d["pbr"] / d["ind_med_pbr"]

    def kind(r):
        if not (r.get("ind_n") == r.get("ind_n")) or r["ind_n"] < 4:
            return "同業樣本不足"
        return "產業性" if r["ind_med_pct"] <= 30 else "個股性"

    d["kind"] = d.apply(kind, axis=1)
    return d


# ----------------------------------------------------------------------------
# 掃描
# ----------------------------------------------------------------------------
def prefilter(row: dict, cfg: Cfg) -> bool:
    """第一階段通過者才會抓第二階段資料（省 API）。"""
    pct, roe = _v(row.get("pb_pct")), _v(row.get("roe_ttm"))
    if pct is None:
        return False
    if cfg.use_val and pct > cfg.max_pct:
        return False
    if cfg.use_roe and (roe is None or roe < cfg.min_roe):
        return False
    return True


def stage2_for(fx, sid: str, roe_ttm: float, errors: list) -> dict:
    out: dict = {}
    for fn, args in (
        (lambda: price_metrics(price_frame(fx, sid)), ()),
        (lambda: fin_metrics(fin_frame(fx, sid), roe_ttm), ()),
        (lambda: rev_metrics(rev_frame(fx, sid)), ()),
    ):
        try:
            out.update(fn())
        except FinMindError as e:
            errors.append((sid, str(e)))
    out["stage2"] = True
    return out


def run_scan(fx, targets: pd.DataFrame, cfg: Cfg, max_stage1: int = 120, max_stage2: int = 50, progress=None) -> dict:
    rows: dict[str, dict] = {}
    errors: list = []
    skipped: list = []
    limited = False
    new_s1 = 0
    start1 = ago(fx.today, 1830)
    total = max(len(targets), 1)

    for i, t in enumerate(targets.to_dict("records")):
        sid = str(t["stock_id"])
        if progress:
            progress(0.55 * i / total, f"第一階段 {i + 1}/{total}：{sid} {t.get('name', '')}")
        cached = fx.is_cached("TaiwanStockPER", sid, start1)
        if not cached and new_s1 >= max_stage1:
            skipped.append(sid)
            continue
        try:
            d = per_frame(fx, sid)
        except RateLimitError:
            limited = True
            break
        except FinMindError as e:
            errors.append((sid, str(e)))
            continue
        if not cached:
            new_s1 += 1
        base = {"stock_id": sid, "name": t.get("name", ""), "industry": t.get("industry", ""), "stage2": False}
        m = stage1_metrics(d, cfg.window_years)
        if m is None:
            errors.append((sid, "歷史資料不足或無 PER/PBR"))
            continue
        base.update(m)
        rows[sid] = base

    cands = sorted([s for s, r in rows.items() if prefilter(r, cfg)], key=lambda s: rows[s]["pb_pct"])
    start2 = ago(fx.today, 1830)
    new_s2 = 0
    for j, sid in enumerate(cands):
        if limited:
            break
        if progress:
            progress(0.55 + 0.45 * j / max(len(cands), 1), f"第二階段 {j + 1}/{len(cands)}：{sid} {rows[sid]['name']}")
        cached = fx.is_cached("TaiwanStockPrice", sid, start2)
        if not cached and new_s2 >= max_stage2:
            skipped.append(sid + "(二)")
            continue
        try:
            rows[sid].update(stage2_for(fx, sid, rows[sid].get("roe_ttm", NAN), errors))
        except RateLimitError:
            limited = True
            break
        if not cached:
            new_s2 += 1

    if progress:
        progress(1.0, "完成")
    df = pd.DataFrame(list(rows.values()))
    if not df.empty:
        df = add_industry_context(df)
    return {
        "df": df,
        "limited": limited,
        "skipped": skipped,
        "errors": errors,
        "requests": fx.requests_made,
        "n_targets": len(targets),
        "n_stage2": sum(1 for r in rows.values() if r.get("stage2")),
        "ts": datetime.now(TZ).strftime("%Y-%m-%d %H:%M"),
    }


# ----------------------------------------------------------------------------
# 河流圖
# ----------------------------------------------------------------------------
BAND_LINES = ["#9ecae1", "#6baed6", "#3182bd", "#1f5f99", "#0b3d6e"]  # 單一色相，淺=便宜、深=昂貴
BAND_FILL = ["rgba(49,130,189,0.06)", "rgba(49,130,189,0.12)", "rgba(49,130,189,0.20)", "rgba(49,130,189,0.30)"]
PRICE_COLOR = "#e8590c"


def river_fig(per: pd.DataFrame, price: pd.DataFrame, col: str, label: str, window_years: int):
    if per.empty or price.empty:
        return None
    d = per.merge(price[["date", "close"]], on="date", how="inner")
    d = d[d[col] > 0].copy()
    if len(d) < 150:
        return None
    last = d["date"].max()
    h = d[d["date"] >= last - pd.DateOffset(years=window_years)].copy()
    h["base"] = h["close"] / h[col]  # BPS 或 TTM EPS
    qs = h[col].quantile([c / 100 for c in LAYER_CUTS]).values
    cur = float(h[col].iloc[-1])
    pct = float((h[col] <= cur).mean() * 100)
    layer = sum(pct > c for c in LAYER_CUTS)

    fig = go.Figure()
    for i, (q, c) in enumerate(zip(qs, LAYER_CUTS)):
        y = h["base"] * q
        fig.add_trace(
            go.Scatter(
                x=h["date"], y=y, mode="lines", name=f"P{c}（{q:.2f}x）",
                line=dict(width=1.2, color=BAND_LINES[i], shape="hv"),
                fill="tonexty" if i > 0 else None,
                fillcolor=BAND_FILL[i - 1] if i > 0 else None,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=h["date"], y=h["close"], mode="lines", name="股價",
            line=dict(width=2, color=PRICE_COLOR),
            hovertemplate="%{x|%Y-%m-%d}<br>收盤 %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=f"{label}河流圖 ｜ 現值 {cur:.2f}x ｜ 分位 {pct:.0f}% ｜ 第 {layer}/5 層（0 = 最便宜）", font=dict(size=15)),
        height=380, margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        yaxis=dict(title="股價（元）", gridcolor="rgba(128,128,128,0.15)", automargin=True),
        xaxis=dict(gridcolor="rgba(128,128,128,0.08)", automargin=True),
    )
    return fig


# ----------------------------------------------------------------------------
# 顯示
# ----------------------------------------------------------------------------
DISPLAY = [
    ("stock_id", "代號"), ("name", "名稱"), ("industry", "產業"), ("pass_txt", "通過"), ("fail_txt", "未通過"),
    ("close", "收盤"), ("pbr", "PBR"), ("pb_pct", "PB分位%"), ("layer", "層(0-5)"), ("per", "PER"),
    ("roe_ttm", "ROE(TTM)%"), ("runrate_roe", "單季年化ROE%"), ("eps_down_q", "EPS連降季"),
    ("rev_3m_yoy", "營收3M年增%"), ("rev_latest_yoy", "最新月年增%"),
    ("gm_now", "毛利率%"), ("gm_prev_avg", "前4季均%"), ("opm_now", "營益率%"),
    ("vs_ma20", "距MA20%"), ("bounce60", "距60日低%"), ("vol20", "20日均量(張)"),
    ("pb_vs_ind", "PB/同業中位"), ("kind", "性質"), ("div_yield", "殖利率%"), ("data_date", "資料日"),
]


def show_table(d: pd.DataFrame):
    cols = [c for c, _ in DISPLAY if c in d.columns]
    t = d[cols].rename(columns=dict(DISPLAY)).copy()
    for c in t.columns:
        if pd.api.types.is_float_dtype(t[c]):
            t[c] = t[c].round(2)
    st.dataframe(t, width="stretch", hide_index=True)


def get_token() -> str:
    try:
        tok = st.secrets["FINMIND_TOKEN"]
        if tok:
            return str(tok)
    except Exception:
        pass
    return st.sidebar.text_input("FinMind Token", type="password", help="部署時建議放在 Streamlit Secrets：FINMIND_TOKEN")


def parse_codes(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b[1-9]\d{3}\b", text or "")))


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def build_cfg() -> Cfg:
    sb = st.sidebar
    cfg = Cfg()
    cfg.window_years = sb.selectbox("估值歷史窗口（年）", [5, 3], index=0, help="PB 分位只跟自己這段期間比")
    with sb.expander("① 估值：河流層", expanded=True):
        cfg.use_val = st.checkbox("啟用", True, key="use_val")
        cfg.max_pct = st.slider("PB 分位 ≤ (%)", 1, 50, 15, help="10% 以下 ≈ 第 0 層；30% 以下 ≈ 第 0~1 層")
    with sb.expander("② ROE", expanded=False):
        cfg.use_roe = st.checkbox("啟用", True, key="use_roe")
        cfg.min_roe = st.number_input("ROE(TTM) ≥ (%)", value=6.0, step=0.5, help="用 PBR/PER 推得")
        cfg.min_runrate = st.number_input("最新單季年化 ROE ≥ (%)", value=4.0, step=0.5, help="抓出『TTM 好看、最近一季已經轉差』")
        cfg.max_consec_down = int(st.number_input("單季 EPS 連降達幾季剔除", value=3, min_value=2, max_value=4))
    with sb.expander("③ 營收", expanded=False):
        cfg.use_rev = st.checkbox("啟用", True, key="use_rev")
        cfg.min_yoy = st.number_input("近 3 月營收年增率 ≥ (%)", value=0.0, step=1.0)
    with sb.expander("④ 毛利率", expanded=False):
        cfg.use_gm = st.checkbox("啟用", True, key="use_gm")
        cfg.gm_tol = st.number_input("最新季毛利率可低於前四季均值 (百分點)", value=1.0, step=0.5)
        cfg.need_op_pos = st.checkbox("最新季營業利益率需 > 0", True)
    with sb.expander("⑤ 股價止跌", expanded=False):
        cfg.use_px = st.checkbox("啟用", True, key="use_px")
        cfg.above_ma20 = st.checkbox("收盤 ≥ MA20", True)
        cfg.above_ma60 = st.checkbox("收盤 ≥ MA60", False)
        cfg.min_bounce = st.number_input("距近 60 日收盤低點 ≥ (%)", value=3.0, step=1.0)
        cfg.min_vol = st.number_input("20 日均量 ≥ (張)", value=300.0, step=100.0)
    return cfg


def tab_scan(token: str, cfg: Cfg):
    sb = st.sidebar
    sb.markdown("---")
    mode = sb.radio("掃描範圍", ["自訂代號", "依產業", "全市場（慢）"])
    codes_text = ""
    inds: list[str] = []
    excl_fin = True
    if mode == "自訂代號":
        codes_text = sb.text_area("股票代號（空白、逗號或換行分隔）", "2387 2385 4915 8163 2474 2603", height=90)
    else:
        excl_fin = sb.checkbox("排除金融保險業（沒有毛利率）", True)
    if mode == "依產業":
        try:
            uni = load_universe(FinMindClient(token))
            inds = sb.multiselect("產業", sorted(uni["industry"].unique()))
        except (FinMindError, RateLimitError) as e:
            sb.warning(f"無法取得產業清單：{e}")
    max_s1 = int(sb.number_input("本次最多新抓第一階段檔數", 10, 3000, 120, step=10, help="每檔 1 次 API。已快取的不算。"))
    max_s2 = int(sb.number_input("本次最多新抓第二階段檔數", 5, 500, 50, step=5, help="每檔 3 次 API。"))
    run = sb.button("🔍 開始掃描", type="primary", width="stretch")

    if run:
        fx = FinMindClient(token)
        try:
            uni = load_universe(fx)
        except (FinMindError, RateLimitError) as e:
            uni = pd.DataFrame(columns=["stock_id", "name", "industry"])
            st.warning(f"股票基本資料抓取失敗（{e}），名稱與產業會是空白。")
        if mode == "自訂代號":
            codes = parse_codes(codes_text)
            base = pd.DataFrame({"stock_id": codes})
            targets = base.merge(uni, on="stock_id", how="left").fillna({"name": "", "industry": ""})
        elif mode == "依產業":
            targets = uni[uni["industry"].isin(inds)]
        else:
            targets = uni
        if mode != "自訂代號" and excl_fin and not targets.empty:
            targets = targets[~targets["industry"].str.contains("金融")]
        if targets.empty:
            st.error("沒有可掃描的股票。請確認代號或產業（或 Token 是否正確）。")
        else:
            bar = st.progress(0.0, text="開始…")
            res = run_scan(fx, targets.reset_index(drop=True), cfg, max_s1, max_s2,
                           progress=lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t))
            bar.empty()
            st.session_state["scan"] = res

    res = st.session_state.get("scan")
    if not res:
        st.info("左側設定好條件後按「開始掃描」。第一次建議先用預設的自訂代號跑，確認 Token 與流程正常。")
        return

    if res["limited"]:
        st.warning("FinMind API 次數用完了（免費註冊約 600 次/小時）。已掃描的結果都有快取，約一小時後再按一次掃描會接著跑。")
    if res["skipped"]:
        st.caption(f"有 {len(res['skipped'])} 檔因為「本次最多新抓」上限先略過，再按一次掃描會續掃。")
    df = res["df"]
    if df.empty:
        st.error("沒有取得任何資料。常見原因：Token 未填、API 次數用完、代號錯誤。")
        if res["errors"]:
            st.write(res["errors"][:10])
        return

    d = apply_filters(df, cfg)
    s2 = d[d["stage2"] == True]  # noqa: E712
    full = s2[s2["n_fail"] == 0].sort_values("pb_pct")
    near = s2[s2["n_fail"] == 1].sort_values("pb_pct")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("掃描檔數", len(d))
    c2.metric("進入第二階段", len(s2))
    c3.metric("全部通過", len(full))
    c4.metric("差一項", len(near))
    st.caption(f"掃描時間 {res['ts']} ｜ 本次實際 API 請求 {res['requests']} 次 ｜ 調整『估值窗口』或股票範圍後，請重新按掃描；其餘條件調整會即時套用。")

    t1, t2, t3 = st.tabs(["✅ 全部通過", "👀 差一項（觀察）", "📋 全部結果"])
    with t1:
        if full.empty:
            st.write("目前沒有符合全部條件的股票。可放寬 PB 分位或品質門檻，或擴大掃描範圍。")
        else:
            show_table(full)
            st.download_button("下載 CSV", full.to_csv(index=False).encode("utf-8-sig"), "screen_pass.csv", "text/csv")
    with t2:
        if near.empty:
            st.write("沒有差一項的股票。")
        else:
            st.caption("看『未通過』欄：是哪一項沒過，再決定是放寬門檻還是放棄。")
            show_table(near)
    with t3:
        show_table(d.sort_values(["stage2", "n_fail", "pb_pct"], ascending=[False, True, True]))
        if res["errors"]:
            with st.expander(f"資料問題 {len(res['errors'])} 筆"):
                st.write(res["errors"])


def tab_diag(token: str, cfg: Cfg):
    st.caption("單檔完整檢查：不管估值有沒有過，直接把五組過濾的數值攤開，並畫出 PB / PE 河流圖。每檔最多 5 次 API，同日重複查詢走快取。")
    sid = st.text_input("股票代號", "2387", max_chars=4).strip()
    go_btn = st.button("診斷", key="diag_btn")
    if go_btn:
        st.session_state["diag_sid"] = sid
    sid = st.session_state.get("diag_sid")
    if not sid:
        return
    fx = FinMindClient(token)
    errors: list = []
    try:
        per = per_frame(fx, sid)
        m1 = stage1_metrics(per, cfg.window_years)
        if m1 is None:
            st.error("找不到這檔的 PER/PBR 歷史資料（或資料不足 200 個交易日）。")
            return
        price = price_frame(fx, sid)
        row = {"stock_id": sid, **m1}
        row.update(price_metrics(price))
        row.update(fin_metrics(fin_frame(fx, sid), m1.get("roe_ttm", NAN)))
        row.update(rev_metrics(rev_frame(fx, sid)))
    except RateLimitError:
        st.warning("API 次數用完，請稍後再試。")
        return
    except FinMindError as e:
        st.error(str(e))
        return

    checks = evaluate_row(row, cfg)
    en = enabled_groups(cfg)
    stt = group_status(checks, cfg)
    n_en = sum(en.values())
    n_ok = sum(1 for g in GROUPS if en[g] and stt[g])
    st.subheader(f"{sid} 診斷結果：{n_ok}/{n_en} 組通過")
    tbl = pd.DataFrame(
        [{"群組": g + ("" if en[g] else "（未啟用）"), "項目": i, "目前值": v, "門檻": r, "結果": "✅" if ok else "❌"}
         for (g, i, v, r, ok) in checks]
    )
    st.dataframe(tbl, width="stretch", hide_index=True)

    extra = []
    if _v(row.get("eps_ttm")) is not None and _v(row.get("eps_q")) is not None:
        extra.append(f"最新單季 EPS {row['eps_q']:.2f}、近四季合計 {row['eps_ttm']:.2f}。TTM 本益比會在下一季財報後被『換掉最舊的一季』，最新一季若遠低於 TTM 平均，本益比看起來便宜只是假象。")
    if _v(row.get("per")) is not None:
        extra.append(f"目前 PER {row['per']:.1f}、PBR {row['pbr']:.2f}、殖利率 {_fmt(row.get('div_yield'),1,'%')}。")
    for t in extra:
        st.write("• " + t)

    c1, c2 = st.columns(2)
    for box, col, label in ((c1, "PBR", "股價淨值比"), (c2, "PER", "本益比")):
        fig = river_fig(per, price, col, label, cfg.window_years)
        with box:
            if fig is None:
                st.info(f"{label}：資料不足或為負值，無法繪圖。")
            else:
                st.plotly_chart(fig, width="stretch")
    st.caption(f"API 請求 {fx.requests_made} 次 ｜ 財報最新季 {row.get('fin_date', '—')} ｜ 營收最新月 {row.get('rev_month', '—')} ｜ 股價資料日 {row.get('data_date', '—')}")


def tab_help():
    st.markdown(
        """
### 這支工具在做什麼
**第一關：河流層。** 把每檔股票當天的 PBR，拿去跟「自己」近 3 或 5 年的 PBR 分佈比，得到分位數。
分位 ≤ 10% 約等於河流圖的第 0 層（最便宜）。這跟你的估值河流圖是同一個概念，只是用 P10/P30/P50/P70/P90 當河流線。

**第二關到第五關：品質過濾。** 低 PB 單獨使用，會把「價值陷阱」一起選進來。所以再加：

| 關卡 | 判斷 | 為什麼 |
|---|---|---|
| ROE | ROE(TTM)=PBR÷PER 有底；最新單季年化 ROE 有底；單季 EPS 沒有連續下滑 | PB 低是否合理取決於 ROE。TTM 數字落後，所以另外看最新單季 |
| 營收 | 近 3 月年增率 | 營收持續衰退，下一季獲利通常還會再差 |
| 毛利率 | 最新季不可明顯低於前四季均值、營益率 > 0 | 毛利率下滑 = 定價力或成本出問題 |
| 股價止跌 | 站上 MA20、離 60 日低點有反彈、量夠 | 避免在下跌途中接刀 |

**同業比較（資訊欄，不影響通過與否）：** `PB/同業中位` 與 `性質`。
只拿「本次掃描到的同業」比較，同業至少 4 檔才會判斷。同業 PB 分位中位數 ≤ 30% 標為「產業性」（整個產業都在低檔），否則是「個股性」。想看得準，請用「依產業」模式掃整個產業。

### 限制（請務必知道）
- **PB 分位只跟自己的歷史比。** 如果這 3~5 年剛好是該公司的高峰期，現在「歷史低」可能只是回到正常。
- **ROE(TTM) 是 PBR/PER 的代理值**，不是財報公布的 ROE；虧損時 PER 為 0，會被視為 TTM 虧損而剔除。
- **財報、營收是 FinMind 當下抓得到的最新資料**，公布前有時間差。
- **金融保險業沒有毛利率**，預設排除。
- 這是篩選器，不是進場訊號。建議把通過清單再拿去做回測：統計歷史上同樣條件出現後 60、120 個交易日的報酬與最大回撤。

### API 次數
免費註冊等級約 600 次/小時。第一階段每檔 1 次、第二階段每檔 3 次。結果快取在本機資料夾（Streamlit Cloud 重啟後會清空），
所以全市場可以分多個小時、多次按掃描慢慢補完。
        """
    )


def main():
    st.set_page_config(page_title="估值河流選股器", page_icon="📉", layout="wide")
    st.title("📉 估值河流選股器")
    st.caption("PB 河流層 × ROE / 營收 / 毛利率 / 股價止跌 四道過濾。資料：FinMind。僅供研究，非買賣建議。")
    token = get_token()
    if not token:
        st.sidebar.info("沒填 Token 也能跑，但次數上限較低（約 300 次/小時）。")
    cfg = build_cfg()
    tabs = st.tabs(["🔍 選股", "🩺 單檔診斷", "📖 說明"])
    with tabs[0]:
        tab_scan(token, cfg)
    with tabs[1]:
        tab_diag(token, cfg)
    with tabs[2]:
        tab_help()


if __name__ == "__main__":
    main()
