"""Kern der Demo (nur numpy).

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):
  Zufallsstrom   stream_seed, uniforms, iid_uniforms, lhs_uniforms   SplitMix64, zählerbasiert; Latin Hypercube: je Kante eine Permutation der N Schichten plus Zitter
  Ziel           connected, enumerate_x, lookup                      X = Netz hängt zusammen; Aufzählung aller 2^m Kantenmengen, Nachschlagen beim Ziehen
  Wahrheit       main_effects, symmetric_share                       exakt: E X, Var X, Haupteffekt jeder Kante, Summe der Haupteffekte (= Anteil, den Latin Hypercube entfernt)
  Schätzer       iid_mean, lhs_mean, anti_mean, strat_mean            einfach, Latin Hypercube, Spiegelpaare, Schichtung nach einer Kante (nachgewichtet)
  Wiederholungen estimates, blocks_interval                          feste Ströme je (Einstellung, Wiederholung)
Zufall kommt nur über einen übergebenen Strom, nie aus einem globalen Generator.

Ziel als Funktion f(u) der Gleichverteilten u_e je Kante (Kante e hält, wenn u_e < p). Latin Hypercube mit N Punkten: u_{i,e} = (pi_e(i) + v_{i,e}) / N, pi_e eine zufällige
Permutation von 0..N-1, v gleichverteilt. Owen (1997): Var(Mittel) ist asymptotisch (Var f - Summe_e Var(E[f | u_e])) / N.
"""

from __future__ import annotations

import math

import numpy as np

import lhy_constants as C

GAMMA = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1


# --- Zufallsstrom -------------------------------------------------------------------------------------------------------------------------

def mix_int(z: int) -> int:
    """SplitMix64-Mischfunktion mit Python-Ganzzahlen."""
    z &= MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK


def _mix_arr(z: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def stream_seed(*keys: int) -> int:
    """Seed eines Stroms aus einem Schlüsseltupel (Einstellung, Wiederholung, ...)."""
    s = 0x1234_5678_9ABC_DEF0
    for k in keys:
        s = mix_int(s ^ mix_int(int(k) + GAMMA))
    return s


def uniforms(seed: int, count: int) -> np.ndarray:
    """`count` gleichverteilte Zahlen in [0, 1) (53 Bit) aus dem Strom `seed`; Ausgabe i hängt nur von (seed, i) ab."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = _mix_arr(np.uint64(seed) + idx * np.uint64(GAMMA))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def iid_uniforms(seed: int, N: int, m: int) -> np.ndarray:
    """(N, m) unabhängige Gleichverteilte: Zahl Nr. i*m + e des Stroms ist die von Kante e in Ziehung i."""
    return uniforms(seed, N * m).reshape(N, m)


def lhs_uniforms(seed: int, N: int, m: int) -> np.ndarray:
    """(N, m) Latin-Hypercube-Punkte: die ersten N*m Zahlen des Stroms sind der Zitter v, die zweiten N*m Zahlen ergeben (spaltenweise sortiert) je Kante eine Permutation der Schichten."""
    v = uniforms(seed, 2 * N * m)
    jitter = v[: N * m].reshape(N, m)
    perm = np.argsort(v[N * m:].reshape(N, m), axis=0)
    return (perm + jitter) / N


# --- Ziel ---------------------------------------------------------------------------------------------------------------------------------

def connected(present: np.ndarray, n: int, edges) -> np.ndarray:
    """Zusammenhang je Zeile (`present` ist (B, m) bool): Kleinste-Marke-Ausbreitung, zusammenhängend genau dann, wenn alle Marken 0 sind."""
    B = present.shape[0]
    if n <= 1:
        return np.ones(B, dtype=bool)
    lab = np.tile(np.arange(n, dtype=np.int16), (B, 1))
    while True:
        before = int(lab.sum())
        for e, (u, v) in enumerate(edges):
            low = np.minimum(lab[:, u], lab[:, v])
            pe = present[:, e]
            lab[:, u] = np.where(pe, low, lab[:, u])
            lab[:, v] = np.where(pe, low, lab[:, v])
        if int(lab.sum()) == before:
            break
    return lab.max(axis=1) == 0


def enumerate_x(n: int, edges, chunk: int = 1 << 15) -> tuple[np.ndarray, np.ndarray]:
    """Alle 2^m Kantenmengen in Maskenreihenfolge (Kante e = Bit e): X (float) und Zahl haltender Kanten k."""
    m = len(edges)
    shifts = np.arange(m, dtype=np.uint32)
    xs, ks = [], []
    for start in range(0, 1 << m, chunk):
        masks = np.arange(start, min(start + chunk, 1 << m), dtype=np.uint32)
        present = ((masks[:, None] >> shifts) & 1).astype(bool)
        xs.append(connected(present, n, edges).astype(np.float64))
        ks.append(present.sum(axis=1))
    return np.concatenate(xs), np.concatenate(ks)


def lookup(X: np.ndarray, u: np.ndarray, p: float) -> np.ndarray:
    """X für Punkte u (N, m): Kante e hält, wenn u_e < p; Nachschlagen in der Aufzählung über die Bitmaske."""
    m = u.shape[1]
    mask = ((u < p).astype(np.uint32) << np.arange(m, dtype=np.uint32)).sum(axis=1)
    return X[mask]


# --- Wahrheit durch Aufzählung ---------------------------------------------------------------------------------------------------------

def main_effects(X: np.ndarray, k: np.ndarray, m: int, p: float) -> dict:
    """Exakt aus der Aufzählung: E X, Var X, je Kante e der Haupteffekt Var(E[X | u_e]) = p (1-p) (a1_e - a0_e)^2 mit a1 = E[X | e hält], a0 = E[X | e fällt aus].

    `share` ist der Anteil an Var X, `add_share` die Summe der Anteile (der additive Anteil der Varianz, den Latin Hypercube entfernt); bei Var X = 0 sind sie NaN."""
    w = p**k * (1.0 - p) ** (m - k)
    ex = float(np.sum(w * X))
    var = float(np.sum(w * X * X) - ex**2)
    masks = np.arange(len(X), dtype=np.uint32)
    eff = np.empty(m)
    for e in range(m):
        up = ((masks >> e) & 1).astype(bool)
        a1 = float(np.sum(w[up] * X[up]) / np.sum(w[up]))
        a0 = float(np.sum(w[~up] * X[~up]) / np.sum(w[~up]))
        eff[e] = p * (1.0 - p) * (a1 - a0) ** 2
    if var > 1e-15:
        share = eff / var
        add = float(share.sum())
        top = int(np.argmax(share))
    else:
        share = np.full(m, np.nan)
        add, top = float("nan"), 0
    return {"EX": ex, "Var": var, "eff": eff, "share": share, "add_share": add, "top": top, "top_share": float(share[top]) if var > 1e-15 else float("nan")}


def symmetric_target(kind: str, k: int, m: int) -> float:
    """Symmetrische Ziele als Funktion der Zahl k haltender Kanten (0/1): Kreis (höchstens eine fällt aus), Reihe (alle halten), Parallel (mindestens eine hält), Mehrheit (mindestens die Hälfte)."""
    if kind == "kreis":
        return 1.0 if k >= m - 1 else 0.0
    if kind == "reihe":
        return 1.0 if k == m else 0.0
    if kind == "parallel":
        return 1.0 if k >= 1 else 0.0
    if kind == "mehrheit":
        return 1.0 if 2 * k >= m else 0.0
    raise ValueError(kind)


def _binom_pmf(m: int, p: float) -> np.ndarray:
    return np.array([math.comb(m, j) * p**j * (1.0 - p) ** (m - j) for j in range(m + 1)])


def symmetric_share(kind: str, m: int, p: float) -> float:
    """Additiver Anteil m * Var(E[f | u_1]) / Var f eines symmetrischen Ziels bei m Kanten, binomial exakt; NaN, wenn f (fast) konstant ist."""
    pk = _binom_pmf(m, p)
    fk = np.array([symmetric_target(kind, j, m) for j in range(m + 1)])
    ex = float(np.sum(pk * fk))
    var = float(np.sum(pk * fk**2) - ex**2)
    if var < 1e-12:
        return float("nan")
    pj = _binom_pmf(m - 1, p)
    a1 = float(np.sum(pj * fk[1:]))
    a0 = float(np.sum(pj * fk[:-1]))
    return float(m * p * (1.0 - p) * (a1 - a0) ** 2 / var)


# --- Schätzer -----------------------------------------------------------------------------------------------------------------------------

def iid_mean(X: np.ndarray, p: float, seed: int, N: int, m: int) -> tuple[float, float]:
    """Einfaches Mittel aus N unabhängigen Punkten und der naive Standardfehler (Stichprobenstreuung / sqrt N)."""
    x = lookup(X, iid_uniforms(seed, N, m), p)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(N))


def lhs_mean(X: np.ndarray, p: float, seed: int, N: int, m: int) -> tuple[float, float]:
    """Mittel aus einem Latin Hypercube mit N Punkten und derselbe naive Standardfehler (der unter Latin Hypercube nicht gilt)."""
    x = lookup(X, lhs_uniforms(seed, N, m), p)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(N))


def anti_mean(X: np.ndarray, p: float, seed: int, N: int, m: int) -> float:
    """N/2 Spiegelpaare (u, 1 - u): Mittel über alle N Auswertungen."""
    u = iid_uniforms(seed, N // 2, m)
    return float(0.5 * (lookup(X, u, p).mean() + lookup(X, 1.0 - u, p).mean()))


def strat_mean(X: np.ndarray, p: float, seed: int, N: int, m: int, edge: int) -> float:
    """Schichtung nach einer Kante: nu = round(N p) Punkte mit haltender Kante (u in [0, p)), der Rest mit ausgefallener (u in [p, 1)), übrige Kanten unabhängig;
    nachgewichtet p * Mittel(hält) + (1 - p) * Mittel(fällt aus), erwartungstreu auch wenn N p nicht ganzzahlig ist."""
    nu = min(max(int(round(N * p)), 1), N - 1)
    u = iid_uniforms(seed, N, m)
    u[:nu, edge] *= p
    u[nu:, edge] = p + u[nu:, edge] * (1.0 - p)
    x = lookup(X, u, p)
    return float(p * x[:nu].mean() + (1.0 - p) * x[nu:].mean())


def blocks_interval(X: np.ndarray, p: float, seed_key: tuple, N: int, m: int) -> tuple[float, float]:
    """BLOCKS unabhängige Latin Hypercubes der Größe N/BLOCKS: Mittel der Blockmittel und Halbbreite des 95-%-Intervalls aus der Streuung der Blockmittel (t-Quantil)."""
    bs = N // C.BLOCKS
    bm = np.array([lookup(X, lhs_uniforms(stream_seed(*seed_key, b), bs, m), p).mean() for b in range(C.BLOCKS)])
    return float(bm.mean()), float(C.T_BLOCKS * bm.std(ddof=1) / np.sqrt(C.BLOCKS))


def estimates(X: np.ndarray, p: float, key: tuple, reps: int, N: int, m: int, edge: int, truth: float, with_blocks: bool = False) -> dict:
    """Je Wiederholung (fester Strom aus `key` und der Nummer) vier Schätzungen mit N Punkten (einfach, Latin Hypercube, Spiegelpaare, Schichtung nach `edge`),
    der naive Standardfehler beider Mittel und die Überdeckung der naiven 95-%-Intervalle (und, wenn verlangt und N >= 2 BLOCKS, des Intervalls aus Blöcken)."""
    out = {k: np.empty(reps) for k in ("iid", "lhs", "anti", "strat")}
    se_iid, se_lhs = np.empty(reps), np.empty(reps)
    cover = {"iid": 0, "lhs": 0, "blocks": 0}
    use_blocks = with_blocks and N >= 2 * C.BLOCKS
    for rep in range(reps):
        out["iid"][rep], se_iid[rep] = iid_mean(X, p, stream_seed(*key, rep, 1), N, m)
        out["lhs"][rep], se_lhs[rep] = lhs_mean(X, p, stream_seed(*key, rep, 2), N, m)
        out["anti"][rep] = anti_mean(X, p, stream_seed(*key, rep, 3), N, m)
        out["strat"][rep] = strat_mean(X, p, stream_seed(*key, rep, 4), N, m, edge)
        cover["iid"] += abs(out["iid"][rep] - truth) <= C.Z95 * se_iid[rep]
        cover["lhs"] += abs(out["lhs"][rep] - truth) <= C.Z95 * se_lhs[rep]
        if use_blocks:
            mean_b, half = blocks_interval(X, p, (*key, rep, 5), N, m)
            cover["blocks"] += abs(mean_b - truth) <= half
    return {"est": out, "se_iid": se_iid, "se_lhs": se_lhs, "coverage": {k: v / reps for k, v in cover.items()}, "blocks": use_blocks}
