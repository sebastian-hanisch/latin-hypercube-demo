"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

NETZE = {"2x3": ("gitter", 2, 3), "3x3": ("gitter", 3, 3), "3x4": ("gitter", 3, 4), "hantel": ("hantel", 0, 0)}
DEFAULT_NETZ = "3x4"
LIVE_EDGE_LIMIT = 17          # Aufzählung aller Kantenmengen live bis hierher (2^17 = 131 072, rund 1 s); eigenes Netz ebenso
CUSTOM_EDGE_LIMIT = 17
CUSTOM_NODE_LIMIT = 14

P_MIN, P_MAX, DEFAULT_P = 0.50, 0.99, 0.70
N_CHOICES = (10, 30, 100, 300)            # Punkte je Schätzung (gerade: die Spiegelpaare brauchen N/2 Paare)
N_MIN, N_MAX, DEFAULT_N = N_CHOICES[0], N_CHOICES[-1], 100
SEED_MAX, DEFAULT_SEED = 999999, 35
REPS = 400                                # Wiederholungen je Schätzung (feste Ströme); Rauschen der Varianzverhältnisse rund 1/sqrt(REPS/2) ≈ 7 %
BLOCKS = 10                               # Blöcke für das Intervall aus wiederholten Latin-Hypercube-Stichproben
T_BLOCKS = 2.262                          # t-Quantil (95 %, 9 Freiheitsgrade)

# Experimente
SWEEP_NS = (10, 30, 100, 300)
SWEEP_REPS = 400
SHARE_PS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
DIM_PS = (0.7, 0.9, 0.99)
DIM_MS = (4, 8, 16, 32, 60)
RARE_P = 0.99
RARE_NS = (10, 30, 100, 300, 1000)
RARE_REPS = 300
INT_NS = (30, 100, 300)
INT_REPS = 400

Z95 = 1.959963984540054

COLOR_PLAIN = "#4c78a8"
COLOR_LHS = "#2F6B65"
COLOR_ANTI = "#E08E0B"
COLOR_STRAT = "#8e6bb0"
COLOR_TRUTH = "#222222"
COLOR_BAD = "#C0392B"
COLOR_OTHER = "rgba(150,150,150,0.55)"

PRESETS = {
    "Standardfall (Voreinstellung)": {"netz": DEFAULT_NETZ, "p": DEFAULT_P, "n": DEFAULT_N, "seed": DEFAULT_SEED},
    "Eine kritische Brücke": {"netz": "hantel", "p": 0.90, "n": 100, "seed": DEFAULT_SEED},
    "Seltener Ausfall im Gitter": {"netz": DEFAULT_NETZ, "p": 0.99, "n": 300, "seed": DEFAULT_SEED},
    "Seltene Brücke, zu wenige Punkte": {"netz": "hantel", "p": 0.99, "n": 30, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Gitter 3×4, jede Kante hält mit 70 %, 100 Punkte: Latin Hypercube senkt die Varianz etwa auf die Hälfte (Vorhersage 0,53; die einzelne Messung mit 400 Wiederholungen zeigt 0,63).",
    "Eine kritische Brücke": "Zwei Dreiecke, durch eine einzige Brücke verbunden: eine wirksame Eingabe, der Gewinn ist groß, und eine Schichtung nach der Brücke holt das meiste heraus.",
    "Seltener Ausfall im Gitter": "Gitter 3×4 bei p = 0,99: kein Gewinn, der additive Anteil der Varianz ist winzig.",
    "Seltene Brücke, zu wenige Punkte": "Die Brücke fällt nur mit 1 % aus; bei 30 Punkten enthält nur die oberste Schicht den Ausfallbereich, und davon nur den Anteil N·q = 0,3: der volle Gewinn erscheint erst ab etwa 100 Punkten.",
}
