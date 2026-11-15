from pathlib import Path

from lab_compton.systematics import (
    window_systematics, model_systematics,
    print_systematics, combine_systematics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "Optimization" / "Scintillator1" / "Spe"

df_w, sp_w = window_systematics(DATA_DIR, min_channel=30)
df_m, sp_m = model_systematics(DATA_DIR, min_channel=30)

print_systematics(sp_w, "window")
print_systematics(sp_m, "model")

combined = combine_systematics(sp_w, sp_m)
print(combined.to_string(index=False))
