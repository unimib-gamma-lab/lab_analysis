'''
Quick check: read test2.Spe and print its main properties.
'''

from maestro import read_spe
import matplotlib.pyplot as plt


s = read_spe("test2.Spe")

print(f"file         : {s.path}")
print(f"description  : {s.description}")
print(f"date         : {s.date}   (raw: '{s.date_str}')")
print(f"live time    : {s.live:.2f} s")
print(f"real time    : {s.real:.2f} s")
print(f"dead time    : {100 * (1 - s.live / s.real):.2f} %")
print(f"channels     : {s.channels.size}  (from {s.channels[0]} to {s.channels[-1]})")
print(f"total counts : {s.counts.sum()}")
print(f"counts array : shape={s.counts.shape}, dtype={s.counts.dtype}")
print(f"counts[:10]  : {s.counts[:10]}")

plt.figure(figsize=(10, 4))
plt.step(s.channels, s.counts, where="mid", color="steelblue")
plt.xlabel("channel")
plt.ylabel("counts")
plt.title(f"{s.path.name}  —  live {s.live:.1f} s")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
