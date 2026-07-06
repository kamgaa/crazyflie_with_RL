import matplotlib
matplotlib.use("Agg")               # headless 저장용 (SSH에서도 됨)
import matplotlib.pyplot as plt

steps = [20,40,60,80,100,120,140,160,180,200,220,240,260,280,300]
steps = [s*1000 for s in steps]
resid = [0.0278,0.0275,0.0271,0.0267,0.0269,0.0292,0.0310,0.0323,
         0.0392,0.0562,0.0612,0.0819,0.1046,0.1155,0.1037]
floor = 0.0277

plt.figure(figsize=(8,5))
plt.plot(steps, resid, "o-", label="PID + residual")
plt.axhline(floor, ls="--", color="gray", label=f"PID floor ({floor:.4f})")
plt.axvspan(40000, 100000, alpha=0.12, color="green")   # WIN 구간
plt.annotate("best ~80k", (80000, 0.0267), textcoords="offset points",
             xytext=(0,-25), ha="center", arrowprops=dict(arrowstyle="->"))
plt.xlabel("timesteps"); plt.ylabel("steady-state pos_err [m] (lower is best)")
plt.title("10g residual: WIN then collapse")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("/home/mrl_6534/gwpark/crazyflie_RL/learning_curve.png", dpi=130)
print("saved learning_curve.png")
