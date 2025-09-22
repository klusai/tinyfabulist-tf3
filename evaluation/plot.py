import matplotlib.pyplot as plt

# Your log data
data = [
    (27600, 0.9517, 2.5901),
    (15400, 0.9777, 2.6583),
    (18000, 0.9825, 2.6712),
    (20600, 0.9952, 2.7053),
    (16000, 1.0015, 2.7223),
    (14600, 1.0103, 2.7464),
    (17600, 1.0129, 2.7537),
    (12800, 1.0267, 2.7918),
    (10600, 1.0439, 2.8403),
    (6200,  1.0772, 2.9365),
    (5200,  1.1188, 3.0611),
    (4600,  1.1536, 3.1697),
]

# Sort by checkpoint (training step)
data.sort(key=lambda x: x[0])

steps = [d[0] for d in data]
ce = [d[1] for d in data]
ppl = [d[2] for d in data]

# Plot CE
plt.figure(figsize=(8,5))
plt.plot(steps, ce, marker="o", label="Cross-Entropy (CE)")
plt.plot(steps, ppl, marker="s", label="Perplexity (PPL)")
plt.xlabel("Checkpoint (Step)")
plt.ylabel("Metric Value")
plt.title("Training Evolution: CE & PPL vs Checkpoints")
plt.legend()
plt.grid(True)
plt.show()
