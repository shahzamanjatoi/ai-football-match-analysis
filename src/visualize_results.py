import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("results/player_statistics.csv")

# -------------------------------
# Basic statistics
# -------------------------------
total_frames = len(df)
duration = df["Time_Seconds"].max()

min_players = df["Player_Count"].min()
max_players = df["Player_Count"].max()
avg_players = df["Player_Count"].mean()

ball_frames = df["Ball_Detected"].sum()
ball_percentage = (ball_frames / total_frames) * 100

print("===================================")
print("FUTSAL ANALYSIS RESULTS")
print("===================================")

print(f"Total Frames: {total_frames}")
print(f"Video Duration: {duration:.2f} seconds")
print(f"Minimum Players Detected: {min_players}")
print(f"Maximum Players Detected: {max_players}")
print(f"Average Players Detected: {avg_players:.2f}")
print(f"Ball Detected Frames: {ball_frames}")
print(f"Ball Detection Rate: {ball_percentage:.2f}%")

print("===================================")


# -------------------------------
# Graph 1: Player Count vs Time
# -------------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    df["Time_Seconds"],
    df["Player_Count"],
    linewidth=1.5
)

plt.xlabel("Time (seconds)")
plt.ylabel("Number of Players")
plt.title("Player Count Over Time")
plt.grid(True)

plt.tight_layout()
plt.savefig("results/player_count_vs_time.png", dpi=300)
plt.show()


# -------------------------------
# Graph 2: Ball Detection
# -------------------------------
plt.figure(figsize=(10, 4))

plt.plot(
    df["Time_Seconds"],
    df["Ball_Detected"],
    linewidth=1.5
)

plt.xlabel("Time (seconds)")
plt.ylabel("Ball Detected")
plt.title("Ball Detection Over Time")
plt.yticks([0, 1], ["Not Detected", "Detected"])
plt.grid(True)

plt.tight_layout()
plt.savefig("results/ball_detection_vs_time.png", dpi=300)
plt.show()


# -------------------------------
# Graph 3: Player Count Distribution
# -------------------------------
plt.figure(figsize=(8, 5))

df["Player_Count"].value_counts().sort_index().plot(
    kind="bar"
)

plt.xlabel("Number of Players")
plt.ylabel("Number of Frames")
plt.title("Distribution of Detected Player Count")
plt.grid(axis="y")

plt.tight_layout()
plt.savefig("results/player_count_distribution.png", dpi=300)
plt.show()

print("\nGraphs saved successfully in results/")
