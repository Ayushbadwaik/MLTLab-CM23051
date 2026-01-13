# Student Marks Prediction with User Input, Prediction & 3 Graphs

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# -----------------------------
# 1. Take Training Data Input
# -----------------------------
n = int(input("Enter number of students for training: "))

hours = []
marks = []

for i in range(n):
    print(f"\nStudent {i+1}")
    h = float(input("  Enter study hours: "))
    m = float(input("  Enter marks: "))
    hours.append(h)
    marks.append(m)

X = np.array(hours).reshape(-1, 1)
y = np.array(marks).reshape(-1, 1)

# -----------------------------
# 2. Build Linear Regression Model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

# -----------------------------
# 3. Compile Model
# -----------------------------
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='mean_squared_error'
)

# -----------------------------
# 4. Train Model
# -----------------------------
history = model.fit(X, y, epochs=500, verbose=0)

# -----------------------------
# 5. Predict for New Student
# -----------------------------
new_hours = float(input("\nEnter study hours to predict marks: "))
predicted_mark = model.predict(np.array([[new_hours]]))

print(f"\nPredicted Marks for {new_hours} hours = {predicted_mark[0][0]:.2f}")

# -----------------------------
# 6. Predictions for Training Data
# -----------------------------
y_pred = model.predict(X)

# -----------------------------
# 7. Graphs (3 Visualizations)
# -----------------------------

# Graph 1: Training Data
plt.figure()
plt.scatter(X, y, color='blue')
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Graph 1: Study Hours vs Actual Marks")
plt.show()

# Graph 2: Regression Line
plt.figure()
plt.scatter(X, y, label="Actual Marks")
plt.plot(X, y_pred, color='red', label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Graph 2: Regression Model")
plt.legend()
plt.show()

# Graph 3: New Prediction
plt.figure()
plt.scatter(X, y, label="Training Data")
plt.scatter(new_hours, predicted_mark, color='green', s=100, label="Predicted Point")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Graph 3: New Student Prediction")
plt.legend()
plt.show()
