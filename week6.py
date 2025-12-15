import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ==========================================
# 1) Function Generate Dataset (คงเดิม)
# ==========================================
def generate_spiral_data(n_points, n_turns, noise=0.2):
    X = []
    y = []
    for class_idx in range(2):
        i = np.arange(n_points)
        theta = (i / n_points) * (n_turns * 2 * np.pi) + (class_idx * np.pi)
        r = (i / n_points) * n_turns 
        r = r + np.random.normal(0, noise, n_points)
        x_val = r * np.cos(theta)
        y_val = r * np.sin(theta)
        data = np.column_stack((x_val, y_val))
        labels = class_idx * np.ones(n_points)
        X.append(data)
        y.append(labels)
    return np.concatenate(X), np.concatenate(y)

# ==========================================
# [หัวใจสำคัญ] Feature Engineering แบบวนลูป
# ==========================================
def add_features(X):
    x = X[:, 0]
    y = X[:, 1]
    radius = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)
    
    # TRICK: เราต้องแปลง Radius ให้เป็น "คลื่น" (Periodic) ด้วย
    # เพื่อให้ Model เห็นว่า รัศมี 1, 2, 3, 4... มีแพทเทิร์นซ้ำเดิม
    # เราใช้ sin(2*pi*radius) เพราะในสูตร Generate วงกลมครบ 1 รอบทุกๆ r เพิ่มขึ้น 1 หน่วย
    sin_r = np.sin(2 * np.pi * radius)
    cos_r = np.cos(2 * np.pi * radius)
    
    # เราส่งค่าเข้าไปแค่ 4 ตัวนี้ (ตัด x, y, radius ดิบออกเลย เพื่อบังคับให้ Model จำแค่คาบคลื่น)
    # วิธีนี้จะทำให้ Model มองว่า วงใน (Train) กับ วงนอก (Test) "หน้าตาเหมือนกันเป๊ะ"
    return np.column_stack((np.sin(angle), np.cos(angle), sin_r, cos_r))

# ==========================================
# 2) เตรียมข้อมูล (Train 2 รอบ / Test 4 รอบ)
# ==========================================
np.random.seed(42)

# Train แค่ 2 รอบ (ตามโจทย์)
X_train_raw, y_train = generate_spiral_data(n_points=1000, n_turns=2, noise=0.1)

# Test โหดๆ 4 รอบ
X_test_raw, y_test = generate_spiral_data(n_points=1000, n_turns=4, noise=0.1)

# แปลง Features
X_train = add_features(X_train_raw)
X_test = add_features(X_test_raw)

# ==========================================
# 3) ออกแบบ Model (Input Dim = 4)
# ==========================================
model = Sequential([
    # Input dim = 4 (sin_ang, cos_ang, sin_r, cos_r)
    Dense(64, input_dim=4, activation='relu'), 
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Start Training...")
# Train กับข้อมูลแค่ 2 รอบ
model.fit(X_train, y_train, epochs=200, batch_size=64, verbose=0)
print("Training Completed.")

# ==========================================
# 5) ทดสอบ (Inference)
# ==========================================
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\nEvaluation Results:")
print(f"Training Accuracy (2 turns): {train_acc*100:.2f}%")
print(f"Testing Accuracy (4 turns):  {test_acc*100:.2f}%")
print("Note: ตอนนี้ Testing Accuracy ควรจะสูงปรี๊ดเท่า Training แล้ว")

# ==========================================
# 6) Plot ผลลัพธ์
# ==========================================
def plot_decision_boundary(X_raw, y, model, title, ax):
    x_min, x_max = X_raw[:, 0].min() - 0.5, X_raw[:, 0].max() + 0.5
    y_min, y_max = X_raw[:, 1].min() - 0.5, X_raw[:, 1].max() + 0.5
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points_raw = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_features = add_features(mesh_points_raw) 
    
    Z = model.predict(mesh_points_features, verbose=0)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap=plt.cm.RdBu, alpha=0.6)
    ax.scatter(X_raw[:, 0], X_raw[:, 1], c=y, cmap=plt.cm.RdBu_r, edgecolors='k', s=20)
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

plot_decision_boundary(X_train_raw, y_train, model, 
                       f"Figure 1: Training Data (2 Turns)\nAcc: {train_acc*100:.1f}%", 
                       axes[0])

plot_decision_boundary(X_test_raw, y_test, model, 
                       f"Figure 2: Testing Data (4 Turns)\nAcc: {test_acc*100:.1f}%", 
                       axes[1])

plt.tight_layout()
plt.show()