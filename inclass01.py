import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal  # ต้องเพิ่ม scipy เพื่อคำนวณความน่าจะเป็นแบบ Gaussian

# --- พารามิเตอร์เดิม ---
mean1 = np.array([-3, 5])   # mean ของคลาส a
mean2 = np.array([3, 5])    # mean ของคลาส b

cov1 = np.array([[1, 0], [0, 1]])  # covariance matrix ของคลาส a (ทรงกลม)
cov2 = np.array([[1, 0], [0, 1]])  # covariance matrix ของคลาส b (ทรงกลม)

pts1 = np.random.multivariate_normal(mean1, cov1, size=300)  # จำนวนจุดสุ่มข้อมูลคลาส a
pts2 = np.random.multivariate_normal(mean2, cov2, size=300)  # จำนวนจุดสุ่มข้อมูลคลาส b

# 1. กำหนดพื้นที่กริดสำหรับคำนวณ (ใช้สำหรับวาด contour และหาค่า posterior)
x_min, x_max = -10, 10
y_min, y_max = -10, 10
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]   # แปลงเป็น list ของจุด (x, y)

# 2. คำนวณ Likelihoods
# L1 = P(x | Class 'a'), L2 = P(x | Class 'b')
L1 = multivariate_normal.pdf(grid_points, mean=mean1, cov=cov1)  # Likelihood ของคลาส a
L2 = multivariate_normal.pdf(grid_points, mean=mean2, cov=cov2)  # Likelihood ของคลาส b

# 3. คำนวณ Posterior Probability P(Class 'a' | x)
# แบ่งพื้นที่ว่าเป็นคลาส a หรือ b โดยใช้ Bayes (แต่ priors เท่ากัน → LDA)
# P(a|x) = L1 / (L1 + L2)
posterior_a = L1 / (L1 + L2)
posterior_a = posterior_a.reshape(xx.shape)   # reshape กลับไปเป็นตารางเหมือน xx, yy

# --- โค้ด Plot กราฟ ---
plt.figure(figsize=(8, 8))
plt.scatter(pts1[:, 0], pts1[:, 1], marker='.', s=50, alpha=0.5,
            color='red', label='a')   # วาดจุดคลาส a
plt.scatter(pts2[:, 0], pts2[:, 1], marker='.', s=50, alpha=0.5,
            color='blue', label='b')  # วาดจุดคลาส b

# 4. พล็อตเส้นแบ่ง (Decision Boundary)
# เส้นแบ่งคือจุดที่ Posterior Probability = 0.5
# หมายถึงความน่าจะเป็นที่จะเป็น a หรือ b เท่ากันพอดี
plt.contour(xx, yy, posterior_a,
            levels=[0.5], colors='Black', linestyles='-', linewidths=2)

plt.axis('equal')        # ให้ scale ของแกน X,Y เท่ากัน
plt.xlabel('X')
plt.ylabel('Y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')  # บังคับให้กราฟไม่บิดเบี้ยว
plt.legend()
plt.grid()
plt.title('Data with Linear Decision Boundary (LDA)')
plt.show()
