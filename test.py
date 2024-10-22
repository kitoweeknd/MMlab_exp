import matplotlib.pyplot as plt
import numpy as np

# 定义数据矩阵
C = np.array([[0, 2, 4, 6],
              [8, 10, 12, 14],
              [16, 18, 20, 22]])

# 显示图像
plt.imshow(C)

# 添加颜色条
plt.colorbar()

plt.show()
 