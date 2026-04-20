# import torch
#
# # ========== 1. 准备数据 ==========
# x = torch.ones(2, 5)  # 2 个样本，5 个特征
# y = torch.zeros(2, 3)  # 目标值
#
# # ========== 2. 初始化参数 ==========
# torch.manual_seed(42)  # 固定随机种子，方便复现
# w = torch.randn(5, 3, requires_grad=True)
# b = torch.randn(3, requires_grad=True)
#
# # ========== 3. 训练配置 ==========
# learning_rate = 0.01
# epochs = 100
#
# print("开始训练...")
# print("=" * 60)
#
# # ========== 4. 训练循环 ==========
# for epoch in range(epochs):
#     # --- 前向传播 ---
#     z = torch.matmul(x, w) + b
#     loss = torch.nn.MSELoss()(z, y)
#
#     # --- 反向传播 ---
#     loss.backward()
#
#     # --- 参数更新 ---
#     with torch.no_grad():
#         w -= learning_rate * w.grad
#         b -= learning_rate * b.grad
#
#         # --- 梯度清零 ---
#         w.grad.zero_()
#         b.grad.zero_()
#
#     # --- 打印进度 ---
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch {epoch + 1:3d} | Loss: {loss.item():.6f} | '
#               f'w.grad norm: {w.grad.norm().item():.6f}')
#
# print("=" * 60)
# print("训练完成！")
#
# # ========== 5. 验证结果 ==========
# with torch.no_grad():
#     final_z = torch.matmul(x, w) + b
#     print(f'\n初始预测 (随机 w,b):\n{x @ torch.randn(5, 3) + torch.randn(3)}')
#     print(f'\n最终预测 (训练后 w,b):\n{final_z}')
#     print(f'\n目标值:\n{y}')
#     print(f'\n最终 loss: {torch.nn.MSELoss()(final_z, y).item():.8f}')


import torch
from torch.utils.data import DataLoader, TensorDataset

# ========== 准备大量数据 ==========
# 假设有 1000 个样本
all_x = torch.randn(1000, 5)  # 1000 个样本，每个 5 个特征
all_y = torch.randn(1000, 3)  # 1000 个目标值

# 创建数据集和数据加载器
dataset = TensorDataset(all_x, all_y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# batch_size=32: 每次随机取 32 个样本
# shuffle=True: 每轮打乱顺序

# ========== 训练 ==========
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

for epoch in range(100):  # 100 个 epoch
    for batch_x, batch_y in dataloader:  # ← 每轮取不同的 batch
        # 前向传播
        z = torch.matmul(batch_x, w) + b
        loss = torch.nn.MSELoss()(z, batch_y)

        # 反向传播
        loss.backward()

        # 参数更新
        with torch.no_grad():
            w -= 0.01 * w.grad
            b -= 0.01 * b.grad
            w.grad.zero_()
            b.grad.zero_()

    print(f'Epoch {epoch + 1}, Loss: {loss.item():.6f}')
