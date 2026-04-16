import torch
print(f"Device: {torch.cuda.get_device_name(0)}")


try:
    # 模拟报错位置的行为：创建一个 BF16 张量并转成 F32
    x = torch.randn(1, 10, 2048, device="cuda", dtype=torch.bfloat16)
    # y = x.to(torch.float32) 
    # print("Basic tensor conversion: Success")
    # 总量同样是 20480 个元素，看一维行不行
    x = torch.randn(20480, device="cuda", dtype=torch.float16)
    # 申请一个比刚才报错的 3D 张量大得多的 2D 张量
    x = torch.randn(1, 8192, device="cuda", dtype=torch.float16)

    # 模拟 RMSNorm 内部操作
    var = x.pow(2).mean()
    print("Math operation: Success")
except Exception as e:
    print(f"Hardware Error Detected: {e}")


# 测试 1: 常规 FP32
try:
    x_f32 = torch.randn(1, 1024, device="cuda", dtype=torch.float32)
    y_f32 = x_f32.pow(2).mean()
    print("FP32 Test: Success")
except Exception as e:
    print(f"FP32 Test: Failed! {e}")

# 测试 2: 换成 FP16
try:
    x_f16 = torch.randn(1, 1024, device="cuda", dtype=torch.float16)
    y_f16 = x_f16.pow(2).mean()
    print("FP16 Test: Success")
except Exception as e:
    print(f"FP16 Test: Failed! {e}")