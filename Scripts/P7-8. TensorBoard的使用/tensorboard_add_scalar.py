from torch.utils.tensorboard import SummaryWriter # 需要先安装tensorboard, pip install tensorboard

# 创建SummaryWriter对象，指定日志保存路径
writer = SummaryWriter("Scripts/P7-8. TensorBoard的使用/logs/add_scalar")

for i in range(100):
    writer.add_scalar("y=2x", 2*i, i) # 第一个参数是标签，第二个参数是y值，第三个参数是x值（步数）
    writer.add_scalar("y=3x", 3*i, i)

writer.close()

# 在终端运行命令: tensorboard --logdir="Scripts/P7-8. TensorBoard的使用/logs/add_scalar"
# 要更换端口使用 --port参数，示例： tensorboard --logdir="Scripts/P7-8. TensorBoard的使用/logs/add_scalar" --port=6666
# 然后打开浏览器访问 http://localhost:6666/  查看结果，默认端口为6006