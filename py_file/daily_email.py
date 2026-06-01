import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.header import Header
import os
import time

# ========== 修改下面这些就行 ==========
smtp_server = "smtp.qq.com"
smtp_port = 465
sender_email = "1960622591@qq.com"
password = "fpcgikhwwbziccje"

# 收件人列表，会逐一发送
receiver_list = [
    "2254225117@qq.com",
    "1960622591@qq.com",
    "Milad.Mousavi@cgnee.com",
    "Zilin.Wang@cgnee.com",
    "wenyu.cao@cgnee.com",
    "Hangming.Lu@cgnee.com",
    "sunyuxi@szu.edu.cn"
]
# 自动获取当前日期
import datetime
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
# 构建主题
subject = f"SE2 Electricity Price Forecast - {current_date}"
body = """Dear CGNEE Trading Team,

We have prepared the latest SE2 electricity price forecast. The results are provided in the attached file.

This email is sent automatically on a daily basis, based on real-time market data and our forecasting model outputs.

Best regards,

Electricity Price Forecasting Team
Shenzhen University"""

# 附件路径列表，没有附件请留空：[]
# 读取 outputs/commits 下的所有文件
# 获取项目根目录
import pathlib
_BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
folder = str(_BASE_DIR / "outputs" / "commits")
attachment_files = [
    os.path.join(folder, f)
    for f in os.listdir(folder)
    if os.path.isfile(os.path.join(folder, f))
]
# =====================================

def send_to_one(receiver):
    """构建并发送给单个收件人的邮件"""
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver                     # 只显示当前收件人
    msg["Subject"] = Header(subject, "utf-8")

    # 正文
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # 添加附件（每个收件人都带相同附件）
    for file_path in attachment_files:
        if not os.path.isfile(file_path):
            print(f"⚠️ 文件不存在，跳过：{file_path}")
            continue
        with open(file_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        filename = os.path.basename(file_path)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={Header(filename, 'utf-8').encode()}"
        )
        msg.attach(part)

    return msg

try:
    # 一次连接，分别发送给每个人
    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(sender_email, password)

        for i, receiver in enumerate(receiver_list, 1):
            msg = send_to_one(receiver)
            server.sendmail(sender_email, receiver, msg.as_string())
            print(f"✅ ({i}/{len(receiver_list)}) 已发送至 {receiver}")
            # 可选：适当延时，避免发送过快被服务器限制
            # time.sleep(2)

    print("全部发送完成！")
except Exception as e:
    print(f"❌ 发送失败：{e}")