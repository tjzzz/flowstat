#encoding=utf8
"""
通用邮件通知（占位，需自行接入 SMTP/邮件服务）。

默认仅打印，避免误发或泄露。需要真实发送时：
1) 设置环境变量 SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS / SMTP_TO
2) 在下方实现 SMTP 发送逻辑（切勿在代码里硬编码密码）
"""
import os
import logging


def send_mail(message, subject='flowstat notification', attach_file_list=None, to_user=None):
    smtp_host = os.environ.get('SMTP_HOST')
    if not smtp_host:
        logging.info('[send_mail] 未配置 SMTP_HOST，仅打印不发送。')
        print('SUBJECT:', subject)
        print('TO:', to_user)
        print('MSG:', message)
        return

    # 真实发送实现（用户按需补全）：
    # import smtplib
    # from email.mime.text import MIMEText
    # ...
    raise NotImplementedError('请在此实现 SMTP 发送逻辑，切勿硬编码密码。')
