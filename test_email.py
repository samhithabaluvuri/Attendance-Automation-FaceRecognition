import smtplib
import config

email = config.MAIL_USERNAME
password = config.MAIL_PASSWORD

if not email or not password:
    print("Error: SMTP credentials (MAIL_USERNAME / MAIL_PASSWORD) are missing in environment variables / .env.")
else:
    try:
        server = smtplib.SMTP(config.MAIL_SERVER, config.MAIL_PORT)
        if config.MAIL_USE_TLS:
            server.starttls()
        server.login(email, password)
        print("Login worked! Credentials are good.")
        server.quit()
    except Exception as e:
        print(f"Error: {e}")