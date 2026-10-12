"""
Absentee Notification Mailer Module.

Sends email notifications to absent students for a specified recognition session
using configured SMTP parameters and letter template.
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime
import config
import database

LETTER_TEMPLATE_PATH = os.path.join(config.TEMPLATES_DIR, "letter_1.txt")

def send_absentee_emails(session_id=None):
    """
    Identifies absent students for a given session from the SQLite database
    and dispatches notification emails via SMTP.
    """
    if not config.MAIL_USERNAME or not config.MAIL_PASSWORD:
        print("SMTP mail credentials not configured in environment variables. Email dispatch skipped.")
        return False, "SMTP credentials not configured."

    # If session_id not passed, find the most recent session
    if not session_id:
        conn = database.get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM sessions ORDER BY id DESC LIMIT 1;")
        res = cursor.fetchone()
        conn.close()
        if not res:
            print("No sessions found in database.")
            return False, "No session available."
        session_id = res["id"]

    report = database.get_session_attendance_report(session_id)
    absentees = [row for row in report if row["status"] == "Absent"]

    if not absentees:
        print("No absent students for session.")
        return True, "No absentees to notify."

    # Read letter template
    template_content = "Dear NAME,\n\nYou were marked absent for class on DATE at TIME."
    if os.path.exists(LETTER_TEMPLATE_PATH):
        with open(LETTER_TEMPLATE_PATH, 'r') as f:
            template_content = f.read()

    now = datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    sent_count = 0
    try:
        server = smtplib.SMTP(config.MAIL_SERVER, config.MAIL_PORT)
        if config.MAIL_USE_TLS:
            server.starttls()
        server.login(config.MAIL_USERNAME, config.MAIL_PASSWORD)

        for student in absentees:
            name = student["name"]
            recipient_email = student["email"]

            body = template_content.replace("NAME", name).replace("DATE", today_date).replace("TIME", current_time)

            msg = MIMEMultipart()
            msg['From'] = config.MAIL_DEFAULT_SENDER
            msg['To'] = recipient_email
            msg['Subject'] = f"[{config.INSTITUTION_NAME}] Attendance Notification"
            msg.attach(MIMEText(body, 'plain'))

            try:
                server.sendmail(config.MAIL_DEFAULT_SENDER, recipient_email, msg.as_string())
                sent_count += 1
                print(f"Absentee notification sent to {name} ({recipient_email})")
            except Exception as ex:
                print(f"Failed sending email to {recipient_email}: {ex}")

        server.quit()
        return True, f"Successfully sent notifications to {sent_count} absent student(s)."

    except Exception as e:
        print(f"SMTP Connection Error: {e}")
        return False, f"SMTP Error: {str(e)}"

if __name__ == "__main__":
    success, msg = send_absentee_emails()
    print(f"Email Dispatch Result: {msg}")