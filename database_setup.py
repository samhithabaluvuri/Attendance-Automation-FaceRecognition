"""
Database setup script for the Biometric Attendance System.
Initializes SQLite database schema for admins, teachers, batches, students,
timetable, sessions, attendance, and biometric recognition logs.

Usage:
  python database_setup.py
  python database_setup.py --create-admin "System Admin" "admin@institution.edu"
"""

import sys
import database

def main():
    admin_name = None
    admin_email = None

    if "--create-admin" in sys.argv:
        idx = sys.argv.index("--create-admin")
        if len(sys.argv) > idx + 2:
            admin_name = sys.argv[idx + 1]
            admin_email = sys.argv[idx + 2]
        else:
            print("Error: --create-admin requires two arguments: <name> <email>")
            print("Example: python database_setup.py --create-admin 'System Admin' 'admin@univ.edu'")
            sys.exit(1)

    database.init_db()
    print("Database schema initialized successfully!")

    if admin_name and admin_email:
        existing = database.get_admin_by_email(admin_email)
        if existing:
            print(f"Admin account with email '{admin_email}' already exists.")
        else:
            admin_id = database.create_admin(admin_name, admin_email)
            print(f"Successfully created Admin account ID {admin_id}: {admin_name} <{admin_email}>")

if __name__ == "__main__":
    main()
