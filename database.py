import sqlite3
import os
import uuid
from datetime import datetime
import face_recognition
import config

SESSION_PERIODS = [
    {"slot": "09:00 AM - 10:00 AM", "name": "Period 1 (09:00 AM - 10:00 AM)"},
    {"slot": "10:00 AM - 11:00 AM", "name": "Period 2 (10:00 AM - 11:00 AM)"},
    {"slot": "11:00 AM - 12:00 PM", "name": "Period 3 (11:00 AM - 12:00 PM)"},
    {"slot": "12:00 PM - 01:00 PM", "name": "Period 4 (12:00 PM - 01:00 PM)"},
    {"slot": "01:00 PM - 02:00 PM", "name": "Period 5 (01:00 PM - 02:00 PM)"},
    {"slot": "02:00 PM - 03:00 PM", "name": "Period 6 (02:00 PM - 03:00 PM)"},
    {"slot": "03:00 PM - 04:00 PM", "name": "Period 7 (03:00 PM - 04:00 PM)"},
    {"slot": "04:00 PM - 05:00 PM", "name": "Period 8 (04:00 PM - 05:00 PM)"},
    {"slot": "24/7 Off-Hours Testing Mode", "name": "⚡ 24/7 Off-Hours Testing Mode (Any Time)"},
]

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def get_current_period_slot(dt=None):
    if dt is None:
        dt = datetime.now()
    hour = dt.hour
    if 9 <= hour < 10:
        return "09:00 AM - 10:00 AM"
    elif 10 <= hour < 11:
        return "10:00 AM - 11:00 AM"
    elif 11 <= hour < 12:
        return "11:00 AM - 12:00 PM"
    elif 12 <= hour < 13:
        return "12:00 PM - 01:00 PM"
    elif 13 <= hour < 14:
        return "01:00 PM - 02:00 PM"
    elif 14 <= hour < 15:
        return "02:00 PM - 03:00 PM"
    elif 15 <= hour < 16:
        return "03:00 PM - 04:00 PM"
    elif 16 <= hour < 17:
        return "04:00 PM - 05:00 PM"
    else:
        return "24/7 Off-Hours Testing Mode"

def get_db_connection():
    """Establishes and returns a database connection with Row factory."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema for multi-role attendance tracking."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")

    # 1. Admins Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            created_at TEXT
        );
    """)

    # 2. Teachers Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL DEFAULT 'ACTIVE'
        );
    """)

    # Schema migration check for teachers table
    cursor.execute("PRAGMA table_info(teachers);")
    t_cols = [col['name'] for col in cursor.fetchall()]
    if 'name' not in t_cols:
        cursor.execute("ALTER TABLE teachers ADD COLUMN name TEXT;")
    if 'status' not in t_cols:
        cursor.execute("ALTER TABLE teachers ADD COLUMN status TEXT NOT NULL DEFAULT 'ACTIVE';")
    if 'teacher_code' not in t_cols:
        cursor.execute("ALTER TABLE teachers ADD COLUMN teacher_code TEXT;")
    conn.commit()

    # 3. Batches Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_code TEXT UNIQUE NOT NULL,
            batch_name TEXT NOT NULL,
            teacher_id INTEGER,
            FOREIGN KEY (teacher_id) REFERENCES teachers(id) ON DELETE SET NULL
        );
    """)

    # 4. Students Roster Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            batch_id INTEGER,
            photo_filename TEXT,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            FOREIGN KEY (batch_id) REFERENCES batches(id) ON DELETE SET NULL
        );
    """)

    cursor.execute("PRAGMA table_info(students);")
    s_cols = [col['name'] for col in cursor.fetchall()]
    if 'status' not in s_cols:
        cursor.execute("ALTER TABLE students ADD COLUMN status TEXT NOT NULL DEFAULT 'ACTIVE';")
    conn.commit()

    # 4b. Multi-Angle Enrollment Photos Table
    # Supports pose-robust "best-of-N" template matching: multiple reference
    # photos per student (e.g. frontal, left-profile, right-profile) so the
    # matcher can compare a live frame against whichever enrolled angle is
    # closest, rather than a single frontal-only template.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS student_face_photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            photo_filename TEXT NOT NULL,
            pose_label TEXT NOT NULL DEFAULT 'frontal',
            created_at TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        );
    """)
    conn.commit()

    # 5. Timetable Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS timetable (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL,
            day_of_week TEXT NOT NULL,
            period_slot TEXT NOT NULL,
            subject TEXT NOT NULL,
            UNIQUE(batch_id, day_of_week, period_slot),
            FOREIGN KEY (batch_id) REFERENCES batches(id) ON DELETE CASCADE
        );
    """)

    # 6. Attendance Sessions Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_uuid TEXT UNIQUE NOT NULL,
            batch_id INTEGER NOT NULL,
            teacher_id INTEGER NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            period_slot TEXT,
            FOREIGN KEY (batch_id) REFERENCES batches(id) ON DELETE CASCADE,
            FOREIGN KEY (teacher_id) REFERENCES teachers(id) ON DELETE CASCADE
        );
    """)

    cursor.execute("PRAGMA table_info(sessions);")
    sess_cols = [col['name'] for col in cursor.fetchall()]
    if 'period_slot' not in sess_cols:
        cursor.execute("ALTER TABLE sessions ADD COLUMN period_slot TEXT;")
        conn.commit()
    if 'test_condition' not in sess_cols:
        # Tags a recognition session with the degradation condition under
        # which it was recorded (normal / low_light / pose_variation /
        # occlusion / low_resolution / motion_blur), set via
        # database.set_session_condition(). Enables evaluate.py to report
        # recognition rate broken down by degradation type.
        cursor.execute("ALTER TABLE sessions ADD COLUMN test_condition TEXT DEFAULT 'normal';")
        conn.commit()

    # 7. Real-Time Attendance Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Present',
            period_slot TEXT,
            UNIQUE(session_id, student_id),
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        );
    """)

    cursor.execute("PRAGMA table_info(attendance);")
    att_cols = [col['name'] for col in cursor.fetchall()]
    if 'period_slot' not in att_cols:
        cursor.execute("ALTER TABLE attendance ADD COLUMN period_slot TEXT;")
        conn.commit()

    # 8. Recognition Logs Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recognition_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TEXT NOT NULL,
            computed_distance REAL,
            matched_student_id INTEGER,
            matched_name TEXT,
            decision TEXT NOT NULL,
            liveness_passed INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY (matched_student_id) REFERENCES students(id) ON DELETE SET NULL
        );
    """)

    # Quality-aware assessment columns (added for compound real-world
    # degradation evaluation: low illumination, pose variation, occlusion,
    # low resolution, motion blur). See quality.py for the scoring formulas.
    cursor.execute("PRAGMA table_info(recognition_logs);")
    rl_cols = [col['name'] for col in cursor.fetchall()]
    quality_columns = [
        ("illumination_score", "REAL"),
        ("blur_score", "REAL"),
        ("pose_score", "REAL"),
        ("yaw_deg", "REAL"),
        ("resolution_score", "REAL"),
        ("occlusion_score", "REAL"),
        ("composite_quality", "REAL"),
        ("dominant_degradation", "TEXT"),
    ]
    for col_name, col_type in quality_columns:
        if col_name not in rl_cols:
            cursor.execute(f"ALTER TABLE recognition_logs ADD COLUMN {col_name} {col_type};")
    conn.commit()
    sync_known_faces(conn)
    conn.close()

def sync_known_faces(conn):
    """
    Ensures baseline teacher & batch records exist if database is completely empty.
    Student records are created exclusively via Admin enrollment.
    """
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as count FROM teachers;")
    if cursor.fetchone()["count"] == 0:
        cursor.execute("INSERT INTO teachers (name, email) VALUES (?, ?);", 
                       ("Faculty Administrator", f"faculty@{config.INSTITUTION_DOMAIN}"))
        conn.commit()

    cursor.execute("SELECT id FROM teachers LIMIT 1;")
    default_teacher = cursor.fetchone()
    default_teacher_id = default_teacher["id"] if default_teacher else 1

    cursor.execute("SELECT COUNT(*) as count FROM batches;")
    if cursor.fetchone()["count"] == 0:
        cursor.execute("INSERT INTO batches (batch_code, batch_name, teacher_id) VALUES (?, ?, ?);",
                       ("CS101", "Computer Science 101", default_teacher_id))
        conn.commit()

# --- Auth Helpers ---
def get_admin_by_email(email):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admins WHERE email = ? AND status = 'ACTIVE';", (email.lower().strip(),))
    admin = cursor.fetchone()
    conn.close()
    return admin

def get_teacher_by_email(email):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM teachers WHERE email = ? AND status = 'ACTIVE';", (email.lower().strip(),))
    teacher = cursor.fetchone()
    conn.close()
    return teacher

def get_student_by_email(email):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, b.batch_code, b.batch_name 
        FROM students s 
        LEFT JOIN batches b ON s.batch_id = b.id 
        WHERE s.email = ? AND s.status = 'ACTIVE';
    """, (email.lower().strip(),))
    student = cursor.fetchone()
    conn.close()
    return student

# --- Admin Management CRUD ---
def create_admin(name, email):
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO admins (name, email, status, created_at) VALUES (?, ?, 'ACTIVE', ?);",
                   (name.strip(), email.lower().strip(), now_str))
    conn.commit()
    admin_id = cursor.lastrowid
    conn.close()
    return admin_id

def get_all_admins():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admins ORDER BY id DESC;")
    admins = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return admins

def update_admin(admin_id, name, email, status='ACTIVE'):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE admins SET name = ?, email = ?, status = ? WHERE id = ?;",
                   (name.strip(), email.lower().strip(), status, admin_id))
    conn.commit()
    conn.close()
    return True

def delete_admin(admin_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM admins WHERE id = ?;", (admin_id,))
    conn.commit()
    conn.close()
    return True

# --- Sequential ID Generators ---
def get_next_student_code():
    """Generates next sequential Student Roll Code (e.g. STU-1001, STU-1002...)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(id) FROM students;")
    row = cursor.fetchone()
    conn.close()
    max_id = (row[0] or 0) if row else 0
    return f"STU-{1000 + max_id + 1}"

def get_next_teacher_code():
    """Generates next sequential Faculty/Teacher Code (e.g. FAC-1001, FAC-1002...)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(id) FROM teachers;")
    row = cursor.fetchone()
    conn.close()
    max_id = (row[0] or 0) if row else 0
    return f"FAC-{1000 + max_id + 1}"

# --- Teacher Management CRUD ---
def create_teacher(name, email, teacher_code=None):
    if not teacher_code or not str(teacher_code).strip():
        teacher_code = get_next_teacher_code()
    else:
        teacher_code = str(teacher_code).strip().upper()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO teachers (teacher_code, name, email, status) VALUES (?, ?, ?, 'ACTIVE');",
                   (teacher_code, name.strip(), email.lower().strip()))
    conn.commit()
    teacher_id = cursor.lastrowid
    conn.close()
    return teacher_id

def get_all_teachers():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM teachers ORDER BY id ASC;")
    teachers = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return teachers

def update_teacher(teacher_id, name, email, status='ACTIVE', teacher_code=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    name_str = (name or "").strip()
    email_str = (email or "").lower().strip()
    code_str = (teacher_code or "").strip().upper() if teacher_code else None
    try:
        if code_str:
            cursor.execute("UPDATE teachers SET teacher_code = ?, name = ?, email = ?, status = ? WHERE id = ?;",
                           (code_str, name_str, email_str, status, teacher_id))
        else:
            cursor.execute("UPDATE teachers SET name = ?, email = ?, status = ? WHERE id = ?;",
                           (name_str, email_str, status, teacher_id))
        conn.commit()
        return True, "Teacher updated."
    except sqlite3.IntegrityError:
        return False, "This email address or teacher code is already registered to another faculty member."
    finally:
        conn.close()

def delete_teacher(teacher_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM teachers WHERE id = ?;", (teacher_id,))
    conn.commit()
    conn.close()
    return True

# --- Student Management CRUD ---
def enroll_student(name, email, student_code=None, batch_id=None, photo_filename=None):
    if not student_code or not str(student_code).strip():
        student_code = get_next_student_code()
    else:
        student_code = str(student_code).strip().upper()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO students (name, email, student_code, batch_id, photo_filename, status)
        VALUES (?, ?, ?, ?, ?, 'ACTIVE');
    """, (name.strip(), email.lower().strip(), student_code, batch_id, photo_filename))
    conn.commit()
    student_id = cursor.lastrowid

    # The primary enrollment photo also becomes the first row in
    # student_face_photos (pose_label='frontal') so it participates in
    # multi-angle best-of-N matching alongside any additional angles added
    # later via add_student_face_photo().
    if photo_filename:
        cursor.execute("""
            INSERT INTO student_face_photos (student_id, photo_filename, pose_label, created_at)
            VALUES (?, ?, 'frontal', ?);
        """, (student_id, photo_filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()

    conn.close()
    return student_id

def add_student_face_photo(student_id, photo_filename, pose_label='additional'):
    """
    Adds an additional enrollment photo at a given pose angle for an existing
    student (e.g. pose_label='left_profile', 'right_profile', 'up', 'down').
    Used to build a pose-robust multi-template gallery -- addresses the pose
    variation research gap without needing to retrain the embedding model.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO student_face_photos (student_id, photo_filename, pose_label, created_at)
        VALUES (?, ?, ?, ?);
    """, (student_id, photo_filename, pose_label, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    photo_id = cursor.lastrowid
    conn.close()
    return photo_id

def get_student_face_photos(student_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM student_face_photos WHERE student_id = ? ORDER BY id ASC;
    """, (student_id,))
    photos = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return photos

def delete_student_face_photo(photo_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM student_face_photos WHERE id = ?;", (photo_id,))
    conn.commit()
    conn.close()
    return True

def get_all_students():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, b.batch_code, b.batch_name
        FROM students s
        LEFT JOIN batches b ON s.batch_id = b.id
        ORDER BY s.name ASC;
    """)
    students = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return students

def update_student(student_id, name, email, student_code=None, batch_id=None, photo_filename=None, status='ACTIVE'):
    conn = get_db_connection()
    cursor = conn.cursor()
    name_str = (name or "").strip()
    email_str = (email or "").lower().strip()
    code_str = (student_code or "").strip().upper() if student_code else None

    try:
        batch_id_int = int(batch_id) if batch_id and str(batch_id).isdigit() else None
    except (ValueError, TypeError):
        batch_id_int = None

    try:
        if photo_filename:
            cursor.execute("""
                UPDATE students SET name = ?, email = ?, student_code = COALESCE(?, student_code), batch_id = ?, photo_filename = ?, status = ?
                WHERE id = ?;
            """, (name_str, email_str, code_str, batch_id_int, photo_filename, status, student_id))
        else:
            cursor.execute("""
                UPDATE students SET name = ?, email = ?, student_code = COALESCE(?, student_code), batch_id = ?, status = ?
                WHERE id = ?;
            """, (name_str, email_str, code_str, batch_id_int, status, student_id))
        conn.commit()
        return True, "Student updated."
    except sqlite3.IntegrityError:
        return False, "This email address or roll code is already registered to another student."
    finally:
        conn.close()

def delete_student(student_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM students WHERE id = ?;", (student_id,))
    conn.commit()
    conn.close()
    return True

# --- Batch & Encodings Helpers ---
def create_batch(batch_code, batch_name, teacher_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO batches (batch_code, batch_name, teacher_id) VALUES (?, ?, ?);",
                   (batch_code.strip().upper(), batch_name.strip(), teacher_id))
    conn.commit()
    batch_id = cursor.lastrowid
    conn.close()
    return batch_id

def update_batch(batch_id, batch_code, batch_name, teacher_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE batches SET batch_code = ?, batch_name = ?, teacher_id = ? WHERE id = ?;",
                   (batch_code.strip().upper(), batch_name.strip(), teacher_id, batch_id))
    conn.commit()
    conn.close()
    return True

def delete_batch(batch_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM batches WHERE id = ?;", (batch_id,))
    conn.commit()
    conn.close()
    return True

def get_all_batches():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT b.*, t.name as teacher_name, t.email as teacher_email,
               (SELECT COUNT(*) FROM students s WHERE s.batch_id = b.id) as student_count
        FROM batches b
        LEFT JOIN teachers t ON b.teacher_id = t.id;
    """)
    batches = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return batches

def get_students_for_batch(batch_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE batch_id = ? AND status = 'ACTIVE';", (batch_id,))
    students = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return students

def load_known_face_encodings(batch_id=None):
    """
    Loads facial encodings for students enrolled in a specific batch (or all
    students if batch_id is None).

    Multi-angle (pose-robust) matching: for each student, ALL photos in
    student_face_photos are encoded (not just the primary photo_filename),
    so the recognition loop can compare a live frame against the closest of
    several enrolled poses ("best-of-N" template matching). If a student has
    no rows in student_face_photos (e.g. legacy data), falls back to their
    single primary photo_filename for backward compatibility.

    Returns dict: student_id -> {
        "id": sid, "name", "email", "student_code",
        "encodings": [np.ndarray, ...]   # one or more, in enrollment order
    }
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    if batch_id:
        cursor.execute("SELECT * FROM students WHERE batch_id = ? AND status = 'ACTIVE';", (batch_id,))
    else:
        cursor.execute("SELECT * FROM students WHERE status = 'ACTIVE';")
    students = cursor.fetchall()

    encodings_db = {}
    if not os.path.exists(config.KNOWN_FACES_DIR):
        conn.close()
        return encodings_db

    for student in students:
        sid = student["id"]
        cursor.execute("SELECT photo_filename FROM student_face_photos WHERE student_id = ?;", (sid,))
        photo_rows = [r["photo_filename"] for r in cursor.fetchall()]

        if not photo_rows and student["photo_filename"]:
            photo_rows = [student["photo_filename"]]  # legacy fallback

        encs = []
        for filename in photo_rows:
            img_path = os.path.join(config.KNOWN_FACES_DIR, filename)
            if not os.path.exists(img_path):
                continue
            try:
                image = face_recognition.load_image_file(img_path)
                found = face_recognition.face_encodings(image)
                if found:
                    encs.append(found[0])
            except Exception as e:
                print(f"Error loading face encoding for {filename}: {e}")

        if encs:
            encodings_db[sid] = {
                "id": sid,
                "name": student["name"],
                "email": student["email"],
                "student_code": student["student_code"],
                "encodings": encs,  # list, used for best-of-N matching in server.py
            }

    conn.close()
    return encodings_db

# --- Timetable Management & Validation ---
def add_timetable_entry(batch_id, day_of_week, period_slot, subject):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO timetable (batch_id, day_of_week, period_slot, subject)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(batch_id, day_of_week, period_slot) DO UPDATE SET subject = ?;
    """, (batch_id, day_of_week.strip().title(), period_slot.strip(), subject.strip(), subject.strip()))
    conn.commit()
    entry_id = cursor.lastrowid
    conn.close()
    return entry_id

def delete_timetable_entry(entry_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM timetable WHERE id = ?;", (entry_id,))
    conn.commit()
    conn.close()
    return True

def get_all_timetable_entries():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t.*, b.batch_code, b.batch_name
        FROM timetable t
        JOIN batches b ON t.batch_id = b.id
        ORDER BY b.batch_code, t.day_of_week, t.period_slot;
    """)
    entries = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return entries

def get_timetable_for_batch(batch_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM timetable WHERE batch_id = ? ORDER BY day_of_week, period_slot;
    """, (batch_id,))
    entries = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return entries

def is_class_scheduled(batch_id, day_of_week, period_slot):
    """
    Checks if a class is scheduled for the given batch, day of week, and period slot.
    Returns matching timetable dict or a 24/7 testing mode fallback.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM timetable 
        WHERE batch_id = ? AND LOWER(day_of_week) = ? AND period_slot = ?;
    """, (batch_id, day_of_week.lower().strip(), period_slot.strip()))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)

    # 24/7 Testing Mode Fallback: allow testing at 12 AM / off-hours / weekends seamlessly!
    return {
        "id": 0,
        "batch_id": batch_id,
        "day_of_week": day_of_week,
        "period_slot": period_slot,
        "subject": "Off-Hours / 24-7 Testing Mode"
    }

# --- Session Management ---
def create_session(batch_id, teacher_id, period_slot=None):
    conn = get_db_connection()
    cursor = conn.cursor()

    if not period_slot:
        period_slot = get_current_period_slot()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("UPDATE sessions SET status = 'COMPLETED', end_time = ? WHERE batch_id = ? AND status = 'ACTIVE';", 
                   (now_str, batch_id))

    session_uuid = str(uuid.uuid4())
    cursor.execute("""
        INSERT INTO sessions (session_uuid, batch_id, teacher_id, start_time, status, period_slot)
        VALUES (?, ?, ?, ?, 'ACTIVE', ?);
    """, (session_uuid, batch_id, teacher_id, now_str, period_slot))

    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return session_id, session_uuid

def end_session(session_id):
    if not session_id:
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("UPDATE sessions SET status = 'COMPLETED', end_time = ? WHERE id = ?;", (now_str, session_id))
    conn.commit()
    conn.close()

def get_session_by_id(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, b.batch_code, b.batch_name, t.name as teacher_name
        FROM sessions s
        JOIN batches b ON s.batch_id = b.id
        JOIN teachers t ON s.teacher_id = t.id
        WHERE s.id = ?;
    """, (session_id,))
    session_row = cursor.fetchone()
    conn.close()
    return dict(session_row) if session_row else None

def get_past_sessions_by_teacher(teacher_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, b.batch_code, b.batch_name,
               (SELECT COUNT(*) FROM attendance a WHERE a.session_id = s.id) as present_count,
               (SELECT COUNT(*) FROM students st WHERE st.batch_id = s.batch_id) as total_students
        FROM sessions s
        JOIN batches b ON s.batch_id = b.id
        WHERE s.teacher_id = ?
        ORDER BY s.start_time DESC;
    """, (teacher_id,))
    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return sessions

def get_all_sessions():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.*, b.batch_code, b.batch_name, t.name as teacher_name,
               (SELECT COUNT(*) FROM attendance a WHERE a.session_id = s.id) as present_count,
               (SELECT COUNT(*) FROM students st WHERE st.batch_id = s.batch_id) as total_students
        FROM sessions s
        JOIN batches b ON s.batch_id = b.id
        JOIN teachers t ON s.teacher_id = t.id
        ORDER BY s.start_time DESC;
    """)
    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return sessions

# --- Attendance Logging & Querying ---
def mark_attendance_event(session_id, student_id, period_slot=None):
    if not session_id or not student_id:
        return False

    if not period_slot:
        session_info = get_session_by_id(session_id)
        period_slot = session_info.get("period_slot") if session_info else get_current_period_slot()

    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute("""
            INSERT INTO attendance (session_id, student_id, timestamp, status, period_slot)
            VALUES (?, ?, ?, 'Present', ?);
        """, (session_id, student_id, now_str, period_slot))
        conn.commit()
        marked = True
    except sqlite3.IntegrityError:
        marked = False
    finally:
        conn.close()

    return marked

def update_attendance_status(session_id, student_id, new_status, period_slot=None):
    if not session_id or not student_id:
        return False

    if not period_slot:
        session_info = get_session_by_id(session_id)
        period_slot = session_info.get("period_slot") if session_info else get_current_period_slot()

    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if new_status.lower() == 'present':
        cursor.execute("""
            INSERT INTO attendance (session_id, student_id, timestamp, status, period_slot)
            VALUES (?, ?, ?, 'Present', ?)
            ON CONFLICT(session_id, student_id) DO UPDATE SET status = 'Present', timestamp = ?, period_slot = ?;
        """, (session_id, student_id, now_str, period_slot, now_str, period_slot))
    else:
        cursor.execute("DELETE FROM attendance WHERE session_id = ? AND student_id = ?;", (session_id, student_id))

    conn.commit()
    conn.close()
    return True

def get_session_attendance_report(session_id):
    session_info = get_session_by_id(session_id)
    if not session_info:
        return []

    batch_id = session_info["batch_id"]
    sess_period = session_info.get("period_slot") or get_current_period_slot()

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT st.id, st.id as student_id, st.student_code, st.name, st.email, st.photo_filename,
               a.timestamp,
               COALESCE(a.period_slot, ?) as period_slot,
               CASE WHEN a.id IS NOT NULL THEN 'Present' ELSE 'Absent' END as status
        FROM students st
        LEFT JOIN attendance a ON st.id = a.student_id AND a.session_id = ?
        WHERE st.batch_id = ?
        ORDER BY st.name ASC;
    """, (sess_period, session_id, batch_id))

    roster = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return roster

def log_recognition_attempt(session_id, distance, matched_student_id, matched_name, decision, liveness_passed, quality=None):
    """
    :param quality: optional dict from quality.assess_face_quality(), logged
        alongside the recognition decision so evaluate.py can report
        recognition rate broken down by degradation type and correlate
        quality sub-scores with accept/reject outcomes.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    q = quality or {}
    cursor.execute("""
        INSERT INTO recognition_logs (
            session_id, timestamp, computed_distance, matched_student_id, matched_name,
            decision, liveness_passed,
            illumination_score, blur_score, pose_score, yaw_deg, resolution_score,
            occlusion_score, composite_quality, dominant_degradation
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        session_id, now_str, float(distance) if distance is not None else None,
        matched_student_id, matched_name, decision, 1 if liveness_passed else 0,
        q.get("illumination_score"), q.get("blur_score"), q.get("pose_score"), q.get("yaw_deg"),
        q.get("resolution_score"), q.get("occlusion_score"), q.get("composite_quality"),
        q.get("dominant_degradation"),
    ))
    log_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return log_id

def set_session_condition(session_id, condition):
    """
    Tags a session with the degradation condition it was recorded under, e.g.:
    normal, low_illumination, pose_variation, occlusion, low_resolution, motion_blur.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE sessions SET test_condition = ? WHERE id = ?;", (condition, session_id))
    conn.commit()
    conn.close()
    return True

def delete_session(session_id):
    """Deletes a session and its associated attendance and recognition logs."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance WHERE session_id = ?;", (session_id,))
    cursor.execute("DELETE FROM recognition_logs WHERE session_id = ?;", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE id = ?;", (session_id,))
    conn.commit()
    conn.close()
    return True

def delete_recognition_log(log_id):
    """Deletes an individual recognition log entry."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM recognition_logs WHERE id = ?;", (log_id,))
    conn.commit()
    conn.close()
    return True

def get_recognition_logs_with_condition(limit=5000):
    """Joins recognition_logs with sessions.test_condition for per-condition evaluation."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT r.*, s.test_condition
        FROM recognition_logs r
        LEFT JOIN sessions s ON r.session_id = s.id
        ORDER BY r.id ASC LIMIT ?;
    """, (limit,))
    logs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return logs

def get_all_recognition_logs(limit=200):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT r.*, s.batch_id, b.batch_code
        FROM recognition_logs r
        LEFT JOIN sessions s ON r.session_id = s.id
        LEFT JOIN batches b ON s.batch_id = b.id
        ORDER BY r.id DESC LIMIT ?;
    """, (limit,))
    logs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return logs

# --- Student Dashboard Statistics ---
def get_student_dashboard_stats(email):
    """
    Calculates student attendance stats using real timetable definitions and conducted sessions.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM students WHERE email = ? AND status = 'ACTIVE';", (email.lower().strip(),))
    student = cursor.fetchone()
    if not student:
        conn.close()
        return None

    student_id = student["id"]

    cursor.execute("""
        SELECT b.id as batch_id, b.batch_code, b.batch_name
        FROM batches b
        WHERE b.id = ?;
    """, (student["batch_id"],))
    batches = cursor.fetchall()

    batch_stats = []
    for b in batches:
        b_id = b["batch_id"]

        # Sessions held for this batch
        cursor.execute("SELECT COUNT(*) as total FROM sessions WHERE batch_id = ?;", (b_id,))
        total_sessions = cursor.fetchone()["total"]

        # Sessions attended by student
        cursor.execute("""
            SELECT COUNT(*) as attended
            FROM attendance a
            JOIN sessions s ON a.session_id = s.id
            WHERE s.batch_id = ? AND a.student_id = ?;
        """, (b_id, student_id))
        attended_sessions = cursor.fetchone()["attended"]

        pct = (attended_sessions / total_sessions * 100.0) if total_sessions > 0 else 0.0

        # Detailed breakdown per period slot cross-referenced with timetable
        period_breakdown = []
        for p in SESSION_PERIODS:
            slot_str = p["slot"]

            # Count scheduled timetable classes for this slot
            cursor.execute("SELECT subject FROM timetable WHERE batch_id = ? AND period_slot = ?;", (b_id, slot_str))
            tt_row = cursor.fetchone()
            subject_name = tt_row["subject"] if tt_row else "Not Scheduled"

            cursor.execute("SELECT COUNT(*) as cnt FROM sessions WHERE batch_id = ? AND period_slot = ?;", (b_id, slot_str))
            p_total = cursor.fetchone()["cnt"]

            cursor.execute("""
                SELECT COUNT(*) as cnt FROM attendance a
                JOIN sessions s ON a.session_id = s.id
                WHERE s.batch_id = ? AND a.student_id = ? AND (s.period_slot = ? OR a.period_slot = ?);
            """, (b_id, student_id, slot_str, slot_str))
            p_attended = cursor.fetchone()["cnt"]

            period_breakdown.append({
                "period_slot": slot_str,
                "period_name": p["name"],
                "subject": subject_name,
                "total_classes": p_total,
                "attended_classes": p_attended,
                "status": "Present" if p_attended > 0 else ("Absent" if p_total > 0 else "No Session")
            })

        batch_stats.append({
            "batch_code": b["batch_code"],
            "batch_name": b["batch_name"],
            "total_sessions": total_sessions,
            "attended_sessions": attended_sessions,
            "attendance_percentage": round(pct, 2),
            "period_breakdown": period_breakdown
        })

    conn.close()

    return {
        "student": dict(student),
        "batches": batch_stats
    }
