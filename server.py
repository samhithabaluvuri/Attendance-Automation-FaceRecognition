import os
import cv2
import numpy as np
import base64
import random
import time
import werkzeug.utils
from datetime import datetime
from functools import wraps
from flask import Flask, Response, jsonify, render_template, request, session, redirect, url_for, send_from_directory
from flask_mail import Mail, Message

import config
import database
import quality
from liveness import LivenessDetector
from main import send_absentee_emails

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = config.SECRET_KEY

# Mail configuration
app.config['MAIL_SERVER'] = config.MAIL_SERVER
app.config['MAIL_PORT'] = config.MAIL_PORT
app.config['MAIL_USE_TLS'] = config.MAIL_USE_TLS
app.config['MAIL_USERNAME'] = config.MAIL_USERNAME
app.config['MAIL_PASSWORD'] = config.MAIL_PASSWORD
app.config['MAIL_DEFAULT_SENDER'] = config.MAIL_DEFAULT_SENDER
mail = Mail(app)

# Database schema initialization
database.init_db()

# Temporary OTP storage: email -> {"otp": str, "role": str}
otp_storage = {}

# Global recognition state
active_session_id = None
active_batch_id = None
last_session_id = None
active_encodings = {}  # student_id -> details
liveness_detector = LivenessDetector()

cap = None
is_running = False
frame_count = 0
process_every_n_frames = 3
last_matches = []
last_match_weights = []  # quality-weighted companion to last_matches, same length
last_matched_name = "Unknown"
last_face_location = None
last_liveness_status = "N/A"
last_min_distance = 1.0
last_quality_status = "N/A"

# Context processor for templates
@app.context_processor
def inject_institution_config():
    return {
        'institution_name': config.INSTITUTION_NAME,
        'institution_domain': config.INSTITUTION_DOMAIN,
        'enable_liveness': config.ENABLE_LIVENESS_CHECK,
        'match_threshold': config.MATCH_THRESHOLD,
        'session_periods': database.SESSION_PERIODS,
        'current_period_slot': database.get_current_period_slot()
    }

# Auth Decorators
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrap

def faculty_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in'] or session.get('role') != 'FACULTY':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrap

def student_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in'] or session.get('role') != 'STUDENT':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrap

def admin_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in'] or session.get('role') != 'ADMIN':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrap

# Frame Generator for Live Recognition & Anti-Spoofing
def generate_frames():
    global cap, is_running, frame_count, last_matches, last_match_weights, last_matched_name, \
        last_face_location, last_liveness_status, last_min_distance, last_quality_status

    if cap is None or not cap.isOpened():
        try:
            cap = cv2.VideoCapture(0)
        except Exception as ex:
            print(f"Error opening video capture: {ex}")

    cached_locations = []
    cached_encodings = []

    while True:
        try:
            if not is_running or active_session_id is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Biometric Recognition Stopped", (40, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, "Select Batch & Click Start Session", (70, 290), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                import time
                time.sleep(0.1)
                continue

            if cap is None or not cap.isOpened():
                import time
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                import time
                time.sleep(0.05)
                continue

            frame_count += 1

            if frame_count % process_every_n_frames == 0:
                import face_recognition
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                detected_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=0)
                if detected_locations:
                    detected_encodings = face_recognition.face_encodings(rgb_small_frame, detected_locations)
                    cached_locations = [(top * 4, right * 4, bottom * 4, left * 4) for (top, right, bottom, left) in detected_locations]
                    cached_encodings = detected_encodings
                else:
                    cached_locations = []
                    cached_encodings = []
                    last_face_location = None
                    last_matched_name = "Unknown"

                for (top, right, bottom, left), face_encoding in zip(cached_locations, cached_encodings):
                    # --- Best-of-N pose-robust matching: compare against ALL
                    # enrolled photo angles per student, keep the closest one. ---
                    distances = []
                    for student_id, details in active_encodings.items():
                        student_encodings = details.get("encodings") or ([details["encoding"]] if "encoding" in details else [])
                        if not student_encodings:
                            continue
                        per_angle_dists = face_recognition.face_distance(student_encodings, face_encoding)
                        best_dist = float(np.min(per_angle_dists))
                        distances.append((best_dist, student_id, details["name"]))

                    matched_name = "Unknown"
                    matched_student_id = None
                    decision = "UNKNOWN"
                    liveness_passed = True
                    min_dist = 1.0
                    quality_result = quality._empty_quality_result()

                    # --- Quality-aware feature assessment (illumination, blur,
                    # pose, resolution, occlusion) on the full-resolution frame ---
                    if config.ENABLE_QUALITY_AWARE_MATCHING:
                        try:
                            quality_result = quality.assess_face_quality(frame, (top, right, bottom, left))
                        except Exception as qe:
                            print(f"Quality assessment error: {qe}")
                            quality_result = quality._empty_quality_result()
                        quality_result["dominant_degradation"] = quality.dominant_degradation(quality_result)
                        composite_q = quality_result.get("composite_quality")
                        last_quality_status = (
                            f"Q={composite_q:.2f} ({quality_result['dominant_degradation']})"
                            if composite_q is not None else "Q=N/A"
                        )

                    if distances:
                        distances.sort()
                        min_dist, candidate_id, candidate_name = distances[0]
                        last_min_distance = min_dist

                        threshold = config.MATCH_THRESHOLD
                        margin = config.MATCH_MARGIN

                        threshold_passed = (min_dist < threshold) and (len(distances) == 1 or distances[1][0] - min_dist > margin)

                        ear_val = liveness_detector.compute_frame_ear(rgb_small_frame, (top//4, right//4, bottom//4, left//4))
                        liveness_passed, liveness_msg = liveness_detector.verify_liveness(candidate_name, ear_val)
                        last_liveness_status = liveness_msg

                        if threshold_passed:
                            if liveness_passed:
                                matched_name = candidate_name
                                matched_student_id = candidate_id
                                decision = "ACCEPTED"

                                # --- Quality gate: an otherwise-accepted match
                                # computed from a badly degraded frame is held
                                # back rather than trusted outright. The system
                                # keeps observing (person stays in frame) until
                                # a sufficiently good-quality frame arrives. ---
                                composite_q = quality_result.get("composite_quality")
                                if (config.ENABLE_QUALITY_AWARE_MATCHING and composite_q is not None
                                        and composite_q < config.MIN_QUALITY_TO_ACCEPT):
                                    decision = "REJECTED_LOW_QUALITY"
                            else:
                                decision = "REJECTED_LIVENESS"
                                matched_name = candidate_name
                        else:
                            decision = "REJECTED_DISTANCE" if min_dist >= threshold else "REJECTED_MARGIN"
                    else:
                        last_liveness_status = "No Rostered Encodings Loaded"

                    database.log_recognition_attempt(
                        session_id=active_session_id,
                        distance=min_dist,
                        matched_student_id=matched_student_id,
                        matched_name=matched_name,
                        decision=decision,
                        liveness_passed=liveness_passed,
                        quality=quality_result
                    )

                    # --- Quality-weighted temporal smoothing: frames with low
                    # composite quality contribute little to the displayed
                    # majority-vote identity, instead of being weighted equally
                    # to a sharp, well-lit, frontal frame. ---
                    vote_weight = quality_result.get("composite_quality")
                    if vote_weight is None:
                        vote_weight = 1.0
                    vote_weight = max(config.QUALITY_VOTE_FLOOR, vote_weight)

                    last_matches.append(matched_name)
                    last_match_weights.append(vote_weight)
                    if len(last_matches) > 10:
                        last_matches.pop(0)
                        last_match_weights.pop(0)

                    if len(last_matches) == 10:
                        weighted_votes = {}
                        for name, w in zip(last_matches, last_match_weights):
                            weighted_votes[name] = weighted_votes.get(name, 0.0) + w
                        unknown_weight = weighted_votes.get("Unknown", 0.0)
                        total_weight = sum(weighted_votes.values()) or 1.0
                        if unknown_weight / total_weight < 0.6:
                            smoothed_name = max(weighted_votes, key=weighted_votes.get)
                        else:
                            smoothed_name = matched_name
                    else:
                        smoothed_name = matched_name

                    if decision == "ACCEPTED" and matched_student_id:
                        database.mark_attendance_event(active_session_id, matched_student_id)

                    last_matched_name = smoothed_name
                    last_face_location = (top, right, bottom, left)

            if last_face_location is not None:
                top, right, bottom, left = last_face_location
                box_color = (0, 255, 0) if last_matched_name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
                label = f"{last_matched_name} (Dist: {last_min_distance:.2f})"
                cv2.putText(frame, label, (left, max(top - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

                if config.ENABLE_LIVENESS_CHECK:
                    cv2.putText(frame, f"Liveness: {last_liveness_status}", (20, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if config.ENABLE_QUALITY_AWARE_MATCHING:
                    cv2.putText(frame, f"Quality: {last_quality_status}", (20, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            import time
            time.sleep(0.03)

        except Exception as e:
            print(f"Exception in generate_frames: {e}")
            import time
            time.sleep(0.05)

# Routes
@app.route('/')
def login():
    if session.get('logged_in'):
        role = session.get('role')
        if role == 'ADMIN':
            return redirect(url_for('admin_dashboard'))
        elif role == 'FACULTY':
            return redirect(url_for('home'))
        elif role == 'STUDENT':
            return redirect(url_for('student_dashboard'))
    return render_template('login.html')

@app.route('/send-otp', methods=['POST'])
def send_otp():
    data = request.json or {}
    email = data.get("email", "").strip().lower()
    role = data.get("role", "FACULTY").strip().upper()

    if not email:
        return jsonify({"error": "Email address is required."}), 400

    if role == 'FACULTY':
        record = database.get_teacher_by_email(email)
        if not record:
            return jsonify({"error": f"Email '{email}' is not registered in the Faculty roster. Please contact Admin."}), 400
    elif role == 'STUDENT':
        record = database.get_student_by_email(email)
        if not record:
            return jsonify({"error": f"Email '{email}' is not registered in the Student roster. Please contact Admin."}), 400
    elif role == 'ADMIN':
        record = database.get_admin_by_email(email)
        if not record:
            return jsonify({"error": f"Email '{email}' is not registered as an Admin. Contact system administrator."}), 400
    else:
        return jsonify({"error": "Invalid role selected."}), 400

    otp = str(random.randint(100000, 999999))
    otp_storage[email] = {
        "otp": otp,
        "role": role,
        "created_at": time.time()
    }

    print(f"\n==========================================", flush=True)
    print(f"[OTP LOG] Generated OTP for {email} ({role}): {otp}", flush=True)
    print(f"==========================================\n", flush=True)

    mail_sent = False
    mail_error_msg = ""
    if config.MAIL_USERNAME and config.MAIL_PASSWORD:
        try:
            app.config['MAIL_SERVER'] = config.MAIL_SERVER
            app.config['MAIL_PORT'] = config.MAIL_PORT
            app.config['MAIL_USE_TLS'] = config.MAIL_USE_TLS
            app.config['MAIL_USERNAME'] = config.MAIL_USERNAME
            app.config['MAIL_PASSWORD'] = config.MAIL_PASSWORD
            app.config['MAIL_DEFAULT_SENDER'] = config.MAIL_DEFAULT_SENDER

            msg = Message(f"[{config.INSTITUTION_NAME}] Login OTP Verification", recipients=[email])
            msg.body = f"Your One-Time Password (OTP) for login as {role} is: {otp}\nValid for 5 minutes."
            mail.send(msg)
            print(f"[MAIL SENT SUCCESS] Dispatched OTP to {email}", flush=True)
            mail_sent = True
        except Exception as e:
            mail_error_msg = str(e)
            print(f"[MAIL ERROR] Failed to send email to {email}: {e}", flush=True)

    if mail_sent:
        return jsonify({"message": f"OTP has been dispatched to {email}."}), 200
    else:
        notice = "OTP generated!"
        if mail_error_msg:
            notice += f" (SMTP delivery failed). OTP logged to server console."
        else:
            notice += " (SMTP credentials not set). OTP logged to server console."

        if config.ALLOW_BYPASS_OTP:
            notice += " You can also use bypass code 123456."

        return jsonify({"message": notice}), 200

@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.json or {}
    email = data.get("email", "").strip().lower()
    entered_otp = data.get("otp", "").strip()
    requested_role = data.get("role", "").strip().upper()

    if not email or not entered_otp:
        return jsonify({"error": "Email and OTP are required."}), 400

    record = otp_storage.get(email)
    bypass_valid = config.ALLOW_BYPASS_OTP and (entered_otp == "123456")

    otp_valid = False
    if record and record["otp"] == entered_otp:
        # OTP valid for 5 minutes (300s)
        if time.time() - record.get("created_at", 0) <= 300:
            otp_valid = True
        else:
            otp_storage.pop(email, None)
            return jsonify({"error": "OTP has expired (valid for 5 minutes). Please request a new OTP."}), 400

    if otp_valid or bypass_valid:
        if record and otp_valid:
            role = record["role"]
            otp_storage.pop(email, None)
        else:
            if requested_role == 'ADMIN' and database.get_admin_by_email(email):
                role = 'ADMIN'
            elif requested_role == 'FACULTY' and database.get_teacher_by_email(email):
                role = 'FACULTY'
            elif requested_role == 'STUDENT' and database.get_student_by_email(email):
                role = 'STUDENT'
            elif database.get_admin_by_email(email):
                role = 'ADMIN'
            elif database.get_teacher_by_email(email):
                role = 'FACULTY'
            elif database.get_student_by_email(email):
                role = 'STUDENT'
            else:
                return jsonify({"error": "User not found in roster."}), 400

        session['logged_in'] = True
        session['user_email'] = email
        session['role'] = role

        if role == 'ADMIN':
            admin = database.get_admin_by_email(email)
            admin_dict = dict(admin) if admin else {}
            session['user_name'] = admin_dict.get('name') or email.split('@')[0].title()
            session['admin_id'] = admin_dict.get('id', 1)
            redirect_url = "/admin_dashboard"
        elif role == 'FACULTY':
            teacher = database.get_teacher_by_email(email)
            teacher_dict = dict(teacher) if teacher else {}
            session['user_name'] = teacher_dict.get('name') or email.split('@')[0].replace('_', ' ').title()
            session['teacher_id'] = teacher_dict.get('id', 1)
            redirect_url = "/Home"
        else:
            student = database.get_student_by_email(email)
            student_dict = dict(student) if student else {}
            session['user_name'] = student_dict.get('name') or email.split('@')[0].replace('_', ' ').title()
            session['student_id'] = student_dict.get('id', 1)
            redirect_url = "/student_dashboard"

        return jsonify({"message": "OTP verified successfully.", "redirect": redirect_url}), 200
    else:
        return jsonify({"error": "Invalid or expired OTP code."}), 400

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Admin Dashboard & APIs
@app.route('/admin_dashboard')
@admin_required
def admin_dashboard():
    teachers = database.get_all_teachers()
    students = database.get_all_students()
    batches = database.get_all_batches()
    timetable_entries = database.get_all_timetable_entries()
    sessions_list = database.get_all_sessions()
    recognition_logs = database.get_all_recognition_logs(limit=200)

    return render_template('admin_dashboard.html',
                           teachers=teachers,
                           students=students,
                           batches=batches,
                           timetable_entries=timetable_entries,
                           sessions=sessions_list,
                           recognition_logs=recognition_logs,
                           days_of_week=database.DAYS_OF_WEEK)

# Admin API Endpoints
@app.route('/admin/api/teachers', methods=['POST'])
@admin_required
def admin_add_teacher():
    data = request.json or {}
    name = data.get('name')
    email = data.get('email')
    teacher_code = data.get('teacher_code')
    if not name or not email:
        return jsonify({"error": "Name and email are required."}), 400
    try:
        teacher_id = database.create_teacher(name, email, teacher_code)
        return jsonify({"status": "success", "teacher_id": teacher_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/admin/api/teachers/<int:teacher_id>', methods=['PUT', 'DELETE'])
@admin_required
def admin_manage_teacher(teacher_id):
    if request.method == 'DELETE':
        database.delete_teacher(teacher_id)
        return jsonify({"status": "success", "message": "Teacher deleted."}), 200
    else:
        data = request.json or {}
        name = data.get('name')
        email = data.get('email')
        status = data.get('status', 'ACTIVE')
        teacher_code = data.get('teacher_code')
        ok, msg = database.update_teacher(teacher_id, name, email, status, teacher_code)
        if ok:
            return jsonify({"status": "success", "message": msg}), 200
        else:
            return jsonify({"status": "error", "error": msg}), 400

@app.route('/admin/api/students', methods=['POST'])
@admin_required
def admin_enroll_student():
    name = request.form.get('name')
    student_code = request.form.get('student_code')
    if not student_code or not student_code.strip():
        student_code = database.get_next_student_code()
    email = request.form.get('email')
    batch_id = request.form.get('batch_id')
    photo_file = request.files.get('photo')

    if not name or not email or not batch_id or not photo_file:
        return jsonify({"error": "All fields including a primary enrollment photo are required."}), 400

    os.makedirs(config.KNOWN_FACES_DIR, exist_ok=True)

    filename = werkzeug.utils.secure_filename(f"{student_code}_{photo_file.filename}")
    file_path = os.path.join(config.KNOWN_FACES_DIR, filename)
    photo_file.save(file_path)

    try:
        student_id = database.enroll_student(name, email, student_code, int(batch_id), filename)

        # --- Multi-angle enrollment (pose-robustness research gap) ---
        # Accepts any number of additional angle photos, submitted as
        # extra_photo_0/extra_label_0, extra_photo_1/extra_label_1, ...
        # from EITHER the file picker (multiple files) or the in-browser
        # camera capture widget. Each is stored as an additional row in
        # student_face_photos and used for best-of-N matching in the live
        # recognition loop.
        idx = 0
        added_angles = []
        while True:
            extra_file = request.files.get(f'extra_photo_{idx}')
            if not extra_file:
                break
            pose_label = request.form.get(f'extra_label_{idx}', f'angle_{idx}')
            extra_filename = werkzeug.utils.secure_filename(
                f"{student_code}_{pose_label}_{extra_file.filename}"
            )
            extra_path = os.path.join(config.KNOWN_FACES_DIR, extra_filename)
            extra_file.save(extra_path)
            database.add_student_face_photo(student_id, extra_filename, pose_label)
            added_angles.append(pose_label)
            idx += 1

        return jsonify({
            "status": "success",
            "student_id": student_id,
            "additional_angles_added": added_angles,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/admin/api/students/<int:student_id>/photo', methods=['POST'])
@admin_required
def admin_replace_student_photo(student_id):
    photo_file = request.files.get('photo')
    if not photo_file:
        return jsonify({"error": "Photo file required."}), 400

    students = database.get_all_students()
    st = next((s for s in students if s["id"] == student_id), None)
    if not st:
        return jsonify({"error": "Student not found."}), 404

    filename = werkzeug.utils.secure_filename(f"{st['student_code']}_{photo_file.filename}")
    file_path = os.path.join(config.KNOWN_FACES_DIR, filename)
    photo_file.save(file_path)

    database.update_student(student_id, st['name'], st['email'], st['student_code'], st['batch_id'], photo_filename=filename)
    return jsonify({"status": "success", "message": "Photo replaced successfully."}), 200

@app.route('/admin/api/students/<int:student_id>/photos', methods=['GET', 'POST'])
@admin_required
def admin_student_angle_photos(student_id):
    """
    Multi-angle enrollment management for pose-robust matching.
    GET  -> list this student's currently enrolled angle photos.
    POST -> add ONE additional angle photo (file field 'photo',
            form field 'pose_label', e.g. 'left_profile', 'right_profile',
            'upward', 'downward'). Called by both the file-picker path and
            the live camera-capture widget in admin_dashboard.html.
    """
    if request.method == 'GET':
        photos = database.get_student_face_photos(student_id)
        return jsonify({"status": "success", "photos": photos}), 200

    photo_file = request.files.get('photo')
    pose_label = request.form.get('pose_label', 'additional')
    if not photo_file:
        return jsonify({"error": "Photo file required."}), 400

    students = database.get_all_students()
    st = next((s for s in students if s["id"] == student_id), None)
    if not st:
        return jsonify({"error": "Student not found."}), 404

    os.makedirs(config.KNOWN_FACES_DIR, exist_ok=True)
    filename = werkzeug.utils.secure_filename(f"{st['student_code']}_{pose_label}_{photo_file.filename}")
    file_path = os.path.join(config.KNOWN_FACES_DIR, filename)
    photo_file.save(file_path)

    photo_id = database.add_student_face_photo(student_id, filename, pose_label)
    return jsonify({"status": "success", "photo_id": photo_id, "pose_label": pose_label}), 200

@app.route('/known_faces/<path:filename>')
def serve_known_face(filename):
    return send_from_directory(config.KNOWN_FACES_DIR, filename)

@app.route('/admin/api/students/<int:student_id>/photos/<int:photo_id>', methods=['DELETE'])
@admin_required
def admin_delete_student_angle_photo(student_id, photo_id):
    database.delete_student_face_photo(photo_id)
    return jsonify({"status": "success", "message": "Angle photo removed."}), 200

@app.route('/admin/api/students/<int:student_id>', methods=['PUT', 'DELETE'])
@admin_required
def admin_manage_student(student_id):
    if request.method == 'DELETE':
        database.delete_student(student_id)
        return jsonify({"status": "success", "message": "Student deleted."}), 200
    else:
        data = request.json or {}
        name = data.get('name')
        email = data.get('email')
        student_code = data.get('student_code')
        batch_id = data.get('batch_id')
        status = data.get('status', 'ACTIVE')
        if not name or not email:
            return jsonify({"status": "error", "error": "Name and email are required."}), 400
        ok, msg = database.update_student(student_id, name, email, student_code, batch_id, status=status)
        if ok:
            return jsonify({"status": "success", "message": msg}), 200
        else:
            return jsonify({"status": "error", "error": msg}), 400

@app.route('/admin/api/sessions/<int:session_id>', methods=['DELETE'])
@admin_required
def admin_delete_session(session_id):
    database.delete_session(session_id)
    return jsonify({"status": "success", "message": f"Session #{session_id} deleted."}), 200

@app.route('/admin/api/recognition_logs/<int:log_id>', methods=['DELETE'])
@admin_required
def admin_delete_recognition_log(log_id):
    database.delete_recognition_log(log_id)
    return jsonify({"status": "success", "message": f"Recognition log #{log_id} deleted."}), 200

@app.route('/admin/api/batches', methods=['POST'])
@admin_required
def admin_create_batch():
    data = request.json or {}
    batch_code = data.get('batch_code')
    batch_name = data.get('batch_name')
    teacher_id = data.get('teacher_id')
    if not batch_code or not batch_name:
        return jsonify({"error": "Batch code and batch name are required."}), 400

    try:
        t_id = int(teacher_id) if teacher_id else None
        batch_id = database.create_batch(batch_code, batch_name, t_id)
        return jsonify({"status": "success", "batch_id": batch_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/admin/api/batches/<int:batch_id>', methods=['DELETE'])
@admin_required
def admin_delete_batch(batch_id):
    database.delete_batch(batch_id)
    return jsonify({"status": "success", "message": "Batch deleted."}), 200

@app.route('/admin/api/timetable', methods=['POST'])
@admin_required
def admin_add_timetable():
    data = request.json or {}
    batch_id = data.get('batch_id')
    days_of_week = data.get('days_of_week') or ([data.get('day_of_week')] if data.get('day_of_week') else [])
    period_slots = data.get('period_slots') or ([data.get('period_slot')] if data.get('period_slot') else [])
    subject = data.get('subject')

    if not batch_id or not days_of_week or not period_slots or not subject:
        return jsonify({"error": "All timetable fields including at least one day and one time slot are required."}), 400

    try:
        added_ids = []
        for day in days_of_week:
            if not day or not str(day).strip():
                continue
            for slot in period_slots:
                if slot and str(slot).strip():
                    entry_id = database.add_timetable_entry(int(batch_id), str(day).strip(), str(slot).strip(), subject)
                    added_ids.append(entry_id)
        return jsonify({"status": "success", "added_count": len(added_ids), "entry_ids": added_ids}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/admin/api/next_sequential_ids', methods=['GET'])
@admin_required
def get_next_ids():
    return jsonify({
        "status": "success",
        "next_student_code": database.get_next_student_code(),
        "next_teacher_code": database.get_next_teacher_code()
    }), 200

@app.route('/admin/api/timetable/<int:entry_id>', methods=['DELETE'])
@admin_required
def admin_delete_timetable(entry_id):
    database.delete_timetable_entry(entry_id)
    return jsonify({"status": "success", "message": "Timetable slot removed."}), 200

# Faculty Dashboard & Session Controls
@app.route('/Home')
@faculty_required
def home():
    batches = database.get_all_batches()
    past_sessions = database.get_past_sessions_by_teacher(session.get('teacher_id'))
    return render_template('Home.html', batches=batches, past_sessions=past_sessions)

@app.route('/scan')
@faculty_required
def scan():
    batches = database.get_all_batches()
    return render_template('scan&mark.html', batches=batches, active_session_id=active_session_id)

@app.route('/video_feed')
@faculty_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
@faculty_required
def start():
    global is_running, active_session_id, active_batch_id, active_encodings, last_session_id
    data = request.json or {}
    batch_id = data.get('batch_id')
    period_slot = data.get('period_slot') or database.get_current_period_slot()
    test_condition = data.get('test_condition', 'normal')

    if not batch_id:
        batches = database.get_all_batches()
        if batches:
            batch_id = batches[0]['id']
        else:
            return jsonify({"error": "No batches available in the database."}), 400

    batch_id = int(batch_id)

    # Timetable Validation & 24/7 Testing Fallback
    day_of_week = datetime.now().strftime("%A")
    tt_entry = database.is_class_scheduled(batch_id, day_of_week, period_slot)
    if not tt_entry:
        tt_entry = {"subject": "Off-Hours / 24-7 Testing Mode", "period_slot": period_slot}

    active_batch_id = batch_id
    active_encodings = database.load_known_face_encodings(batch_id)
    session_id, session_uuid = database.create_session(batch_id, session.get('teacher_id'), period_slot=period_slot)
    active_session_id = session_id
    last_session_id = session_id

    # Set initial test condition if specified
    if test_condition:
        database.set_session_condition(session_id, test_condition)

    liveness_detector.reset()
    is_running = True
    print(f"Session {session_id} ({period_slot}, condition={test_condition}) started for Batch {batch_id} with {len(active_encodings)} face encodings.")
    return jsonify({
        "status": "started",
        "session_id": session_id,
        "period_slot": period_slot,
        "test_condition": test_condition,
        "encodings_loaded": len(active_encodings),
        "subject": tt_entry.get("subject", "")
    })

@app.route('/stop', methods=['POST'])
@faculty_required
def stop():
    global is_running, active_session_id, last_session_id, cap
    try:
        is_running = False
        ended_id = active_session_id
        if active_session_id:
            database.end_session(active_session_id)
            last_session_id = active_session_id
            active_session_id = None
        if cap is not None:
            try:
                cap.release()
            except Exception as e:
                print(f"Error releasing camera cap: {e}")
            cap = None
        print("Face recognition session stopped cleanly.")
        return jsonify({"status": "stopped", "session_id": ended_id or last_session_id})
    except Exception as ex:
        print(f"Error in /stop route: {ex}")
        is_running = False
        active_session_id = None
        cap = None
        return jsonify({"status": "stopped", "session_id": last_session_id or 0, "message": str(ex)})

@app.route('/api/session_condition', methods=['POST'])
@faculty_required
def set_session_condition():
    """
    Operator route to tag the active or specified test session with a
    degradation condition: normal, low_illumination, pose_variation,
    occlusion, low_resolution, motion_blur.
    """
    data = request.json or {}
    session_id = data.get('session_id') or active_session_id or last_session_id
    condition = data.get('condition', 'normal')
    if not session_id:
        return jsonify({"error": "No active or past session specified."}), 400

    database.set_session_condition(int(session_id), condition)
    return jsonify({"status": "success", "session_id": session_id, "condition": condition}), 200

@app.route('/api/recognize_image', methods=['POST'])
@faculty_required
def api_recognize_image():
    """
    Single-image face recognition & research quality assessment endpoint.
    Accepts an uploaded file ('photo') OR base64 camera string ('photo_b64').
    Evaluates face recognition unconditionally across all real-world conditions
    (low illumination, pose variation, occlusion, low resolution, motion blur),
    draws bounding boxes and identity matches, computes all research quality metrics,
    and returns annotated image base64 + quality breakdown + auto-detected condition.
    """
    img = None
    batch_id = request.form.get('batch_id') or (request.json and request.json.get('batch_id')) or active_batch_id
    
    if 'photo' in request.files and request.files['photo'].filename != '':
        file_bytes = np.frombuffer(request.files['photo'].read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif request.form.get('photo_b64') or (request.json and request.json.get('photo_b64')):
        b64_str = request.form.get('photo_b64') or (request.json and request.json.get('photo_b64'))
        if ',' in b64_str:
            b64_str = b64_str.split(',', 1)[1]
        file_bytes = np.frombuffer(base64.b64decode(b64_str), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "No valid image provided via file upload or camera capture."}), 400

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    encodings_map = {}
    if batch_id:
        try:
            encodings_map = database.load_known_face_encodings(int(batch_id))
        except Exception:
            encodings_map = database.load_known_face_encodings()
    else:
        encodings_map = database.load_known_face_encodings()

    import face_recognition
    locations = face_recognition.face_locations(rgb_img)
    encodings = face_recognition.face_encodings(rgb_img, locations) if locations else []

    annotated_img = img.copy()
    matches_result = []

    for (top, right, bottom, left), face_encoding in zip(locations, encodings):
        quality_res = quality.assess_face_quality(img, (top, right, bottom, left))
        quality_res["dominant_degradation"] = quality.dominant_degradation(quality_res)

        distances = []
        for student_id, details in encodings_map.items():
            student_encs = details.get("encodings") or ([details["encoding"]] if "encoding" in details else [])
            if not student_encs:
                continue
            per_angle_dists = face_recognition.face_distance(student_encs, face_encoding)
            best_dist = float(np.min(per_angle_dists))
            distances.append((best_dist, student_id, details["name"], details.get("student_code", "")))

        matched_name = "Unknown"
        student_code = ""
        matched_student_id = None
        min_dist = 1.0
        decision = "UNKNOWN"

        if distances:
            distances.sort()
            min_dist, candidate_id, candidate_name, c_code = distances[0]
            if min_dist < config.MATCH_THRESHOLD:
                matched_name = candidate_name
                matched_student_id = candidate_id
                student_code = c_code
                decision = "ACCEPTED"
            else:
                decision = "REJECTED_DISTANCE"

        box_color = (0, 255, 0) if matched_name != "Unknown" else (0, 0, 255)
        cv2.rectangle(annotated_img, (left, top), (right, bottom), box_color, 2)
        disp_label = f"{matched_name} ({min_dist:.2f})"
        cv2.putText(annotated_img, disp_label, (left, max(top - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        
        cond_label = f"Condition: {quality_res['dominant_degradation']}"
        cv2.putText(annotated_img, cond_label, (left, min(annotated_img.shape[0]-10, bottom + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        log_id = database.log_recognition_attempt(
            session_id=active_session_id or last_session_id,
            distance=min_dist,
            matched_student_id=matched_student_id,
            matched_name=matched_name,
            decision=decision,
            liveness_passed=True,
            quality=quality_res
        )

        matches_result.append({
            "log_id": log_id,
            "matched_name": matched_name,
            "student_code": student_code,
            "student_id": matched_student_id,
            "distance": round(min_dist, 4),
            "confidence": round(max(0.0, (1.0 - min_dist) * 100), 1),
            "decision": decision,
            "quality": quality_res,
            "face_location": {"top": top, "right": right, "bottom": bottom, "left": left}
        })

    _, buffer = cv2.imencode('.jpg', annotated_img)
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "status": "success",
        "faces_detected": len(locations),
        "matches": matches_result,
        "annotated_image": annotated_b64
    }), 200

@app.route('/api/log_research_condition', methods=['POST'])
@faculty_required
def api_log_research_condition():
    """
    Updates the logged recognition event with user-confirmed or tagged research condition.
    """
    data = request.json or {}
    log_id = data.get('log_id')
    condition = data.get('condition', 'normal')
    if not log_id:
        return jsonify({"error": "log_id is required."}), 400

    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE recognition_logs
        SET dominant_degradation = ?
        WHERE id = ?;
    """, (condition, int(log_id)))
    conn.commit()
    conn.close()

    return jsonify({"status": "success", "log_id": log_id, "tagged_condition": condition}), 200

@app.route('/api/session_status')
@app.route('/session_status')
@faculty_required
def session_status():
    return jsonify({
        "is_running": is_running,
        "active_session_id": active_session_id,
        "last_session_id": last_session_id,
        "active_batch_id": active_batch_id
    })

@app.route('/attendance')
@faculty_required
def attendance():
    batches = database.get_all_batches()
    return render_template('attendance.html', batches=batches)

@app.route('/attendance_data')
@faculty_required
def attendance_data():
    target_id = active_session_id or last_session_id
    if not target_id:
        return jsonify([])
    report = database.get_session_attendance_report(target_id)
    return jsonify(report)

@app.route('/api/session_report/<int:session_id>')
@faculty_required
def session_report(session_id):
    report = database.get_session_attendance_report(session_id)
    session_info = database.get_session_by_id(session_id)
    return jsonify({"session": session_info, "roster": report})

@app.route('/send_absentee_emails', methods=['POST'])
@faculty_required
def send_emails():
    success, msg = send_absentee_emails(active_session_id)
    status_str = "success" if success else "error"
    return jsonify({"status": status_str, "message": msg}), (200 if success else 500)

@app.route('/toggle_attendance', methods=['POST'])
@faculty_required
def toggle_attendance():
    data = request.json or {}
    student_id = data.get("student_id")
    target_session_id = data.get("session_id") or active_session_id
    new_status = data.get("status", "Present")

    if not student_id or not target_session_id:
        return jsonify({"error": "Student ID and session ID are required."}), 400

    success = database.update_attendance_status(target_session_id, student_id, new_status)
    if success:
        return jsonify({"status": "success", "message": f"Attendance status updated to {new_status}."}), 200
    else:
        return jsonify({"error": "Failed to update attendance status."}), 500

# Student Role Dashboard
@app.route('/student_dashboard')
@student_required
def student_dashboard():
    email = session.get('user_email')
    stats = database.get_student_dashboard_stats(email)
    if not stats:
        return "Student record not found.", 404
    return render_template('student_dashboard.html', stats=stats)

if __name__ == "__main__":
    print(f"Starting generic biometric attendance server at http://127.0.0.1:5000")
    print(f"Institution Branding: {config.INSTITUTION_NAME} ({config.INSTITUTION_DOMAIN})")
    app.run(debug=True, host='0.0.0.0', port=5000)