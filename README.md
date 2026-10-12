# Generic Research-Grade Biometric Attendance System

An institutional, research-grade biometric attendance management system built with Flask, OpenCV, Dlib facial landmarks, and SQLite. Designed for academic research and publication evaluation across higher education institutions.

---

## Key Technical Features

1. **Security & Zero-Hardcoding Architecture**:
   - Environment-driven configuration via `python-dotenv` and `config.py`.
   - Cross-platform relative file paths for universal deployment across Linux, macOS, and Windows.
   - Zero hardcoded email passwords, personal names, or university branding.

2. **Institution Generalization**:
   - Configurable institution branding (`INSTITUTION_NAME`, `INSTITUTION_DOMAIN`) in `.env`.
   - Adapts to any college, university, or academic organization without code modifications.

3. **Persistent SQLite Attendance Engine**:
   - Replaces transient state with an ACID-compliant SQLite relational database (`attendance_system.db`).
   - Attendance marks are recorded **strictly** from live webcam facial recognition events. No pre-populated, fake, or synthetic records.

4. **Multi-Batch Roster Support**:
   - Supports course/batch rosters (`batches`, `students` tables).
   - Facial encodings are cached dynamically per batch roster during active recognition sessions.

5. **Dual Role Authentication (OTP-Based)**:
   - **Faculty Role**: Select course batch, launch live video recognition stream, view real-time and historical attendance reports, and dispatch absentee emails.
   - **Student Role**: View personalized dashboard showing total classes held, classes attended, and attendance percentage per batch. Enforces strict student data privacy and isolation.

6. **Real-Time Liveness / Anti-Spoofing Detection**:
   - Eye Aspect Ratio (EAR) blink verification algorithm calculated from 68 facial landmarks.
   - Rejects static photo attacks and screen replay spoofing attempts.
   - Toggleable via `ENABLE_LIVENESS_CHECK` in `.env` for controlled baseline vs. anti-spoofing experiments.

7. **Attempt-Level Biometric Logging**:
   - Logs every recognition event to `recognition_logs`: timestamp, computed distance metric, matched candidate, decision (`ACCEPTED`, `REJECTED_DISTANCE`, `REJECTED_MARGIN`, `REJECTED_LIVENESS`), and liveness pass/fail status.

8. **Offline Research Evaluation Suite (`evaluate.py`)**:
   - Standalone biometric performance evaluation tool.
   - Joins real logged events with user-annotated `ground_truth.csv`.
   - Calculates Accuracy, False Accept Rate (FAR), False Reject Rate (FRR), and Equal Error Rate (EER) across threshold sweeps.
   - Plots and saves high-resolution ROC curve visualization (`evaluation_roc_curve.png`).

---

## Biometric System Formulation

### Facial Encoding Matching
Candidate faces detected in live webcam frames are encoded into 128-dimensional vector representations $\mathbf{v}_{live} \in \mathbb{R}^{128}$ using dlib's ResNet face recognition model. The Euclidean distance metric to known roster encodings $\mathbf{v}_{known}$ is computed:

$$d(\mathbf{v}_{live}, \mathbf{v}_{known}) = \|\mathbf{v}_{live} - \mathbf{v}_{known}\|_2$$

A candidate match is accepted iff:
$$d_{min} < T \quad \text{and} \quad (d_{second} - d_{min}) > M$$

Where $T$ is `MATCH_THRESHOLD` (default 0.60) and $M$ is `MATCH_MARGIN` (default 0.10).

### Eye Aspect Ratio (EAR) Anti-Spoofing Formula
For facial eye landmarks $(p_1, p_2, p_3, p_4, p_5, p_6)$:

$$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2.0 \cdot \|p_1 - p_4\|}$$

A dynamic eye blink causes $\text{EAR}$ to dip below $\text{EAR\_THRESHOLD}$ (default 0.20), confirming dynamic proof-of-life.

---

## Project Structure

```text
├── config.py             # Centralized environment loader & system settings
├── database.py           # SQLite connection manager, schema, and queries
├── database_setup.py     # Script to initialize database schema & sync faces
├── liveness.py           # EAR calculation and blink anti-spoofing module
├── server.py             # Flask web server (Dual role auth, video feed, APIs)
├── main.py               # Absentee email dispatch module
├── evaluate.py           # Offline research evaluation script (ROC, FAR, FRR, EER)
├── .env.example          # Environment variables template
├── requirements.txt      # Python dependencies
├── known_faces/          # Enrolled student face image directory
└── templates/            # Generic Jinja HTML templates (Faculty & Student)
```

---

## Quick Start & Setup Guide

### 1. Installation
Clone the repository and install required dependencies:

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` and set your institution configuration and mail parameters:

```bash
cp .env.example .env
```

Example `.env`:
```env
INSTITUTION_NAME=Universal Institute of Technology
INSTITUTION_DOMAIN=univ.edu
SECRET_KEY=secure_random_key_123

MATCH_THRESHOLD=0.60
MATCH_MARGIN=0.10
ENABLE_LIVENESS_CHECK=True
EAR_THRESHOLD=0.20
```

### 3. Initialize Database & Sync Faces
Place student reference images (e.g. `john_doe.jpg`) in the `known_faces/` folder, then run:

```bash
python database_setup.py
```

### 4. Run Server
Start the Flask application:

```bash
python server.py
```

Navigate to `http://127.0.0.1:5000` in your web browser.

---

## Running Offline Research Evaluation

1. Conduct real webcam attendance recognition sessions via the Faculty Portal (`/scan`).
2. Run `evaluate.py` to generate the `ground_truth.csv` template:
   ```bash
   python evaluate.py
   ```
3. Annotate `ground_truth.csv` with human ground-truth labels (`correct_match`, `wrong_match`, `correctly_rejected_spoof`, `missed_spoof`).
4. Re-run `evaluate.py` to calculate system performance:
   ```bash
   python evaluate.py
   ```
   The evaluation summary will print to stdout and save `evaluation_roc_curve.png`.
