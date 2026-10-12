import os
from dotenv import load_dotenv

# Load environment variables from .env file if available
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path, override=True)

# System Branding & Configuration
INSTITUTION_NAME = os.getenv("INSTITUTION_NAME", "Universal Institute of Technology")
INSTITUTION_DOMAIN = os.getenv("INSTITUTION_DOMAIN", "srmap.edu.in")
SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key_change_in_production_12345")

# File Paths (Cross-Platform Relative Paths)
DB_PATH = os.path.join(BASE_DIR, "attendance_system.db")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# SMTP Mail Configuration
MAIL_SERVER = os.getenv("MAIL_SERVER", "smtp.gmail.com")
MAIL_PORT = int(os.getenv("MAIL_PORT", "587"))
MAIL_USE_TLS = os.getenv("MAIL_USE_TLS", "True").lower() in ["true", "1", "yes"]
MAIL_USERNAME = os.getenv("MAIL_USERNAME", "samhitha_baluvuri@srmap.edu.in").strip()
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD", "ijai wabk mayy vxhf").replace(" ", "").strip()
MAIL_DEFAULT_SENDER = os.getenv("MAIL_DEFAULT_SENDER", MAIL_USERNAME).strip()

# Security & Debug Parameters
ALLOW_BYPASS_OTP = os.getenv("ALLOW_BYPASS_OTP", "True").lower() in ["true", "1", "yes"]

# Biometric & Liveness Parameters
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.6"))
MATCH_MARGIN = float(os.getenv("MATCH_MARGIN", "0.1"))
ENABLE_LIVENESS_CHECK = os.getenv("ENABLE_LIVENESS_CHECK", "True").lower() in ["true", "1", "yes"]
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", "0.2"))
EAR_CONSEC_FRAMES = int(os.getenv("EAR_CONSEC_FRAMES", "2"))

# Quality-Aware Face Recognition Parameters
ENABLE_QUALITY_AWARE_MATCHING = os.getenv("ENABLE_QUALITY_AWARE_MATCHING", "True").lower() in ["true", "1", "yes"]
QUALITY_WEIGHT_ILLUMINATION = float(os.getenv("QUALITY_WEIGHT_ILLUMINATION", "0.25"))
QUALITY_WEIGHT_BLUR = float(os.getenv("QUALITY_WEIGHT_BLUR", "0.25"))
QUALITY_WEIGHT_POSE = float(os.getenv("QUALITY_WEIGHT_POSE", "0.20"))
QUALITY_WEIGHT_RESOLUTION = float(os.getenv("QUALITY_WEIGHT_RESOLUTION", "0.15"))
QUALITY_WEIGHT_OCCLUSION = float(os.getenv("QUALITY_WEIGHT_OCCLUSION", "0.15"))

BLUR_REFERENCE_VARIANCE = float(os.getenv("BLUR_REFERENCE_VARIANCE", "100.0"))
MIN_FACE_SIZE_PX = int(os.getenv("MIN_FACE_SIZE_PX", "80"))
MAX_YAW_DEGREES = float(os.getenv("MAX_YAW_DEGREES", "45.0"))
MIN_QUALITY_TO_ACCEPT = float(os.getenv("MIN_QUALITY_TO_ACCEPT", "0.35"))
QUALITY_VOTE_FLOOR = float(os.getenv("QUALITY_VOTE_FLOOR", "0.1"))


