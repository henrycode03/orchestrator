"""Reset password for test users"""

from app.database import SessionLocal
from app.models import User
from app.auth import get_password_hash

def reset_password(email: str, new_password: str):
    """Reset password for a user"""
    db = SessionLocal()
    
    user = db.query(User).filter(User.email == email).first()
    if user:
        user.hashed_password = get_password_hash(new_password)
        db.commit()
        db.refresh(user)
        print(f"✅ Password reset for: {email}")
    else:
        print(f"❌ User not found: {email}")
    
    db.close()

# Reset passwords for common test emails
print("Resetting passwords...")
reset_password("eric@example.com", "test123")
reset_password("demo@demo.com", "demo123")
reset_password("admin@orchestrator.local", "admin123")
reset_password("test@test.com", "test123")

print("\nYou can now login with these credentials:")
print("  eric@example.com / test123")
print("  demo@demo.com / demo123")
print("  admin@orchestrator.local / admin123")
print("  test@test.com / test123")
