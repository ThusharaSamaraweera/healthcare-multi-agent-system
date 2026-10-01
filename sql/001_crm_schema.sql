-- ============================================================================
-- CRM: LOCATIONS (matches ORM: Location.__tablename__ = "locations")
-- ============================================================================

CREATE TABLE IF NOT EXISTS locations (
    location_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL CHECK (type IN ('HOSPITAL','OPD','LAB','CLINIC')),
    address TEXT,
    tz TEXT NOT NULL,  -- IANA timezone
    lat REAL,
    lng REAL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

-- ============================================================================
-- CRM: SPECIALTIES (matches ORM: Specialty.__tablename__ = "specialties")
-- ============================================================================

CREATE TABLE IF NOT EXISTS specialties (
    specialty_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

-- ============================================================================
-- CRM: DOCTORS (matches ORM: Doctor.__tablename__ = "doctors")
-- ============================================================================

CREATE TABLE IF NOT EXISTS doctors (
    doctor_id TEXT PRIMARY KEY,
    full_name TEXT NOT NULL,
    specialty_id TEXT REFERENCES specialties(specialty_id) ON DELETE SET NULL,
    license_no TEXT UNIQUE,
    phone TEXT,
    email TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_doctors_specialty ON doctors(specialty_id);

-- ============================================================================
-- CRM: PATIENTS (matches ORM: Patient.__tablename__ = "patients")
-- ============================================================================

CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    external_user_id TEXT NOT NULL UNIQUE,  -- Phone number without '+'
    full_name TEXT NOT NULL,
    dob TEXT,  -- ISO format YYYY-MM-DD
    gender TEXT CHECK (gender IN ('M','F','X')),
    phone TEXT,
    email TEXT,
    notes TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_patients_external_user_id ON patients(external_user_id);
CREATE INDEX IF NOT EXISTS idx_patients_phone ON patients(phone);

-- ============================================================================
-- CRM: BOOKINGS (matches ORM: Booking.__tablename__ = "bookings")
-- ============================================================================

CREATE TABLE IF NOT EXISTS bookings (
    booking_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
    doctor_id TEXT NOT NULL REFERENCES doctors(doctor_id) ON DELETE CASCADE,
    location_id TEXT NOT NULL REFERENCES locations(location_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    reason TEXT,
    start_at INTEGER NOT NULL,  -- Epoch seconds UTC
    end_at INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('PENDING','CONFIRMED','RESCHEDULED','CANCELLED','NO_SHOW','COMPLETED')),
    source TEXT NOT NULL CHECK (source IN ('CRM','MEMORY','MIGRATED','SEED')),
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bookings_patient ON bookings(patient_id);
CREATE INDEX IF NOT EXISTS idx_bookings_doctor ON bookings(doctor_id);
CREATE INDEX IF NOT EXISTS idx_bookings_location ON bookings(location_id);
CREATE INDEX IF NOT EXISTS idx_bookings_start_at ON bookings(start_at DESC);
CREATE INDEX IF NOT EXISTS idx_bookings_status ON bookings(status);

