from loguru import logger
import uuid
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import and_
from sqlalchemy.orm import sessionmaker

from infrastructure.config import TIMEZONE
from infrastructure.db import get_sql_engine
from infrastructure.db.crm_models import (
    Booking,
    Doctor,
    Location,
    Patient,
    Specialty,
)
from infrastructure.observability import observe, update_current_observation
class CRMTool:

    def __init__(self) -> None:
        self.engine = get_sql_engine()
        self._tz = ZoneInfo(TIMEZONE)

    def _session(self):
        factory = sessionmaker(bind=self.engine)
        return factory()

    @staticmethod
    def _epoch_to_str(epoch: int) -> str:
        return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")

    def _epoch_to_local(self, epoch: int) -> str:
        return datetime.fromtimestamp(epoch, self._tz).strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _doctor_label(full_name: Optional[str]) -> str:
        if not full_name:
            return "—"
        name = full_name.strip()
        if name.lower().startswith(("dr.", "dr ")):
            return name
        return f"Dr. {name}"

    def _parse_iso_datetime(self, value: str) -> Optional[int]:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.strip().replace(" ", "T"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self._tz)
        return int(dt.timestamp())

    def patient_id_for_external_user(self, external_user_id: str) -> Optional[str]:
        session = self._session()
        try:
            patient = (
                session.query(Patient)
                .filter(Patient.external_user_id == external_user_id)
                .first()
            )
            return patient.patient_id if patient else None
        finally:
            session.close()

    def lookup_patient(
        self,
        phone: Optional[str] = None,
        name: Optional[str] = None,
        patient_id: Optional[str] = None,
        external_user_id: Optional[str] = None,
    ) -> str:
        session = self._session()
        try:
            query = session.query(Patient)

            if patient_id:
                query = query.filter(Patient.patient_id == patient_id)
            elif external_user_id:
                query = query.filter(Patient.external_user_id == external_user_id)
            elif phone:
                query = query.filter(Patient.phone == phone)
            elif name:
                query = query.filter(Patient.full_name.ilike(f"%{name}%"))
            else:
                return "No search criteria provided. Please supply phone, name, or patient ID."

            patients = query.limit(5).all()

            if not patients:
                return "No patient found matching the given criteria."

            lines: List[str] = []
            for pat in patients:
                lines.append(
                    f"• {pat.full_name}  |  Phone: {pat.phone or 'N/A'}  "
                    f"|  DOB: {pat.dob or 'N/A'}  |  ID: {pat.patient_id}"
                )

                # Fetch upcoming bookings
                now_epoch = int(time.time())
                bookings = (
                    session.query(Booking)
                    .filter(
                        and_(
                            Booking.patient_id == pat.patient_id,
                            Booking.start_at >= now_epoch,
                            Booking.status.in_(["PENDING", "CONFIRMED", "RESCHEDULED"]),
                        )
                    )
                    .order_by(Booking.start_at)
                    .limit(5)
                    .all()
                )

                if bookings:
                    lines.append("  Upcoming bookings:")
                    for bk in bookings:
                        doctor = session.query(Doctor).get(bk.doctor_id)
                        loc = session.query(Location).get(bk.location_id)
                        lines.append(
                            f"    - {self._epoch_to_str(bk.start_at)} → "
                            f"{self._epoch_to_str(bk.end_at)}  "
                            f"| Dr. {doctor.full_name if doctor else 'N/A'} "
                            f"| {loc.name if loc else 'N/A'} "
                            f"| Status: {bk.status}  | Booking ID: {bk.booking_id}"
                        )
                else:
                    lines.append("  No upcoming bookings.")

            return "\n".join(lines)

        except Exception as exc:
            logger.error("lookup_patient failed: {}", exc)
            return f"Error looking up patient: {exc}"
        finally:
            session.close()

    def search_doctors(
        self,
        specialty: Optional[str] = None,
        location: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        session = self._session()
        try:
            query = session.query(Doctor).filter(Doctor.active == 1)

            if specialty:
                query = query.join(Specialty).filter(
                    Specialty.name.ilike(f"%{specialty}%")
                )
            if name:
                query = query.filter(Doctor.full_name.ilike(f"%{name}%"))
            if location:
                # Find doctors who have bookings at this location
                loc_sub = (
                    session.query(Location.location_id)
                    .filter(Location.name.ilike(f"%{location}%"))
                    .subquery()
                )
                booking_docs = (
                    session.query(Booking.doctor_id)
                    .filter(Booking.location_id.in_(loc_sub.select()))
                    .distinct()
                    .subquery()
                )
                query = query.filter(Doctor.doctor_id.in_(booking_docs.select()))

            doctors = query.limit(10).all()

            if not doctors:
                return "No doctors found matching the criteria."

            lines: List[str] = []
            for doc in doctors:
                spec_name = doc.specialty.name if doc.specialty else "General"
                lines.append(
                    f"• Dr. {doc.full_name}  |  Specialty: {spec_name}  "
                    f"|  Phone: {doc.phone or 'N/A'}  |  ID: {doc.doctor_id}"
                )

            return "\n".join(lines)

        except Exception as exc:
            logger.error("search_doctors failed: {}", exc)
            return f"Error searching doctors: {exc}"
        finally:
            session.close()

    def create_booking(
        self,
        patient_id: str,
        start_at: Optional[str] = None,
        doctor_name: Optional[str] = None,
        specialty: Optional[str] = None,
        doctor_id: Optional[str] = None,
        location_name: Optional[str] = None,
        location_id: Optional[str] = None,
        duration_minutes: int = 30,
        title: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> str:
        session = self._session()
        try:
            patient = session.get(Patient, patient_id)
            if not patient:
                return f"Patient {patient_id} not found."

            start_epoch = self._parse_iso_datetime(start_at) if start_at else None
            if start_epoch is None:
                return "Cannot book — no valid start time provided."
            if start_epoch <= int(time.time()):
                return "Cannot book — appointments must be scheduled in the future."

            doctor: Optional[Doctor] = None
            if doctor_id:
                doctor = session.get(Doctor, doctor_id)
            elif doctor_name:
                doctors = (
                    session.query(Doctor)
                    .outerjoin(Specialty, Specialty.specialty_id == Doctor.specialty_id)
                    .filter(Doctor.active == 1)
                    .filter(Doctor.full_name.ilike(f"%{doctor_name}%"))
                )
                if specialty:
                    doctors = doctors.filter(Specialty.name.ilike(f"%{specialty}%"))
                doctor_list = doctors.order_by(Doctor.full_name).limit(5).all()
                if not doctor_list:
                    return f"No active doctor matching '{doctor_name}'."
                if len(doctor_list) > 1:
                    rows = [
                        f"| {self._doctor_label(doc.full_name)} | "
                        f"{doc.specialty.name if doc.specialty else '—'} |"
                        for doc in doctor_list
                    ]
                    return (
                        f"Multiple doctors match '{doctor_name}'. Please clarify which one:\n\n"
                        "| Name | Specialty |\n|---|---|\n" + "\n".join(rows)
                    )
                doctor = doctor_list[0]
            elif specialty:
                doctor = (
                    session.query(Doctor)
                    .outerjoin(Specialty, Specialty.specialty_id == Doctor.specialty_id)
                    .filter(Doctor.active == 1)
                    .filter(Specialty.name.ilike(f"%{specialty}%"))
                    .order_by(Doctor.full_name)
                    .first()
                )
                if doctor is None:
                    return f"No active doctor found in specialty '{specialty}'."
            else:
                return "Cannot book — no doctor or specialty provided."

            loc: Optional[Location] = None
            if location_id:
                loc = session.get(Location, location_id)
            elif location_name:
                loc = session.query(Location).filter(
                    Location.name.ilike(f"%{location_name}%")
                ).first()
            if loc is None:
                doctor_loc = (
                    session.query(Booking, Location)
                    .join(Location, Location.location_id == Booking.location_id)
                    .filter(Booking.doctor_id == doctor.doctor_id)
                    .filter(Location.type.in_(("OPD", "CLINIC", "HOSPITAL")))
                    .order_by(Booking.start_at.desc())
                    .first()
                )
                if doctor_loc is not None:
                    loc = doctor_loc[1]
            if loc is None:
                loc = (
                    session.query(Location)
                    .filter(Location.active == 1)
                    .filter(Location.type.in_(("OPD", "CLINIC")))
                    .order_by(Location.name)
                    .first()
                )
            if loc is None:
                loc = session.query(Location).filter(Location.active == 1).order_by(Location.name).first()
            if loc is None:
                return "Cannot book — no available location configured."

            end_epoch = start_epoch + duration_minutes * 60

            # Conflict check
            conflict = (
                session.query(Booking)
                .filter(
                    and_(
                        Booking.doctor_id == doctor.doctor_id,
                        Booking.status.in_(["PENDING", "CONFIRMED"]),
                        Booking.start_at < end_epoch,
                        Booking.end_at > start_epoch,
                    )
                )
                .first()
            )
            if conflict:
                return (
                    f"Conflict: Dr. {doctor.full_name} already has a booking at "
                    f"{self._epoch_to_str(conflict.start_at)}–{self._epoch_to_str(conflict.end_at)}. "
                    "Please choose another time."
                )

            booking_id = str(uuid.uuid4())
            now_epoch = int(time.time())

            new_booking = Booking(
                booking_id=booking_id,
                patient_id=patient_id,
                doctor_id=doctor.doctor_id,
                location_id=loc.location_id,
                title=title or f"{doctor.specialty.name if doctor.specialty else 'General'} appointment",
                reason=reason,
                start_at=start_epoch,
                end_at=end_epoch,
                status="CONFIRMED",
                source="MEMORY",
                created_at=now_epoch,
                updated_at=now_epoch,
            )
            session.add(new_booking)
            session.commit()

            return (
                f"✅ Booking confirmed!\n"
                f"  Booking ID: {booking_id}\n"
                f"  Patient: {patient.full_name}\n"
                f"  Doctor: {self._doctor_label(doctor.full_name)} ({doctor.specialty.name if doctor.specialty else 'General'})\n"
                f"  Location: {loc.name}\n"
                f"  Time: {self._epoch_to_str(start_epoch)} → {self._epoch_to_str(end_epoch)}\n"
                f"  Reason: {reason or 'N/A'}"
            )

        except Exception as exc:
            session.rollback()
            logger.error("create_booking failed: {}", exc)
            return f"Error creating booking: {exc}"
        finally:
            session.close()


    @observe(name="crm_dispatch")
    def dispatch(self, action: str, params: Dict[str, Any]) -> str:
        handler_map = {
            "lookup_patient": self.lookup_patient,
            "search_doctors": self.search_doctors,
            "create_booking": self.create_booking,
        }
        handler = handler_map.get(action)
        if not handler:
            return f"Unknown CRM action: {action}. Available: {list(handler_map.keys())}"

        update_current_observation(
            input=f"action={action} params={params}",
        )

        start = time.time()
        result = handler(**params)
        latency_ms = int((time.time() - start) * 1000)

        update_current_observation(
            output=result[:500],
            metadata={"action": action, "latency_ms": latency_ms},
        )

        return result
