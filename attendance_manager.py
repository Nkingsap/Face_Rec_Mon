"""
Attendance Manager for the AI Face Recognition and Monitoring System.
Provides high-level attendance operations using the database handler.
"""
import csv
import os
from datetime import date, datetime


class AttendanceManager:
    """Manages attendance records and report generation."""

    def __init__(self, db_handler, reports_dir):
        self.db = db_handler
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def mark_attendance(self, name, user_id=None):
        """Mark attendance for a recognized person."""
        return self.db.mark_attendance(name, user_id)

    def get_attendance(self, date_filter=None):
        """Get attendance records, optionally filtered by date."""
        return self.db.get_attendance(date_filter)

    def get_today_stats(self):
        """Get today's statistics for the dashboard."""
        return {
            'total_users': self.db.get_user_count(),
            'today_attendance': self.db.get_today_attendance_count(),
            'unknown_alerts': self.db.get_unknown_faces_count(),
            'date': date.today().strftime('%B %d, %Y'),
        }

    def export_to_csv(self, date_filter=None):
        """
        Export attendance records to a CSV file.
        Returns the file path of the generated CSV.
        """
        records = self.db.get_attendance(date_filter)

        if date_filter:
            filename = f"attendance_{date_filter}.csv"
        else:
            filename = f"attendance_all_{date.today().strftime('%Y%m%d')}.csv"

        filepath = os.path.join(self.reports_dir, filename)

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Record ID', 'User ID', 'Name', 'Date',
                'Check-in Time', 'Status'
            ])
            for record in records:
                writer.writerow([
                    record['record_id'],
                    record['user_id'],
                    record['name'],
                    record['date'],
                    record['check_in_time'],
                    record['status']
                ])

        return filepath, filename

    def get_attendance_summary(self, date_filter=None):
        """Get a summary of attendance statistics."""
        records = self.get_attendance(date_filter)
        total = len(records)
        present = sum(1 for r in records if r['status'] == 'Present')
        late = sum(1 for r in records if r['status'] == 'Late')

        return {
            'total': total,
            'present': present,
            'late': late,
            'date': date_filter or date.today().strftime('%Y-%m-%d')
        }
