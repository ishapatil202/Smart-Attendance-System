from apscheduler.schedulers.background import BackgroundScheduler
import pymysql
from datetime import datetime
import smtplib
from mark_attendance import Mark_Attendance
from email.message import EmailMessage
from config import ATTENDANCE_DETAILS_DIR, DATABASE_CONFIG, EMAIL_CONFIG


def get_db_connection():
    return pymysql.connect(**DATABASE_CONFIG)


def email_is_configured():
    return all(EMAIL_CONFIG[key] for key in ("username", "password", "sender"))

def getall_staffs():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("select eid,email_address from attendance")
    data = cur.fetchall()
    all_staffs = {}
    if len(data) != 0:
        for(id,email) in data:
            all_staffs[id] = email
    conn.close()
    return all_staffs

def registered_vs_absent_staffs(all_staffs):
    dt = datetime.now()
    dt = dt.strftime("%Y-%m-%d %I:%M:%S")
    date = str(dt).split(' ')[0]
    time = str(dt).split(' ')[1]
    time_hour = time.split(':')[0]
    time_minute = time.split(':')[1]
    start_hour = 1
    end_hour = 11
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("select id,time from report where date=%s ",(date))
    output = cur.fetchall()
    if len(output)!= 0:
        registered_staff_ids = []
        for(x,y) in output:
            y_hour = y.split(":")[0]
            y_minute = y.split(":")[1]
            if(int(y_hour) >= start_hour and int(y_hour) <= end_hour):
                registered_staff_ids.append(int(x))
        absent_staff_ids = []
        all_staff_ids = list(all_staffs.keys())
        for x in all_staff_ids:
            if x not in registered_staff_ids:
                absent_staff_ids.append(x)
    else:
        absent_staff_ids = list(all_staffs.keys())
    conn.close()
    return absent_staff_ids

def absent_emails():
    all_staffs = getall_staffs()
    absent_staff_ids = registered_vs_absent_staffs(all_staffs)
    absent_staff_emails = []
    for item in absent_staff_ids:
        email = all_staffs[item]
        absent_staff_emails.append(email)
    return absent_staff_emails

def get_manager_email():
    department = "Faculty"
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("select email_address from attendance where department=%s ",(department))
    data = cur.fetchall()
    if len(data) != 0:
        manager_email = data[0][0]
    conn.close()
    return manager_email

def generate_attendance_sheet():
    dt = datetime.now()
    date = str(dt).split(' ')[0]
    csv_name = str(ATTENDANCE_DETAILS_DIR / 'attendance_{}.csv'.format(date))
    mark_attendance_obj = Mark_Attendance(csv_filename=csv_name)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('select * from report where date = %s ', (date))
    mydata = cur.fetchall()
    if len(mydata) < 1:
        print("No data found in database")
    else:
        mark_attendance_obj.write_csv_header(id ='Id',staff_name='Staff_Name',date='Date',time='Time',status='Status')
        for items in mydata:
            attendance_record = list(items)
            mark_attendance_obj.append_csv_rows(records=attendance_record)
    return csv_name


def send_mail():
    all_staffs = getall_staffs()
    absent_staff_ids = registered_vs_absent_staffs(all_staffs)
    absent_staff_emails = absent_emails()
    print(absent_staff_emails)
    dt = datetime.now()
    dt = dt.strftime("%Y-%m-%d %I:%M:%S")
    date = str(dt).split(' ')[0]
    time = str(dt).split(' ')[1]
    for id in absent_staff_ids:
        status = "Absent"
        conn = get_db_connection()
        cur1 = conn.cursor()
        cur1.execute("select fname from attendance where eid=%s ",id)
        output = cur1.fetchone()
        (name,) = output
        cur2 = conn.cursor()
        cur2.execute("insert into report(id,name,date,time,status) VALUES (%s,%s,%s,%s,%s)", (id,
                                                                                              name,
                                                                                              date,
                                                                                              time,
                                                                                              status))
        conn.commit()
        conn.close()
        print("Attendance for absent students has been recorded successfully")

    server = smtplib.SMTP(EMAIL_CONFIG['host'], EMAIL_CONFIG['port'])
    server.starttls()
    server.login(EMAIL_CONFIG['username'], EMAIL_CONFIG['password'])
    for email in absent_staff_emails:
        server.sendmail(EMAIL_CONFIG['sender'],
                    email,
                    'We regret to inform you that. Your son/daughter was absent today.')

    print("Email sent successfully")

    manager_email = get_manager_email()
    msg = EmailMessage()
    msg['Subject'] = 'Attendance Details'
    msg['From'] = EMAIL_CONFIG['sender']
    msg['To'] = manager_email
    msg.set_content('Attendance Report Attached')

    csv_name = generate_attendance_sheet()
    with open(csv_name,'rb') as f:
        file_data = f.read()
        file_type = 'csv'
        file_name = f.name


    msg.add_attachment(file_data,maintype='file',subtype=file_type,filename=file_name)

    with smtplib.SMTP(EMAIL_CONFIG['host'], EMAIL_CONFIG['port']) as smtp:
        smtp.starttls()
        smtp.login(EMAIL_CONFIG['username'], EMAIL_CONFIG['password'])
        smtp.send_message(msg)

    print("Report sent successfully")


if email_is_configured():
    sched = BackgroundScheduler(daemon=True)
    sched.add_job(send_mail, 'cron', day_of_week='mon-sun', hour=21, minute=30)
    sched.start()
else:
    sched = None
    print("Absence-email scheduler disabled: configure SMTP values in .env to enable it.")
