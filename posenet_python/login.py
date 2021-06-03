import cx_Oracle as oracle

oracle_dsn = oracle.makedsn(host='192.168.2.131'
                            ,port=1521
                            ,sid="orcl")

conn = oracle.connect(dsn=oracle_dsn
                    ,user='jyhoon94'
                    ,password='123')

def get_user_info(id, password):
    sql = """
        SELECT user_name
        FROM user_info 
        WHERE user_id=:id AND
            user_pw=:password
    """

    cursor = conn.cursor()
    cursor.execute(sql, {'id':id, 'password':password})
    
    user_name = cursor.fetchone()
    return user_name

def insert_set_and_count(user_id, set_count, rep_count):
    sql = """
        INSERT INTO squat_archive(
            user_id,
            set_count,
            rep_count,
            squat_date
        )VALUES(
            :user_id,
            :set_count,
            :rep_count,
            sysdate
        )
    """

    cursor = conn.cursor()
    cursor.execute(sql, {"user_id" : user_id, "set_count" : set_count, "rep_count" : rep_count})
    conn.commit()

def insert_knee_angle(user_id, angles, frame):
    sql = """
        INSERT INTO knee_angle(
            user_id, angles, frame, angle_date
        )VALUES(
            :user_id, :angles, :frame, sysdate
        )
    """
    cursor = conn.cursor()
    cursor.execute(sql, {"user_id" : user_id, "angles" : angles, "frame" : frame})
    conn.commit()