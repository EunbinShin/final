import cx_Oracle as oracle

oracle_dsn = oracle.makedsn(host='52.78.190.218'
                            ,port=1521
                            ,sid="xe")

conn = oracle.connect(dsn=oracle_dsn
                    ,user='team'
                    ,password='123456')

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
