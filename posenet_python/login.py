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