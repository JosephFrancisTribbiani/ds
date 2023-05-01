import psycopg2
from functools import wraps
from config import DATABASE_URL


def get_connection(func):
    """
    Декоратор для создания подключения к базе и гарантированного его закрытия
    :param func: функция, которой требуется подключиться к базе
    :return:
    """

    @wraps(func)
    def inner(*args, **kwargs):
        with psycopg2.connect(DATABASE_URL) as conn, conn.cursor() as cur:
            res = func(*args, conn=conn, cur=cur, **kwargs)
        return res

    return inner


@get_connection
def init_db(conn, cur, force: bool = False, hidden_size: int = 128) -> None:
    """
    Функция формирует систему таблиц базы данных
    :param conn: connection к базе
    :param cur: курсор подключения к базе
    :param force: True если необходимо пересоздать базу данных (все данные будут удалены)
    :return: nothing
    """

    if force:
        cur.execute('DROP TABLE IF EXISTS encodings, students CASCADE;')

    query_students = """
    CREATE TABLE IF NOT EXISTS students
    (
        id INTEGER PRIMARY KEY,
        firstname TEXT NOT NULL,
        secondname TEXT NOT NULL,
        standing CHAR NOT NULL,
        last_attendance_time TIMESTAMP WITH TIME ZONE,
        major TEXT NOT NULL,
        starting_year INTEGER NOT NULL,
        total_attendance INTEGER NOT NULL DEFAULT 0
    );
    
    COMMENT ON TABLE students
    IS 'Таблица с метаданными студентов';

    COMMENT ON COLUMN students.id
    IS 'Уникальный ID студента';

    COMMENT ON COLUMN students.firstname
    IS 'Имя';

    COMMENT ON COLUMN students.secondname
    IS 'Фамилия';

    COMMENT ON COLUMN students.standing
    IS 'Статус';

    COMMENT ON COLUMN students.last_attendance_time
    IS 'Дата и время последнего посещения (с timezone)';

    COMMENT ON COLUMN students.major
    IS 'Специализация';

    COMMENT ON COLUMN students.starting_year
    IS 'Год начала обучения';

    COMMENT ON COLUMN students.total_attendance
    IS 'Общее количество посещений';"""
    cur.execute(query_students)

    sub_query = ",\n\t".join(map(lambda idx: '"%s" DOUBLE PRECISION NOT NULL' % idx, range(1, 1 + hidden_size)))
    query_embeddings = """
    CREATE TABLE IF NOT EXISTS embeddings
    (   
        emb_id INTEGER PRIMARY KEY,
        student_id INTEGER NOT NULL REFERENCES students (id),
        writetime TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
        %s
    );

    COMMENT ON TABLE embeddings
    IS 'Таблица с эмбеддингами лиц студентов';
    
    COMMENT ON COLUMN embeddings.emb_id
    IS 'Уникальный ID эмбеддинга';

    COMMENT ON COLUMN embeddings.student_id
    IS 'Уникальный ID студента';

    COMMENT ON COLUMN embeddings.writetime
    IS 'Дата и время записи эмбеддинга (с timezone)';
    """ % sub_query
    cur.execute(query_embeddings)
    
    conn.commit()
    return


@get_connection
def check_student(id: int, conn, cur) -> bool:
    """
    Функция проверки существования уникального ID студента в DB
    :param id: ID студента, который необходимо проверить
    """
    query = """
    SELECT EXISTS (SELECT 1 FROM students WHERE id = %s);
    """ % id
    cur.execute(query)
    return cur.fetchone()[0]


@get_connection
def get_next_id(conn, cur) -> int:
    """
    Функция для получения следующего свободного ID студента
    """
    query = """
    SELECT max(id) FROM students;
    """
    cur.execute(query)
    max_id = cur.fetchone()[0]
    if max_id is None:
        return 0
    return max_id + 1


@get_connection
def save_student(id: int, firstname: str, secondname: str, standing: str, 
                 major: str, starting_year: int, conn, cur) -> None:
    query = """
    INSERT INTO students (id, firstname, secondname, standing, major, starting_year)
    VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING;
    """
    cur.execute(query, (id, firstname, secondname, standing, major, starting_year))
    conn.commit()
    return
