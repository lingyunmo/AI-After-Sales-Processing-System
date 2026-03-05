-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
);

-- 工单表
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    text TEXT,
    time DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- 分类记录
CREATE TABLE IF NOT EXISTS classify_log (
    ticket_id INTEGER,
    label TEXT,
    FOREIGN KEY(ticket_id) REFERENCES tickets(id)
);

-- 情感记录
CREATE TABLE IF NOT EXISTS sentiment_log (
    ticket_id INTEGER,
    sentiment TEXT,
    FOREIGN KEY(ticket_id) REFERENCES tickets(id)
);

-- 回复记录
CREATE TABLE IF NOT EXISTS reply_log (
    ticket_id INTEGER,
    reply_text TEXT,
    FOREIGN KEY(ticket_id) REFERENCES tickets(id)
);
