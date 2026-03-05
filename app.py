from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os
from models.predictor import predict_classifier, predict_sentiment
from models.reply_llm_local import stream_generate_reply

app = Flask(__name__)
app.secret_key = '_key'  # 用于加密 session

DATABASE = 'identifier.sqlite'

# 数据库连接
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# 初始化数据库（仅首次运行时使用）
def init_db():
    with app.app_context():
        db = get_db()
        with open('identifier.sqlite', 'r', encoding='utf-8') as f:
            db.executescript(f.read())
        db.commit()

# 首页 + 提交表单
@app.route("/", methods=["GET"])
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", username=session["username"])

@app.route("/submit", methods=["POST"])
def submit():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user_input = request.form["user_input"]
    category = predict_classifier(user_input)
    sentiment = predict_sentiment(user_input)
    reply = stream_generate_reply(user_input, category, sentiment)

    db = get_db()
    cursor = db.cursor()

    # 插入工单记录
    cursor.execute("INSERT INTO tickets(user_id, text) VALUES (?, ?)", (session["user_id"], user_input))
    ticket_id = cursor.lastrowid

    # 插入各模块结果
    cursor.execute("INSERT INTO classify_log(ticket_id, label) VALUES (?, ?)", (ticket_id, category))
    cursor.execute("INSERT INTO sentiment_log(ticket_id, sentiment) VALUES (?, ?)", (ticket_id, sentiment))
    cursor.execute("INSERT INTO reply_log(ticket_id, reply_text) VALUES (?, ?)", (ticket_id, reply))

    db.commit()

    return render_template("index.html", result={
        "text": user_input,
        "category": category,
        "sentiment": sentiment,
        "reply": reply
    }, username=session["username"])

# 注册
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        db = get_db()
        try:
            db.execute("INSERT INTO users(username, password) VALUES (?, ?)", (username, password))
            db.commit()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            return render_template("register.html", error="用户名已存在")
    return render_template("register.html")

# 登录
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="用户名或密码错误")
    return render_template("login.html")

# 退出
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# 管理页面：显示所有工单
@app.route("/admin")
def admin():
    if "user_id" not in session:
        return redirect(url_for("login"))
    db = get_db()
    tickets = db.execute("""
        SELECT t.*, u.username FROM tickets t
        JOIN users u ON t.user_id = u.id
        ORDER BY t.time DESC
    """).fetchall()
    classify_logs = db.execute("SELECT * FROM classify_log").fetchall()
    sentiment_logs = db.execute("SELECT * FROM sentiment_log").fetchall()
    reply_logs = db.execute("SELECT * FROM reply_log").fetchall()

    return render_template(
        "admin.html",
        tickets=tickets,
        classify_logs=classify_logs,
        sentiment_logs=sentiment_logs,
        reply_logs=reply_logs,
        username=session["username"]
    )


# 主函数
if __name__ == "__main__":
    if not os.path.exists(DATABASE):
        print("🔧 检测到未初始化数据库，正在初始化...")
        init_db()
        print("✅ 数据库初始化完成！")

    app.run(debug=True)
