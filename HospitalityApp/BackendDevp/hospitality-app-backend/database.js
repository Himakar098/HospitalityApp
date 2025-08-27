const sqlite3 = require("sqlite3").verbose();

const db = new sqlite3.Database("./hospitality.db", (err) => {
    if (err) {
        console.error("Error opening database:", err);
    } else {
        console.log("Connected to SQLite database.");
        db.run(
            `CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL
            )`
        );
        db.run(
            `CREATE TABLE IF NOT EXISTS clock_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employeeId INTEGER,
                clockIn DATETIME,
                clockOut DATETIME,
                FOREIGN KEY (employeeId) REFERENCES employees(id)
            )`
        );
    }
});

module.exports = db;
