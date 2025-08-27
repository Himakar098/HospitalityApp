const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");

const app = express();
app.use(bodyParser.json());
app.use(cors());

const PORT = 3000;

// Test route
app.get("/", (req, res) => {
    res.send("API is working!");
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});

const db = require("./database");

// Clock-in
app.post("/clock-in", (req, res) => {
    const { employeeId } = req.body;
    const clockInTime = new Date();

    db.run(
        `INSERT INTO clock_records (employeeId, clockIn) VALUES (?, ?)`,
        [employeeId, clockInTime],
        function (err) {
            if (err) {
                res.status(500).json({ error: err.message });
            } else {
                res.status(201).json({ message: "Clocked in successfully", id: this.lastID });
            }
        }
    );
});

// Clock-out
app.post("/clock-out", (req, res) => {
    const { recordId } = req.body;
    const clockOutTime = new Date();

    db.run(
        `UPDATE clock_records SET clockOut = ? WHERE id = ?`,
        [clockOutTime, recordId],
        function (err) {
            if (err) {
                res.status(500).json({ error: err.message });
            } else {
                res.status(200).json({ message: "Clocked out successfully" });
            }
        }
    );
});
