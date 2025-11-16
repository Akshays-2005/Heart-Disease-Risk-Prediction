const express = require("express");
const path = require("path");
const { spawn } = require("child_process");
const bodyParser = require("body-parser");
const fs = require("fs");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.static(path.join(__dirname, "..", "Frontend")));
app.use(bodyParser.json({ limit: "300kb" }));

const PYTHON = process.env.PYTHON_CMD || "python"; 
const PREDICTOR_PATH = path.join(__dirname, "predictor.py");

function demoPredict(features) {
  let score = 0.05;
  const age = Number(features.age || 0);
  const trestbps = Number(features.resting_blood_pressure || features.trestbps || 0);
  const chol = Number(features.cholestoral || features.chol || 0);
  const exang = Number(features.exercise_induced_angina || features.exang || 0);
  const oldpeak = Number(features.oldpeak || 0);

  score += Math.max(0, (age - 30) / 70) * 0.25;
  score += Math.min(1, trestbps / 200) * 0.2;
  score += Math.min(1, chol / 300) * 0.15;
  if (exang === 1) score += 0.15;
  score += Math.min(1, oldpeak / 6) * 0.2;
  score = Math.max(0, Math.min(1, score));
  return { risk_pct: Math.round(score * 10000) / 100, class: score < 0.33 ? "Low" : score < 0.66 ? "Medium" : "High", raw_score: score };
}

app.post("/api/predict", (req, res) => {
  const body = req.body || {};

  if (Object.keys(body).length === 0) {
    return res.status(400).json({ error: "empty_payload" });
  }

  if (!fs.existsSync(PREDICTOR_PATH)) {
    return res.json({ warning: "predictor_missing", demo: demoPredict(body) });
  }

  const py = spawn(PYTHON, [PREDICTOR_PATH], { cwd: path.join(__dirname) });

  let stdout = "";
  let stderr = "";

  py.stdout.on("data", (data) => { stdout += data.toString(); });
  py.stderr.on("data", (data) => { stderr += data.toString(); });

  py.on("close", (code) => {
    if (stderr && stderr.trim() !== "") console.error("predictor stderr:", stderr);
    if (!stdout) {
      return res.json({ warning: "no_stdout", demo: demoPredict(body), stderr });
    }
    try {
      const parsed = JSON.parse(stdout);
      if (parsed.error) {
        return res.json({ warning: "model_error", message: parsed.error, demo: demoPredict(body) });
      }
      return res.json(parsed);
    } catch (e) {
      return res.json({ warning: "parse_error", message: e.toString(), demo: demoPredict(body), raw_stdout: stdout });
    }
  });

  py.stdin.write(JSON.stringify(body));
  py.stdin.end();

  setTimeout(() => {
    try { py.kill("SIGKILL"); } catch (e) {}
  }, 6000);
});

app.get("/health", (_, res) => res.json({ status: "ok" }));

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
