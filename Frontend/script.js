(() => {
  const PREFIX = "hdp-"; 

  const fieldIds = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_blood_pressure",
    "cholestoral",
    "fasting_blood_sugar",
    "rest_ecg",
    "thalach",
    "exercise_induced_angina",
    "oldpeak",
    "slope",
    "vessels_colored_by_flourosopy",
    "thalassemia"
  ];

  const form = document.getElementById("predictForm");
  const predictBtn = document.getElementById("predictBtn");
  const demoBtn = document.getElementById("demoBtn");
  const resetBtn = document.getElementById("resetBtn");
  const clearSavedBtn = document.getElementById("clearSavedBtn");
  const resultBox = document.getElementById("predictionResult");

  function showResult(text, isError = false) {
    if (!resultBox) {
      console.warn("#predictionResult not found");
      return;
    }
    resultBox.classList.remove("hidden");
    resultBox.style.color = isError ? "crimson" : "";
    resultBox.textContent = text;
  }

  function hideResult() {
    if (!resultBox) return;
    resultBox.classList.add("hidden");
    resultBox.textContent = "";
  }

  function byId(id) { return document.getElementById(id); }

  fieldIds.forEach(id => {
    const el = byId(id);
    if (!el) return;
    el.addEventListener("input", () => {
      localStorage.setItem(PREFIX + id, el.value);
    });
  });

  window.addEventListener("DOMContentLoaded", () => {
    fieldIds.forEach(id => {
      const saved = localStorage.getItem(PREFIX + id);
      const el = byId(id);
      if (saved !== null && el) el.value = saved;
    });
  });

  demoBtn && demoBtn.addEventListener("click", () => {
    const demo = {
      age: 63,
      sex: 1,
      chest_pain_type: 3,
      resting_blood_pressure: 145,
      cholestoral: 233,
      fasting_blood_sugar: 0,
      rest_ecg: 1,
      thalach: 150,
      exercise_induced_angina: 0,
      oldpeak: 2.3,
      slope: 1,
      vessels_colored_by_flourosopy: 0,
      thalassemia: 2
    };
    Object.entries(demo).forEach(([k,v]) => {
      const el = byId(k);
      if (el) { el.value = v; localStorage.setItem(PREFIX + k, v); }
    });
    showResult("Demo data loaded.");
    setTimeout(() => hideResult(), 1400);
  });

  resetBtn && resetBtn.addEventListener("click", () => {
    form.reset();
    fieldIds.forEach(id => localStorage.removeItem(PREFIX + id));
    hideResult();
  });

  clearSavedBtn && clearSavedBtn.addEventListener("click", () => {
    fieldIds.forEach(id => {
      localStorage.removeItem(PREFIX + id);
      const el = byId(id);
      if (el) el.value = "";
    });
    showResult("Saved inputs cleared.");
    setTimeout(() => hideResult(), 1200);
  });

  function collectFormData() {
    const data = {};
    fieldIds.forEach(id => {
      const el = byId(id);
      if (!el) {
        data[id] = 0;
        console.warn(`Missing element #${id}`);
      } else {
        const raw = el.value;
        data[id] = raw === "" ? 0 : (raw.includes(".") ? Number(parseFloat(raw)) : Number(parseInt(raw)));
        if (Number.isNaN(data[id])) data[id] = 0;
      }
    });
    return data;
  }

  function localDemoPredict(features) {
    let score = 0.05;
    score += Math.max(0, (Number(features.age) - 30) / 70) * 0.25;
    score += Math.min(1, Number(features.resting_blood_pressure) / 200) * 0.2;
    score += Math.min(1, Number(features.cholestoral) / 300) * 0.15;
    if (Number(features.exercise_induced_angina) === 1) score += 0.15;
    score += Math.min(1, Number(features.oldpeak) / 6) * 0.2;
    score = Math.max(0, Math.min(1, score));
    return { risk_pct: Math.round(score * 10000)/100, class: score < 0.33 ? "Low" : score < 0.66 ? "Medium" : "High", raw_score: score };
  }

  async function predict() {
    const data = collectFormData();
    predictBtn.disabled = true;
    predictBtn.innerText = "Predicting...";
    showResult("Predicting...");

    if (location.protocol === "file:") {
      alert("Open the page via the server (http://localhost:3000). API calls won't work with file://");
      const demo = localDemoPredict(data);
      showResult(`Demo result — Risk: ${demo.risk_pct}% (${demo.class})`);
      predictBtn.disabled = false; predictBtn.innerText = "Get Prediction";
      return;
    }

    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });

      if (!resp.ok) {
        const t = await resp.text();
        console.error("Server response not OK:", resp.status, t);
        const demo = localDemoPredict(data);
        showResult(`Server error ${resp.status}. Showing demo: ${demo.risk_pct}% (${demo.class})`, true);
        predictBtn.disabled = false; predictBtn.innerText = "Get Prediction";
        return;
      }

      const json = await resp.json();
      console.log("Prediction response:", json);

      if (json.demo) {
        showResult(`Demo result — Risk: ${json.demo.risk_pct}% (${json.demo.class})`);
      } else if (json.risk_pct !== undefined) {
        showResult(`Risk: ${json.risk_pct}% (${json.class})`);
      } else {
        showResult(`Unexpected response. See console.`, true);
        console.warn(json);
      }
    } catch (err) {
      console.error("Fetch/Network error:", err);
      const demo = localDemoPredict(data);
      showResult(`Network error. Demo: ${demo.risk_pct}% (${demo.class})`, true);
    } finally {
      predictBtn.disabled = false;
      predictBtn.innerText = "Get Prediction";
    }
  }

  predictBtn && predictBtn.addEventListener("click", predict);

  form && form.addEventListener("submit", (e) => { e.preventDefault(); predict(); });

  hideResult();

})(); 
