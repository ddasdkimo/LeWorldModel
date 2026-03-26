/**
 * LeWM Demos - 共用多語系模組
 * 用法: 在 HTML 中用 data-i18n="key" 標記需要翻譯的元素
 *       呼叫 setLang('zh') 或 setLang('en') 切換語言
 */

const LANGS = {
  // ===== 共用 =====
  "common": {
    "zh": {
      "lang_toggle": "EN",
      "record": "錄製",
      "train": "訓練",
      "detect": "偵測",
      "setup": "設定",
      "models": "模型",
      "about": "關於",
      "start": "開始",
      "stop": "停止",
      "preview": "預覽",
      "save": "儲存",
      "loading": "載入中...",
      "rtsp_url": "RTSP 網址",
      "duration_sec": "錄製秒數",
      "fps": "幀率",
      "dataset_name": "資料集名稱",
      "model_name": "模型名稱",
      "epochs": "訓練輪數",
      "batch_size": "批次大小",
      "learning_rate": "學習率",
      "checkpoint": "模型檔",
      "start_recording": "開始錄製",
      "start_training": "開始訓練",
      "start_detection": "開始偵測",
      "stop_detection": "停止偵測",
      "no_datasets": "尚無資料集，請先錄製。",
      "no_models": "尚無模型，請先訓練。",
      "device": "運算裝置",
      "inference_time": "推論時間",
      "surprise": "異常指數",
      "threshold": "閾值",
      "status": "狀態",
      "normal": "正常",
      "anomaly": "異常",
      "persons": "偵測人數",
      "github": "GitHub 原始碼",
    },
    "en": {
      "lang_toggle": "中",
      "record": "Record",
      "train": "Train",
      "detect": "Detect",
      "setup": "Setup",
      "models": "Models",
      "about": "About",
      "start": "Start",
      "stop": "Stop",
      "preview": "Preview",
      "save": "Save",
      "loading": "Loading...",
      "rtsp_url": "RTSP URL",
      "duration_sec": "Duration (sec)",
      "fps": "FPS",
      "dataset_name": "Dataset Name",
      "model_name": "Model Name",
      "epochs": "Epochs",
      "batch_size": "Batch Size",
      "learning_rate": "Learning Rate",
      "checkpoint": "Checkpoint",
      "start_recording": "Start Recording",
      "start_training": "Start Training",
      "start_detection": "Start Detection",
      "stop_detection": "Stop Detection",
      "no_datasets": "No datasets yet. Record one first.",
      "no_models": "No models yet. Train one first.",
      "device": "Device",
      "inference_time": "Inference Time",
      "surprise": "Surprise",
      "threshold": "Threshold",
      "status": "Status",
      "normal": "Normal",
      "anomaly": "ANOMALY",
      "persons": "Persons",
      "github": "GitHub Source",
    }
  }
};

let currentLang = localStorage.getItem('lewm_lang') || 'zh';
let demoTranslations = {};

function registerTranslations(translations) {
  demoTranslations = translations;
}

function t(key) {
  // Check demo-specific first, then common
  if (demoTranslations[currentLang] && demoTranslations[currentLang][key]) {
    return demoTranslations[currentLang][key];
  }
  if (LANGS.common[currentLang] && LANGS.common[currentLang][key]) {
    return LANGS.common[currentLang][key];
  }
  return key;
}

function setLang(lang) {
  currentLang = lang;
  localStorage.setItem('lewm_lang', lang);
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    const val = t(key);
    if (el.tagName === 'INPUT' && el.type !== 'number') {
      el.placeholder = val;
    } else if (el.tagName === 'OPTION') {
      // skip
    } else {
      el.textContent = val;
    }
  });
  document.querySelectorAll('[data-i18n-html]').forEach(el => {
    el.innerHTML = t(el.getAttribute('data-i18n-html'));
  });
  // Update toggle button
  const toggleBtn = document.getElementById('langToggle');
  if (toggleBtn) toggleBtn.textContent = t('lang_toggle');
}

function toggleLang() {
  setLang(currentLang === 'zh' ? 'en' : 'zh');
}

// Auto-apply on load
document.addEventListener('DOMContentLoaded', () => setLang(currentLang));
