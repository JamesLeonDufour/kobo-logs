# 🕵️ KoboToolbox Logs Viewer & Analyzer

A **Streamlit** app to fetch, explore, visualize, and export **KoboToolbox logs** — including **Audit Logs**, **Project History Logs**, and **Access Logs** — via the KoboToolbox REST API.

---

## 🚀 Features

- 🔐 Connect to any KoboToolbox server using an API Token  
- 📅 Filter logs by date range (presets or custom)  
- ⚙️ Filter by **action**, **log type**, **username**, **asset UID**, and **metadata fields**  
- 🔎 Advanced query builder for complex searches  
- 📊 Interactive visualizations with **Plotly**:
  - Logs over time  
  - Action distribution  
  - Top active users  
- 📥 Export results as:
  - **CSV**
  - **Excel**
  - **Raw JSON**

---

## 🧰 Requirements

- Python 3.9+
- Streamlit
- Pandas
- Requests
- Plotly
- XlsxWriter

Install dependencies:


```bash
pip install streamlit pandas requests plotly xlsxwriter
```

---

## ▶️ Usage

1. Download this repository:

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser at [http://localhost:8501](http://localhost:8501)

---

## ⚙️ Configuration

- **Server URL:** Enter your KoboToolbox base domain (e.g. `eu.kobotoolbox.org` or `kf.kobotoolbox.org`)
- **API Token:** Found in your Kobo account settings
- **Timezone:** Defaults to `Europe/Brussels` (editable in code)

---

## 📈 Example Visuals

- Line chart of log volume over time  
- Bar charts for action frequency and top users  
- Interactive tables with search and filters  

---

## 🧾 Exports

All fetched logs can be downloaded in:
- `CSV` — clean, processed table view  
- `Excel` — with formatted headers and autosized columns  
- `JSON` — raw API response

---

## 🪪 License

MIT License — free to use, modify, and distribute.
