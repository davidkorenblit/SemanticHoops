# SemanticHoops: Context & Task List / מסמך איפיון ומשימות

This document serves as the "North Star" for the SemanticHoops project, aiming for a Production-Ready tactical video retrieval system for WSC Sports.
המסמך הזה משמש כמצפן של הפרויקט, במטרה להגיע למערכת חיפוש וידאו טקטי שמוכנה לפרודקשן ומתאימה לדרישות של WSC Sports.

## 🏗️ Phase 1: Data Pipeline & Infrastructure (תשתית ודאטא)
- `[x]` **Dev Environment / הקמת סביבת פיתוח:** Set up Docker, Qdrant, and networking. (הגדרת דוקר, Qdrant ורשת)
- `[x]` **Metadata Strategy / אסטרטגיית מטא-דאטא:** Locate Play-by-Play datasets (using NSVA as Ground Truth). (מציאת דאטאסטים ל-Play-by-Play)
- `[x]` **Basic Data Ingestion / הזרקת דאטה בסיסית:** Scripts for indexing frames into the Vector DB (`upsert_data.py`). (אינדוקס פריימים ל-DB)
- `[ ]` **(Low Priority) YouTube Downloader Refactor / ריפקטורינג להורדת וידאו:** Write a Python script (`yt-dlp`) to automatically download NBA clips instead of manual downloads. (כתיבת סקריפט מבוסס פייתון להורדה אוטומטית מיוטיוב במקום עבודה ידנית)
- `[ ]` **(Medium Priority) Temporal Windows / חילוץ סגמנטים טקטיים:** Add logic to slice video exactly 10 seconds before a statistical event to reduce noise. (חיתוך וידאו ל-10 שניות בדיוק לפני האירוע הסטטיסטי)

## 🧠 Phase 2: Semantic Understanding & Search (הבנה סמנטית וחיפוש - אתגר ה-CLIP)
- `[x]` **Semantic Search Infrastructure / בניית תשתית חיפוש סמנטי:** `CLIPWrapper`, `search_semantic.py`.
- `[x]` **QA Benchmark & Reporting / בנצ'מרק ודוחות QA:** Identify exactly where vanilla CLIP fails. (הבנו בדיוק איפה CLIP נופל)
- `[x]` **CuPL (Text-Only Expansion) / ניסוי הרחבת שאילתות בטקסט:** Used Gemini to break concepts into 20 visual atoms (`sandbox_query_expansion.py`). Tested and ruled out (not accurate enough). (נוסה ונפסל - לא מדויק מספיק)
- `[ ]` **(High Priority) The CLIP Alternative / הפתרון האלטרנטיבי ל-CLIP.** Choose ONE of the following:
  - *Option A (VLM Ensembling / הפתרון של החבר):* Send a Ground Truth image to an LLM, get 25 descriptions of what it sees, and use them for queries. (לשלוח ל-LLM תמונה של מהלך ולקבל 25 תיאורים שונים)
  - *Option B (Tip-Adapter / Few-Shot Cache / הזיכרון החזותי):* Extract 5-10 correct frames of a tactic, save their vectors in Qdrant as "short-term memory", and compare new video to them. (לשמור וקטורים של פריימים נכונים ב-Qdrant ולהשוות וידאו חדש אליהם)
  - *Option C (Scene Graphs & YOLO / זיהוי אובייקטים):* Use a simple object detection model to find players, pass their distances to an LLM, and let the LLM deduce the tactic. (להשתמש במודל זיהוי אובייקטים פשוט ולתת ל-LLM להסיק טקטיקה לפי מרחקים)

## 📈 Phase 3: Engineering, Performance & Demo (הנדסה, ביצועים והדגמה ל-WSC)
- `[ ]` **Optimization & Latency / אופטימיזציה ומדידת חביון:** Ensure fast retrieval times. (וידוא זמני חיפוש מהירים)
- `[ ]` **Engineering Justification / הנדסה והצדקות:** Formulate the document explaining technological choices for the interview. (גיבוש מסמך בחירות טכנולוגיות להגשה)
- `[ ]` **UI Demo / ממשק דמו:** Build a Streamlit interface. (בניית ממשק ב-Streamlit)
- `[ ]` **End-to-End Validation / וידוא ריצה חלקה:** Ensure everything runs smoothly for the live interview demo. (וידוא שהכל רץ בצורה חלקה להדגמה)
