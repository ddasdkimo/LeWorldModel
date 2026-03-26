"""
為所有 Demo 的 web_app.py 加入 /about 路由
只需執行一次，之後各 Demo 會自動提供 /about 頁面
"""
import sys
sys.path.insert(0, "/Users/davidyang/code/2026/LeWorldModel/shared")
from about_pages import ABOUT_PAGES, generate_about_html, get_lang_toggle_css, get_lang_toggle_js, get_lang_toggle_button

# Generate standalone about HTML files for each demo
from pathlib import Path

BASE = Path("/Users/davidyang/code/2026/LeWorldModel/demos")

DEMOS = {
    "demo1-trajectory-prediction": "demo1",
    "demo2-voe-anomaly": "demo2",
    "demo3-dual-vision": "demo3",
    "demo4-smart-ptz": "demo4",
    "demo5-full-system": "demo5",
}

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
:root {{ --bg:#0f172a; --card:#1e293b; --accent:#3b82f6; --text:#e2e8f0; --dim:#64748b; --border:#334155; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }}
a {{ color:var(--accent); }}
.back {{ display:inline-block; padding:8px 16px; margin:1rem 2rem; color:var(--dim); text-decoration:none; font-size:0.9rem; }}
.back:hover {{ color:var(--text); }}
{lang_css}
{about_css}
</style>
</head>
<body>
{lang_btn}
<a class="back" href="/">← 返回 <span class="lang-en" style="display:none">← Back</span></a>
{about_html}
<script>{lang_js}</script>
</body>
</html>"""

for demo_dir, demo_key in DEMOS.items():
    info = ABOUT_PAGES.get(demo_key, {})
    title = info.get("title_zh", demo_key)

    about_css = """
    .about-content { max-width:800px; margin:0 auto; padding:1rem 2rem 3rem; }
    .about-content h1 { font-size:1.8rem; margin-bottom:1rem; }
    .about-content h2 { font-size:1.3rem; margin:1.5rem 0 0.8rem; color:var(--accent); }
    .about-content h3 { font-size:1.05rem; margin:1.2rem 0 0.5rem; }
    .about-content p { margin-bottom:0.8rem; line-height:1.7; }
    .about-content ul, .about-content ol { margin:0.5rem 0 1rem 1.5rem; }
    .about-content li { margin-bottom:0.3rem; line-height:1.6; }
    .about-content table { width:100%; border-collapse:collapse; margin:0.5rem 0 1rem; }
    .about-content th, .about-content td { padding:8px 12px; border:1px solid var(--border); text-align:left; font-size:0.9rem; }
    .about-content th { background:rgba(255,255,255,0.05); }
    .about-content code { background:rgba(255,255,255,0.1); padding:2px 6px; border-radius:4px; font-size:0.85rem; }
    """

    about_content = f"""
    <div class="about-content">
        <div class="lang-zh">
            <h1>{info.get('title_zh', '')}</h1>
            {info.get('content_zh', '')}
        </div>
        <div class="lang-en">
            <h1>{info.get('title_en', '')}</h1>
            {info.get('content_en', '')}
        </div>
    </div>
    """

    html = TEMPLATE.format(
        title=title,
        lang_css=get_lang_toggle_css(),
        about_css=about_css,
        lang_btn=get_lang_toggle_button(),
        about_html=about_content,
        lang_js=get_lang_toggle_js(),
    )

    out_path = BASE / demo_dir / "about.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"✅ {out_path}")

# Also generate training manager about
info = ABOUT_PAGES["train"]
html = TEMPLATE.format(
    title=info["title_zh"],
    lang_css=get_lang_toggle_css(),
    about_css=about_css,
    lang_btn=get_lang_toggle_button(),
    about_html=f"""
    <div class="about-content">
        <div class="lang-zh"><h1>{info['title_zh']}</h1>{info['content_zh']}</div>
        <div class="lang-en"><h1>{info['title_en']}</h1>{info['content_en']}</div>
    </div>
    """,
    lang_js=get_lang_toggle_js(),
)
out = BASE / "demo2-voe-anomaly" / "about_train.html"
out.write_text(html, encoding="utf-8")
print(f"✅ {out}")

print("\nDone! About pages generated for all demos.")
