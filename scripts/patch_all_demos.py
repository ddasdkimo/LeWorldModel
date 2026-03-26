"""
為所有 Demo 的 web_app.py 添加 /about 路由
透過在每個 FastAPI app 中掛載 about.html 作為靜態頁面
"""
import re
from pathlib import Path

BASE = Path("/Users/davidyang/code/2026/LeWorldModel/demos")

# Patch pattern: find "@app.get("/")" and add about route after it
# Also add lang toggle + about link to the header in FRONTEND HTML

DEMOS = [
    ("demo1-trajectory-prediction/src/web_app.py", 8770),
    ("demo2-voe-anomaly/src/web_demo.py", 8765),
    ("demo2-voe-anomaly/src/web_train.py", 8766),
    ("demo3-dual-vision/src/web_app.py", 8771),
    ("demo4-smart-ptz/src/web_app.py", 8772),
    ("demo5-full-system/src/web_app.py", 8780),
]

ABOUT_ROUTE = '''
@app.get("/about")
async def about_page():
    about_file = Path(__file__).parent.parent / "about.html"
    if about_file.exists():
        return HTMLResponse(about_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>About page not found</h1>")

'''

LANG_TOGGLE_CSS = '.lang-btn{position:fixed;top:12px;right:16px;z-index:999;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);color:#fff;padding:4px 14px;border-radius:999px;cursor:pointer;font-size:.8rem;font-weight:600;backdrop-filter:blur(8px)}.lang-btn:hover{background:rgba(255,255,255,.25)}.about-link{position:fixed;top:12px;right:70px;z-index:999;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);color:#94a3b8;padding:4px 14px;border-radius:999px;cursor:pointer;font-size:.8rem;text-decoration:none}.about-link:hover{color:#fff;background:rgba(255,255,255,.2)}.lang-zh{display:block}.lang-en{display:none}body.en .lang-zh{display:none}body.en .lang-en{display:block}'

LANG_TOGGLE_JS = """function toggleLang(){document.body.classList.toggle('en');var b=document.getElementById('langToggle');b.textContent=document.body.classList.contains('en')?'中文':'EN';localStorage.setItem('lewm_lang',document.body.classList.contains('en')?'en':'zh')}if(localStorage.getItem('lewm_lang')==='en'){document.body.classList.add('en');document.addEventListener('DOMContentLoaded',function(){var b=document.getElementById('langToggle');if(b)b.textContent='中文';})}"""

HEADER_BUTTONS = '<a class="about-link" href="/about" target="_blank">?</a><button class="lang-btn" id="langToggle" onclick="toggleLang()">EN</button>'

count = 0
for demo_file, port in DEMOS:
    fpath = BASE / demo_file
    if not fpath.exists():
        print(f"⚠️ {fpath} not found, skipping")
        continue

    content = fpath.read_text(encoding="utf-8")

    # 1. Add /about route if not already present
    if '/about' not in content:
        # Insert after the first @app.get("/") block
        content = content.replace(
            '@app.get("/")\nasync def index():',
            ABOUT_ROUTE + '@app.get("/")\nasync def index():',
            1
        )

    # 2. Add lang toggle CSS to <style> if not present
    if 'lang-btn' not in content:
        content = content.replace('</style>', LANG_TOGGLE_CSS + '</style>', 1)

    # 3. Add lang toggle JS before </script> or </body>
    if 'toggleLang' not in content:
        if '</script>' in content:
            # Add before last </script>
            idx = content.rfind('</script>')
            content = content[:idx] + LANG_TOGGLE_JS + '\n' + content[idx:]
        elif '</body>' in content:
            content = content.replace('</body>', '<script>' + LANG_TOGGLE_JS + '</script></body>', 1)

    # 4. Add header buttons after <body> or in header
    if 'langToggle' not in content:
        if '<body>' in content:
            content = content.replace('<body>', '<body>' + HEADER_BUTTONS, 1)
        elif '<body\\n>' in content:
            content = content.replace('<body\\n>', '<body\\n>' + HEADER_BUTTONS, 1)

    fpath.write_text(content, encoding="utf-8")
    count += 1
    print(f"✅ Patched {demo_file}")

print(f"\nDone! Patched {count} files.")
