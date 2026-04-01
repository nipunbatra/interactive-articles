import os
import re

for article in ['autograd', 'multivariate-normal', 'text-diffusion']:
    html_file = f'src/articles/{article}/index.html'
    css_file = f'src/articles/{article}/styles.css'
    js_file = f'src/articles/{article}/app.js'
    
    # HTML
    if os.path.exists(html_file):
        with open(html_file, 'r') as f:
            content = f.read()
        content = re.sub(r'<nav class="toc".*?</nav>', '', content, flags=re.DOTALL)
        with open(html_file, 'w') as f:
            f.write(content)
            
    # CSS
    if os.path.exists(css_file):
        with open(css_file, 'r') as f:
            content = f.read()
        content = re.sub(r'/\*\s*─── TOC ───\s*\*/.*?/\*\s*─── Sections ───\s*\*/', '/* ─── Sections ─── */', content, flags=re.DOTALL)
        content = re.sub(r'\.toc ul \{ flex-direction: column; \}\n\s*', '', content)
        with open(css_file, 'w') as f:
            f.write(content)
            
    # JS
    if os.path.exists(js_file):
        with open(js_file, 'r') as f:
            content = f.read()
        content = re.sub(r'/\*\s*═══════ TOC active tracking ═══════\s*\*/.*?function setupToc\(\) \{.*?\n\s*\}', '', content, flags=re.DOTALL)
        content = re.sub(r'setupToc\(\);\n\s*', '', content)
        with open(js_file, 'w') as f:
            f.write(content)

