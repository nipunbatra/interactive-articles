import re

with open('src/articles/rag/index.html', 'r') as f:
    content = f.read()

# Replace <section id="..."> with <section class="step-section" id="...">
content = re.sub(r'<section id="([^"]+)">', r'<section class="step-section" id="\1">', content)

# Replace <h2><span class="num">I.</span> The problem: frozen knowledge</h2>
# with <div class="step-badge">I. The problem</div> (Wait, the user wants: <div class="step-badge">Step 1</div> <h2>...</h2>)
# But there are roman numerals.
# Wait, for p-values, we just had `<div class="step-badge">Step 1</div><h2>Choose a scenario</h2>`.
# In RAG, there is a mix of `<h2><span class="num">I.</span> Title</h2>` and `<h2><span class="num">IV.</span> <span class="step-tag">Step 1</span> Title</h2>`.

# Let's extract the Roman Numeral into the step-badge, e.g., `<div class="step-badge">Part I</div> <h2>The problem...</h2>`.
def replace_h2(match):
    roman = match.group(1)
    rest = match.group(2)
    # If there's a step-tag like <span class="step-tag">Step 1</span>
    step_match = re.match(r'\s*<span class="step-tag">([^<]+)</span>\s*(.*)', rest)
    if step_match:
        step_text = step_match.group(1)
        h2_text = step_match.group(2)
        return f'<div class="step-badge">{roman}. {step_text}</div>\n        <h2>{h2_text}</h2>'
    else:
        return f'<div class="step-badge">Part {roman}</div>\n        <h2>{rest.strip()}</h2>'

content = re.sub(r'<h2><span class="num">([^<]+)</span>(.*?)</h2>', replace_h2, content)

with open('src/articles/rag/index.html', 'w') as f:
    f.write(content)

