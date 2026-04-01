import re

with open('src/articles/rag/index.html', 'r') as f:
    content = f.read()

# Fix the step badges. Currently some are:
# <div class="step-badge">IV.. Step 1</div>
# <div class="step-badge">Part I.</div>

def fix_badge(match):
    text = match.group(1)
    if '..' in text:
        text = text.replace('..', ':')
        # It's something like "IV: Step 1", maybe change to "Part IV: Step 1"
        parts = text.split(':')
        roman = parts[0].strip()
        step = parts[1].strip()
        return f'<div class="step-badge">Part {roman}: {step}</div>'
    elif 'Part' in text and text.endswith('.'):
        text = text[:-1] # Remove trailing dot
        return f'<div class="step-badge">{text}</div>'
    return match.group(0)

content = re.sub(r'<div class="step-badge">([^<]+)</div>', fix_badge, content)

# Remove abstract--secondary
content = content.replace('<p class="abstract abstract--secondary">', '<p class="abstract">')

with open('src/articles/rag/index.html', 'w') as f:
    f.write(content)

