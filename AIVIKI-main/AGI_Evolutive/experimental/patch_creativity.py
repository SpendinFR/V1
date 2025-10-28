
from pathlib import Path
import re, shutil

TARGET = Path('creativity/__init__.py')

def main():
    if not TARGET.exists():
        raise SystemExit(f'Fichier introuvable : {TARGET}')
    text = TARGET.read_text(encoding='utf-8')
    original = text

    # 1️⃣ Protection: ensure all idea loops validate type before using attributes
    text = re.sub(
        r'for\s+(\w+)\s+in\s+(self\.[\w\[\]\.]+):',
        r'for \1 in \2:\n        if isinstance(\1, str):\n            try:\n                \1 = {"raw_text": \1}\n            except Exception:\n                print(f"[⚠️ Créativité] Idée ignorée car invalide: {\1}")\n                continue',
        text
    )

    # 2️⃣ Safeguard all idea[<key>] index accesses against strings
    text = re.sub(
        r'(\w+)\[\"(\w+)\"\]',
        r'(\1["\2"] if isinstance(\1, dict) else getattr(\1, "\2", None))',
        text
    )

    # 3️⃣ Add helper to convert raw string ideas before use
    helper = (
        "\n\ndef _ensure_idea_struct(idea):\n"
        "    if isinstance(idea, str):\n"
        "        return {'raw_text': idea, 'state': 'RAW'}\n"
        "    return idea\n"
    )
    text = re.sub(r'(class\s+CreativitySystem[^\n]*\n)', r'\1' + helper, text, count=1)

    # 4️⃣ Add safeguard call in generate_ideas and develop methods
    text = re.sub(
        r'(def\s+generate_ideas\s*\(self[^\)]*\)\s*:\s*\n)',
        r'\1        # Ensure idea structures are valid\n        self.idea_pool = [_ensure_idea_struct(i) for i in getattr(self, "idea_pool", [])]\n',
        text
    )
    text = re.sub(
        r'(def\s+_develop_raw_idea\s*\(self[^\)]*\)\s*:\s*\n)',
        r'\1        idea = _ensure_idea_struct(raw_idea)\n',
        text
    )

    if text == original:
        print("Aucun changement détecté - fichier déjà conforme.")
        return

    backup = TARGET.with_suffix('.py.bak')
    shutil.copy2(TARGET, backup)
    TARGET.write_text(text, encoding='utf-8')
    print(f"✅ Patch appliqué avec succès. Sauvegarde : {backup}")

if __name__ == "__main__":
    main()
