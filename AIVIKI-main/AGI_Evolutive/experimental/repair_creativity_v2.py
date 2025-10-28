
import ast, re, sys
from pathlib import Path

TARGET = Path("creativity/__init__.py")

def fix_conditional_assignments(text: str) -> str:
    # Detect patterns like: (x["y"] if isinstance(x, dict) else getattr(x, "y", None)) = val
    pattern = re.compile(
        r'\((\w+)\[["\'](\w+)["\']\]\s+if\s+isinstance\(\1,\s*dict\)\s*else\s*getattr\(\1,\s*["\']\2["\'],\s*None\)\)\s*=\s*(.+)'
    )
    fixed_lines = []
    for line in text.splitlines():
        m = pattern.search(line)
        if not m:
            fixed_lines.append(line)
            continue
        var, key, value = m.groups()
        replacement = (
            f"if isinstance({var}, dict):\n"
            f"    {var}['{key}'] = {value}\n"
            f"else:\n"
            f"    setattr({var}, '{key}', {value})"
        )
        fixed_lines.append(replacement)
    return "\n".join(fixed_lines)

def close_missing_parens_before_comment(line: str) -> str:
    if '#' not in line:
        return line
    code, comment = line.split('#', 1)
    opens = code.count('(')
    closes = code.count(')')
    if opens > closes:
        code = code.rstrip() + (')' * (opens - closes))
        return code + '  #' + comment
    return line

def process_text(txt: str) -> str:
    out_lines = []
    for line in txt.splitlines():
        line = close_missing_parens_before_comment(line)
        out_lines.append(line)
    intermediate = "\n".join(out_lines)
    return fix_conditional_assignments(intermediate)

def main():
    if not TARGET.exists():
        print(f"[ERR] Fichier introuvable: {TARGET}")
        sys.exit(1)
    src = TARGET.read_text(encoding="utf-8")
    bak = TARGET.with_suffix(".py.bak")
    bak.write_text(src, encoding="utf-8")

    fixed = process_text(src)

    try:
        ast.parse(fixed)
    except SyntaxError as e:
        print(f"[ERR] SyntaxError line {e.lineno}, col {e.offset}: {e.msg}")
        lines = fixed.splitlines()
        if 1 <= (e.lineno or 0) <= len(lines):
            print(f"[CTX] {e.lineno}: {lines[e.lineno-1]}")
        TARGET.write_text(fixed, encoding="utf-8")
        sys.exit(2)

    TARGET.write_text(fixed, encoding="utf-8")
    print("[OK] Toutes les affectations conditionnelles réparées et fichier valide. Sauvegarde: __init__.py.bak")

if __name__ == "__main__":
    main()
