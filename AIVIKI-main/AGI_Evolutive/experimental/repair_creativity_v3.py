
import ast, re, sys
from pathlib import Path

TARGET = Path("creativity/__init__.py")

def fix_conditional_assignments(text: str) -> str:
    """Fix invalid conditional assignments like:
       (x["key"] if isinstance(x, dict) else getattr(x, "key", None)) = value
    """
    pattern = re.compile(
        r'\((\w+)\[["\'](\w+)["\']\]\s+if\s+isinstance\(\1,\s*dict\)\s*else\s*getattr\(\1,\s*["\']\2["\'],\s*None\)\)\s*=\s*(.+)'
    )
    lines = text.splitlines()
    fixed = []
    for line in lines:
        m = pattern.search(line)
        if not m:
            fixed.append(line)
            continue
        var, key, value = m.groups()
        block = [
            f"if isinstance({var}, dict):",
            f"    {var}['{key}'] = {value}",
            "else:",
            f"    setattr({var}, '{key}', {value})"
        ]
        fixed.extend(block)
    return "\n".join(fixed)

def fix_conditional_appends(text: str) -> str:
    """Fix invalid conditional method calls like:
       (x["key"] if isinstance(x, dict) else getattr(x, "key", None)).append(<value>)
    """
    pattern = re.compile(
        r'\((\w+)\[["\'](\w+)["\']\]\s+if\s+isinstance\(\1,\s*dict\)\s*else\s*getattr\(\1,\s*["\']\2["\'],\s*None\)\)\.append\((.*)\)'
    )
    lines = text.splitlines()
    fixed = []
    for line in lines:
        m = pattern.search(line)
        if not m:
            fixed.append(line)
            continue
        var, key, value = m.groups()
        block = [
            f"if isinstance({var}, dict):",
            f"    {var}['{key}'].append({value})",
            "else:",
            f"    getattr({var}, '{key}', []).append({value})"
        ]
        fixed.extend(block)
    return "\n".join(fixed)

def close_missing_parens_before_comment(line: str) -> str:
    """Close missing parentheses before inline comments"""
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
    # Close missing parens first
    out_lines = [close_missing_parens_before_comment(line) for line in txt.splitlines()]
    intermediate = "\n".join(out_lines)
    # Fix append and assignment patterns
    intermediate = fix_conditional_appends(intermediate)
    final = fix_conditional_assignments(intermediate)
    return final

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
    print("[OK] Toutes les affectations et appels conditionnels réparés et fichier valide. Sauvegarde: __init__.py.bak")

if __name__ == "__main__":
    main()
