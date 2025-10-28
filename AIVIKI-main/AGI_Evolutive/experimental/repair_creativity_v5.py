
import ast, re, sys
from pathlib import Path

TARGET = Path("creativity/__init__.py")

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

def fix_self_paren(line: str) -> str:
    return re.sub(r'\bself\.\s*\(', '(', line)

# Match conditional .append({payload}) starting on one line
APP_START = re.compile(
    r'^(?P<indent>\s*)\((?P<var>\w+)\[(?P<q1>["\'])(?P<key>\w+)(?P=q1)\]\s+'
    r'if\s+isinstance\(\s*(?P=var)\s*,\s*dict\s*\)\s*'
    r'else\s*getattr\(\s*(?P=var)\s*,\s*(?P<q2>["\'])(?P=key)(?P=q2)\s*,\s*None\)\)\.append\(\s*\{\s*$'
)

# Match conditional single-line assignment pattern
ASSIGN_LINE = re.compile(
    r'^\s*\((?P<var>\w+)\[(?P<q1>["\'])(?P<key>\w+)(?P=q1)\]\s+'
    r'if\s+isinstance\(\s*(?P=var)\s*,\s*dict\s*\)\s*'
    r'else\s*getattr\(\s*(?P=var)\s*,\s*(?P<q2>["\'])(?P=key)(?P=q2)\s*,\s*None\)\)\s*=\s*(?P<rhs>.+)$'
)

def process_text(txt: str) -> str:
    # First pass: simple per-line fixes
    lines = [fix_self_paren(close_missing_parens_before_comment(ln)) for ln in txt.splitlines()]
    out = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        # Handle multiline conditional append that starts with `.append({` on the same line
        m = APP_START.match(line)
        if m:
            indent = m.group('indent')
            var = m.group('var')
            key = m.group('key')

            # Collect lines until we hit a line that ends a block with `})`
            payload_lines = []
            j = i + 1
            brace_depth = 1  # we have consumed first `{`
            paren_depth = 1  # inside append(
            while j < n:
                L = lines[j]
                payload_lines.append(L)
                # count braces and parentheses
                brace_depth += L.count('{') - L.count('}')
                paren_depth += L.count('(') - L.count(')')
                # Check for the typical terminator `})` at end of a line (after accounting for depths)
                if brace_depth == 0 and paren_depth == 0 and L.strip().endswith('})'):
                    # remove the final '})' from the last payload line
                    payload_lines[-1] = L.rstrip()[:-2]
                    break
                j += 1

            payload = "\n".join(payload_lines).rstrip()
            block = (
                f"{indent}if isinstance({var}, dict):\n"
                f"{indent}    {var}['{key}'].append({{\n{payload}\n{indent}    }})\n"
                f"{indent}else:\n"
                f"{indent}    getattr({var}, '{key}', []).append({{\n{payload}\n{indent}    }})"
            )
            out.append(block)
            i = j + 1
            continue

        # Handle single-line conditional assignment
        m2 = ASSIGN_LINE.match(line)
        if m2:
            var = m2.group('var'); key = m2.group('key'); rhs = m2.group('rhs').rstrip()
            indent = line[:len(line)-len(line.lstrip())]
            block = (
                f"{indent}if isinstance({var}, dict):\n"
                f"{indent}    {var}['{key}'] = {rhs}\n"
                f"{indent}else:\n"
                f"{indent}    setattr({var}, '{key}', {rhs})"
            )
            out.append(block)
            i += 1
            continue

        out.append(line)
        i += 1

    return "\n".join(out) + ("\n" if not txt.endswith("\n") else "")

def main():
    TARGET = Path("AGI_Evolutive/creativity/__init__.py")
    if not TARGET.exists():
        print(f"[ERR] Fichier introuvable: {TARGET}")
        sys.exit(1)

    src = TARGET.read_text(encoding="utf-8")
    TARGET.with_suffix(".py.bak").write_text(src, encoding="utf-8")

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
    print("[OK] Réparation v5 terminée. Fichier valide. Sauvegarde: __init__.py.bak")

if __name__ == "__main__":
    main()
