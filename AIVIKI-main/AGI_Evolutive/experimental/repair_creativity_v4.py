
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
    # Fix invalid self.<something> from earlier broken patches
    return re.sub(r'\bself\.\s*\(', '(', line)

# ---------- Single-line conditional assignment fix ----------
_cond_assign = re.compile(
    r'\((?P<var>\w+)\[(?P<q1>["\'])(?P<key>\w+)(?P=q1)\]\s+'
    r'if\s+isinstance\(\s*(?P=var)\s*,\s*dict\s*\)\s*'
    r'else\s*getattr\(\s*(?P=var)\s*,\s*(?P<q2>["\'])(?P=key)(?P=q2)\s*,\s*None\)\)\s*=\s*(?P<rhs>.+)$'
)

def try_fix_conditional_assignment(line: str, indent: str):
    m = _cond_assign.match(line.lstrip())
    if not m:
        return None
    var = m.group('var')
    key = m.group('key')
    rhs = m.group('rhs').rstrip()
    ind = indent
    block = (
        f"{ind}if isinstance({var}, dict):\n"
        f"{ind}    {var}['{key}'] = {rhs}\n"
        f"{ind}else:\n"
        f"{ind}    setattr({var}, '{key}', {rhs})"
    )
    return block

# ---------- Multiline conditional .append({payload}) fix ----------
_append_start = re.compile(
    r'^(?P<indent>\s*)\((?P<var>\w+)\[(?P<q1>["\'])(?P<key>\w+)(?P=q1)\]\s+'
    r'if\s+isinstance\(\s*(?P=var)\s*,\s*dict\s*\)\s*'
    r'else\s*getattr\(\s*(?P=var)\s*,\s*(?P<q2>["\'])(?P=key)(?P=q2)\s*,\s*None\)\)\.append\(\s*$'
)

def fix_multiline_append(lines, i):
    """If line i starts a conditional-append(<payload>), capture payload until matching ')'
       Returns (new_lines_list, new_index_after_block) or (None, i) if not matched.
    """
    m = _append_start.match(lines[i])
    if not m:
        return None, i
    indent = m.group('indent')
    var = m.group('var')
    key = m.group('key')

    # Capture payload across multiple lines until closing ')'
    payload_lines = []
    # Inside append(<payload>) we track parentheses depth
    depth = 1  # we are after '('
    j = i + 1
    while j < len(lines):
        line = lines[j]
        # Count parentheses in this line, but ignore those in strings to keep it simple
        payload_lines.append(line)
        # crude balance tracking
        depth += line.count('(')
        depth -= line.count(')')
        if depth <= 0:
            break
        j += 1

    # Remove the last line's trailing ')' to get pure payload
    if depth <= 0 and payload_lines:
        last = payload_lines[-1]
        pos = last.rfind(')')
        if pos != -1:
            payload_lines[-1] = last[:pos]
    payload = "\n".join(payload_lines).rstrip()

    # Build replacement block
    block = (
        f"{indent}if isinstance({var}, dict):\n"
        f"{indent}    {var}['{key}'].append(\n{payload}\n{indent}    )\n"
        f"{indent}else:\n"
        f"{indent}    getattr({var}, '{key}', []).append(\n{payload}\n{indent}    )"
    )
    return block, j  # j is the index of the line that had the closing ')'

def process_text(txt: str) -> str:
    # First pass: line-wise fixes (parens and self.()
    lines = [fix_self_paren(close_missing_parens_before_comment(ln)) for ln in txt.splitlines()]

    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        indent = line[:len(line)-len(stripped)]

        # 1) Try conditional assignment single-line
        fixed_assign = try_fix_conditional_assignment(line, indent)
        if fixed_assign is not None:
            out.append(fixed_assign)
            i += 1
            continue

        # 2) Try conditional multiline .append(
        new_block, new_i = fix_multiline_append(lines, i)
        if new_block is not None:
            out.append(new_block)
            i = new_i + 1
            continue

        out.append(line)
        i += 1

    return "\n".join(out) + ("\n" if not txt.endswith("\n") else "")

def main():
    if not TARGET.exists():
        print(f"[ERR] Fichier introuvable: {TARGET}")
        sys.exit(1)
    src = TARGET.read_text(encoding="utf-8")
    # backup
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
    print("[OK] Réparation v4 terminée. Fichier valide. Sauvegarde: __init__.py.bak")

if __name__ == "__main__":
    main()
