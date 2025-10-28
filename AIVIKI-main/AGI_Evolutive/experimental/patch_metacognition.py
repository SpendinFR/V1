
import re
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'metacognition' / '__init__.py'

def main():
    if not TARGET.exists():
        raise SystemExit(f'Fichier introuvable: {TARGET}')
    src = TARGET.read_text(encoding='utf-8')

    original = src

    # --- 1) Guard global_activation on cognitive_arch / cognitive_architecture
    # Replace any 'cognitive_arch.global_activation' or 'cognitive_architecture.global_activation'
    src = re.sub(
        r'(?<![A-Za-z0-9_])self\.(cognitive_arch(?:itecture)?)\.global_activation',
        r'getattr(self.\\1 if hasattr(self, "cognitive_arch") else getattr(self, "cognitive_architecture", None), "global_activation", 0.5)',
        src
    )

    # --- 2) Fix string-typed reasoning_system before any use
    # Insert a small guard in __init__ and/or at first method that sets self.reasoning_system
    # We'll inject a helper function once and call it wherever obvious patterns appear.
    helper = (
        "\\n\\n"
        "def _ensure_reasoning_ref(self):\\n"
        "    arch = getattr(self, 'cognitive_arch', getattr(self, 'cognitive_architecture', None))\\n"
        "    if isinstance(getattr(self, 'reasoning_system', None), str):\\n"
        "        self.reasoning_system = getattr(getattr(arch, 'reasoning', None), '__class__', None) and getattr(arch, 'reasoning', None)\\n"
        "    if self.reasoning_system is None and arch is not None:\\n"
        "        self.reasoning_system = getattr(arch, 'reasoning', None)\\n"
    )

    # Inject helper after first class definition header of MetacognitiveSystem
    src = re.sub(
        r'(class\s+MetacognitiveSystem\s*\(.*?\):\s*\n)',
        r'\\1' + helper,
        src,
        count=1,
        flags=re.DOTALL
    )

    # Call helper at starts of common init/update methods if present
    src = re.sub(
        r'(\n\s*def\s+__init__\s*\(self[^\)]*\)\s*:\s*\n)',
        r'\\1        _ensure_reasoning_ref(self)\\n',
        src
    )
    src = re.sub(
        r'(\n\s*def\s+_evaluate_metacognitive_state\s*\(self[^\)]*\)\s*:\s*\n)',
        r'\\1        _ensure_reasoning_ref(self)\\n',
        src
    )
    src = re.sub(
        r'(\n\s*def\s+monitor[^\(]*\s*\(self[^\)]*\)\s*:\s*\n)',
        r'\\1        _ensure_reasoning_ref(self)\\n',
        src
    )

    # --- 3) Safe access to reasoning_history and get_reasoning_stats
    src = src.replace(
        'self.reasoning_system.get_reasoning_stats()',
        '(self.reasoning_system.get_reasoning_stats() if hasattr(self.reasoning_system, "get_reasoning_stats") else (getattr(getattr(getattr(self, "cognitive_arch", getattr(self, "cognitive_architecture", None)), "reasoning", None), "get_reasoning_stats", lambda: {})()))'
    )
    src = src.replace(
        'self.reasoning_system.reasoning_history',
        'getattr(self.reasoning_system, "reasoning_history", getattr(getattr(getattr(self, "cognitive_arch", getattr(self, "cognitive_architecture", None)), "reasoning", None), "reasoning_history", []))'
    )

    # --- 4) Replace any deque slicing patterns [-N:] or [:N] with explicit list conversions
    # Basic patterns (variable[-N:] and variable[:N])
    src = re.sub(r'(\w+)\[-(\d+):\]', r'list(\\1)[-\\2:]', src)
    src = re.sub(r'(\w+)\[:(\d+)\]', r'list(\\1)[:\\2]', src)
    # Common dict/deque cases seen earlier
    src = src.replace('["error_corrections"][-', '["error_corrections"])][-').replace(')])[', ')][')
    src = src.replace('["error_corrections"][-10:]', 'list(self.metacognitive_history["error_corrections"])[-10:]')
    src = src.replace('["insights"][-5:]', 'list(self.metacognitive_history["insights"])[-5:]')
    src = src.replace('events[-', 'list(events)[-')

    if src == original:
        print("Aucun changement nécessaire - le fichier semble déjà corrigé.")
        return

    # Backup then write
    backup = TARGET.with_suffix('.py.bak')
    shutil.copy2(TARGET, backup)
    TARGET.write_text(src, encoding='utf-8')
    print(f"Patch appliqué. Sauvegarde: {backup}")

if __name__ == "__main__":
    main()
