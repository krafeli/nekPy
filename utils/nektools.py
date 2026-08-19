from pathlib import Path
import re
from nekPy.utils.bash import run_command
from pprint import pformat

def gmsh2nek(dir, mshfile, outname, dim=3, periodic_pairs=[]):
    gmsh2nek_input = f'3\n{mshfile}\n0\n{len(periodic_pairs)}'
    for pair in periodic_pairs:
        gmsh2nek_input += f'\n{pair[0]} {pair[1]}'
    
    gmsh2nek_input += f'\n{outname}'
    run_command(["gmsh2nek"], dir=dir, input=gmsh2nek_input)
    
def genmap(dir, re2file, tol=0.001):
    genmap_input = f'{re2file}\n+{tol}'
    run_command(["genmap"], dir=dir, input=genmap_input)

def msh2nek(dir, mshfile, outname, dim=3, periodic_pairs=[], tol=0.001):
    gmsh2nek(dir, mshfile, outname, dim=dim, periodic_pairs=periodic_pairs)
    genmap(dir, outname, tol)

def cleannek(dir):
    run_command(["rm -r obj *.nek5000"], dir=dir)

def cleandir(dir):
    run_command(["rm -r out obj *.f* *.txt *.nek5000"], dir=dir)

def makenek(dir, name):
    run_command([f"makenek {name}" ], dir=dir)

def parse_value(value):
    value = value.strip()
    low = value.lower()
    if low == "yes":
        return True
    if low == "no":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value

def format_value(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)

class ParFile:
    
    def __init__(self, path):
        self.path = Path(path)
        self.lines = self.path.read_text().splitlines()
        self.entries = {}
        self.sections = {}
        self._parse()

    def _parse(self):
        self.entries = {}
        self.sections = {}
        section = None
        for i, line in enumerate(self.lines):
            code = line.split("#", 1)[0].strip()
            if not code:
                continue
            # section
            if code.startswith("[") and code.endswith("]"):
                section = code[1:-1].strip()
                self.sections[section] = i
                continue
            if section is None or "=" not in code:
                continue
            key, value = code.split("=", 1)
            self.entries[(section, key.strip())] = {
                "line": i,
                "value": parse_value(value),
            }

    def get(self, section, key):
        return self.entries[(section.upper(), key)]["value"]

    def set(self, section, key, value):
        entry = self.entries[(section, key)]
        i = entry["line"]
        line = self.lines[i]
        # Separate inline comment
        code, sep, comment = line.partition("#")
        key_part, old_value = code.split("=", 1)

        # Preserve whitespace around value
        leading = old_value[:len(old_value) - len(old_value.lstrip())]
        trailing = old_value[len(old_value.rstrip()):]
        code = key_part + "=" + leading + format_value(value) + trailing
        
        if sep:
            line = code + "#" + comment
        else:
            line = code

        self.lines[i] = line
        entry["value"] = value

    def add(self, section, key, value, comment=None):
        if (section, key) in self.entries:
            raise KeyError(
                f"Parameter '{key}' already exists in section '{section}'"
            )

        value = format_value(value)

        if comment:
            new_line = f"{key} = {value}    # {comment}"
        else:
            new_line = f"{key} = {value}"

        if section not in self.sections:
            raise KeyError(f"Section {section} does not exist")

        # Find end of section
        start = self.sections[section]
        insert_at = len(self.lines)
        for i in range(start + 1, len(self.lines)):
            stripped = self.lines[i].strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                insert_at = i
                break

        # Insert before trailing blank lines of the section
        while (
            insert_at > start + 1
            and not self.lines[insert_at - 1].strip()
        ):
            insert_at -= 1

        self.lines.insert(insert_at, new_line)
        self._parse()

    def write(self, path=None):
        path = Path(path) if path is not None else self.path
        path.write_text("\n".join(self.lines) + "\n")

    def __getitem__(self, key):
        section, name = key
        return self.get(section, name)

    def __setitem__(self, key, value):
        section, name = key
        self.set(section, name, value)
    
    def __str__(self):
        return pformat(self.entries)
        
class SizeFile:
    
    def __init__(self, path):
        self.path = Path(path)
        self.lines = self.path.read_text().splitlines()

        self.entries = {}

        self._parse()

    def _parse(self):
        self.entries = {}

        for i, line in enumerate(self.lines):
            code = line.split("!", 1)[0]
            match = re.search(r"\bparameter\s*\((.*?)\)", code, re.IGNORECASE)

            if not match: continue
            content = match.group(1)
            for item in content.split(","):
                if "=" not in item:
                    continue
                key, value = item.split("=", 1)
                key = key.strip()
                value = value.strip()
                self.entries[key] = {
                    "line": i,
                    "value": parse_value(value),
                }

    def get(self, key):
        return self.entries[key]["value"]

    def set(self, key, value):
        if key not in self.entries:
            raise KeyError(f"Unknown SIZE parameter '{key}'")

        i = self.entries[key]["line"]
        line = self.lines[i]

        # Only replace this specific parameter value
        pattern = (
            rf"(\b{re.escape(key)}\s*=\s*)"
            rf"([^,\)]+)"
        )

        line, count = re.subn(
            pattern,
            lambda m: m.group(1) + str(value),
            line,
            count=1,
        )

        if count != 1:
            raise ValueError(
                f"Could not replace parameter '{key}'"
            )

        self.lines[i] = line
        self.entries[key]["value"] = value

    def write(self, path=None):
        path = Path(path) if path is not None else self.path
        path.write_text("\n".join(self.lines) + "\n")

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)
    
    def __str__(self):
        return pformat(self.entries)