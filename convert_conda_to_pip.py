# convert_conda_to_pip.py
output = []
with open("requirements.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "pypi_0" in line:
            try:
                pkg, version, _ = line.split("=")
                output.append(f"{pkg}=={version}")
            except ValueError:
                continue

with open("requirements_pip.txt", "w") as f:
    for line in output:
        f.write(line + "\n")
