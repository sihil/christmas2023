import subprocess

import vpype_cli

number = 4

if number not in [2,4]:
    raise ValueError("number must be 2 or 4")

for i in range(number):
    # execute the generative_tree.py script a number of times
    svg_data = subprocess.Popen(["python", "generative_tree.py", "--svg-std-out"], stdout=subprocess.PIPE)

    # write svg data to a file
    with open(f"multicard_{i}.svg", "wb") as f:
        f.write(svg_data.stdout.read())

files_to_process = [f"multicard_{i}.svg" for i in range(number)]

cols = number // 2

vpype_cli.execute(f"""
    eval "files={repr(files_to_process)}" 
    eval "cols={cols}; rows=2" 
    grid -o 210mm 148mm "%cols%" "%rows%" 
        read --no-fail "%files[_i] if _i < len(files) else ''%" 
    end
    write combined-grid.svg
""")