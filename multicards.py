import os
import subprocess
import sys
from os import listdir
from os.path import isfile, join
import re

import vpype_cli

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))

# number = 4
#
# if number not in [2,4]:
#     raise ValueError("number must be 2 or 4")
#
# # make directory if it doesn't exist
# subprocess.Popen(
#     ["mkdir", "-p", "output/grid"],
#     stdout=subprocess.PIPE
# )
#
# for i in range(number):
#     # execute the generative_tree.py script a number of times
#     svg_data = subprocess.Popen(
#         ["python", "generative_tree.py", "--svg-std-out"],
#         stdout=subprocess.PIPE
#     )
#
#     # write svg data to a file
#     with open(f"output/grid/multicard_{i}.svg", "wb") as f:
#         f.write(svg_data.stdout.read())
#
# files_to_process = [f"output/grid/multicard_{i}.svg" for i in range(number)]
#
# cols = number // 2
#
# vpype_cli.execute(f"""
#     eval "files={repr(files_to_process)}"
#     eval "cols={cols}; rows=2"
#     grid -o 210mm 148mm "%cols%" "%rows%"
#         read --no-fail "%files[_i] if _i < len(files) else ''%"
#     end
#     write output/grid/combined-grid.svg
# """)

vpype_cli.execute(f"""
    read output/grid/combined-grid.svg
    forlayer write "output/grid/combined-grid-%_name%.svg" end
""")

svgs_to_translate = [relative for f in listdir("output/grid")
             if isfile(relative := join("output/grid", f)) and
             re.fullmatch(r"combined-grid-[123].*\.svg", f)]

juicy_gcode = join(script_directory, "juicy-gcode")

for svg in svgs_to_translate:
    output = svg.replace(".svg", ".gcode")
    print(f"Converting {svg} to {output}")
    result = subprocess.run([juicy_gcode, svg, "-f", "juicy-gcode-flavour.yaml", "-o", output], capture_output=True)
    print(result.returncode)
    print(result.stderr.decode("utf-8"))
    print(result.stdout.decode("utf-8"))
