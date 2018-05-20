import os
import subprocess
import sys

def main():
    subprocess.call("Rscript {} {}"
    .format(
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "ransack.cli.R"
        ),
        ' '.join(sys.argv[1:])
        )
    )
