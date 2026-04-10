import sys

from odyssey.cli import main


if __name__ == "__main__":
    main(["run-ablations", *sys.argv[1:]])

