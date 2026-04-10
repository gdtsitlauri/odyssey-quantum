import sys

from odyssey.cli import main


if __name__ == "__main__":
    main(["run-odyssey", *sys.argv[1:]])

