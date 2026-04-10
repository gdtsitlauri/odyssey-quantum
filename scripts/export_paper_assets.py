import sys

from odyssey.cli import main


if __name__ == "__main__":
    main(["export-paper-assets", *sys.argv[1:]])

