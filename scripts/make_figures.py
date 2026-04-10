import sys

from odyssey.cli import main


if __name__ == "__main__":
    main(["make-figures", *sys.argv[1:]])

