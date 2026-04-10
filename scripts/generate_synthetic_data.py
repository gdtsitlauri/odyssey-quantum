import sys

from odyssey.cli import main


if __name__ == "__main__":
    main(["generate-synthetic", *sys.argv[1:]])

