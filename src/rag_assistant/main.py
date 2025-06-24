from conversation import Conversation
from utils import load_env


def main():
    load_env()

    conv = Conversation("gemma")

    conv.run()


if __name__ == "__main__":
    main()
