import argparse
import glob


def configure_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tests", 
        type=str,
        default="tests",
        help="Path to folder with tests. Each test should end with *_test.txt suffix. (default: tests)"
    )
    parser.add_argument(
        "--benchmark",
        type=bool,
        default=False,
        help="Enables benchmark mode. (default: False)"
    )
    parser.add_argument(
        "--benchmark-retries",
        type=int,
        default=50,
        help="Specifies how many times each test will be runned in benchmark mode. (default: 50)"
    )
    return parser.parse_args()



def run_test(path: str, amount: int = 1):
    with open(path, mode="r", encoding="utf-8") as test:
        pass

if __name__ == "__main__":
    args = configure_parser()

    search_pattern = f"{args.tests}/*_test.txt"
    for test_path in glob.glob(search_pattern):
        run_test(test_path)
