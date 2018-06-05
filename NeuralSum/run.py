
from evaluation import test_single, test_all
from preprocessing import parse_duc

from config import config
import sys

def dev_test():
    # preprocessing tasks
    articles = parse_duc()

    # h = "Cambodian government rejects opposition's call for talks abroad"
    # r = "Cambodian leader Hun Sen rejects opposition demands for talks in Beijing."
    # test_single(h,r)

    # print('')
    # test_all([h,h1], [r,r1])

def main():
    if len(sys.argv) == 1:
        dev_test()
        return

    if sys.argv[1] == 'train':
        pass
    elif sys.argv[1] == 'test':
        pass

if __name__ == '__main__':
    print("log level: " + config["log_level"])
    main()
