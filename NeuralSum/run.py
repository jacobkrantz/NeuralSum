
from evaluation import test_single, test_all
from preprocessing import parse_duc_2004,parse_duc_2003, display_articles

from config import config
import sys

def dev_test():
    # preprocessing tasks
    duc_2004_articles = parse_duc_2004()
    duc_2003_articles = parse_duc_2003()
    display_articles(duc_2003_articles, 3, random=False)
    #
    # h = "Cambodian government rejects opposition 's call for talks abroad"
    # r = "Cambodian leader Hun Sen rejects opposition demands for talks in Beijing."
    # test_single(h,r)
    # print test_single(h,r)
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
