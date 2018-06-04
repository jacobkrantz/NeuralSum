
from evaluation import test_single, test_all

from config import config

print("log level: " + config["log_level"])



# preprocessing matters:


h = "Cambodian government rejects opposition's call for talks abroad"
r = "Cambodian leader Hun Sen rejects opposition demands for talks in Beijing."
test_single(h,r)


# print('')
# test_all([h,h1], [r,r1])
