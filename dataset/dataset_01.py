import requests
from classes.Crawler import Crawler
import time


_URL = "..."
_data = []

f = open("dataset_01.txt", "w")

i = 1
while True:
    print("%s%d" % (_URL, i))
    _html = Crawler.html(_URL + str(i))

    j = 0
    while True:
        _layer_1 = Crawler.find(_html, 'class="c-build-item">', '</a>', j)
        if _layer_1 == "":
            break

        _layer_2 = Crawler.find(_layer_1, 'class="total-price">', '<label')
        price = int(_layer_2.replace(" ", "").replace("HK", "").replace("$", "").replace(",", ""))

        _layer_3 = Crawler.find(_layer_1, 'class="c-build-size">', '<label>')
        size = int(_layer_3.replace(" ", "").replace(",", ""))

        _data.append([price, size])
        f.write("%d %d\n" % (price, size))
        print(j, ":", price, ",", size)
        j = j + 1

    if j == 0:
        break
    else:
        i = i + 1
        # time.sleep(5)

f.close()
