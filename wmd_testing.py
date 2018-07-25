import NeuralSum as ns
from copy import deepcopy

"""
inter annotator WMD on DUC2004
- if you run, change max_dist to 20.0111 in word_mover_distance.py
"""

def main():
    articles = ns.parse_duc_2004()
    wmd = ns.WordMoverDistance()
    distances = []

    for art in articles:
        for target in art.gold_summaries:
            gens = deepcopy(art.gold_summaries)
            gens.remove(target)
            distances += list(map(lambda g: wmd.get_wmd(g,target), gens))

    print 'Total compared: ', len(distances)
    print 'Minimum: ', min(distances)
    print 'Maximum: ', max(distances)

    a,b,c,d,e,f,inf = 0,0,0,0,0,0,0
    for dist in distances:
        if dist > 5:
            f += 1
        elif dist > 4:
            e += 1
        elif dist > 3:
            d += 1
        elif dist > 2:
            c += 1
        elif dist > 1:
            b += 1
        elif dist == 20.0111:
            inf += 1
        else:
            a += 1

    print '0 -> 1: ', a
    print '1 -> 2: ', b
    print '2 -> 3: ', c
    print '3 -> 4: ', d
    print '4 -> 5: ', e
    print '5 ->:   ', f
    print 'inf:    ', inf


if __name__ == '__main__':
    main()
