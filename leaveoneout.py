#!/usr/bin/env python3
import knn
import example
import sys
from optparse import OptionParser

if sys.version_info[0] < 3 or sys.version_info[0] == 3 and sys.version_info[1] < 2:
    raise Exception('Need Python with version newer than 3.2')

def leave_one_out(examples,k):
    right_classified = 0
    for ex in examples:
        # disable only this example
        ex.active = False
        # run the k-Nearest-Neighbor algorithm
        rank_list = knn.knn(k,examples,ex)
        # check the voting for correctness
        outcome = knn.voting(rank_list)
        if outcome == ex.outcome:
            right_classified += 1
        ex.active = True
    # return which share was correctly classified
    return right_classified/float(len(examples.examples))

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename", help="file with examples", metavar="FILE")
    parser.add_option("-k", dest="k", help="number of nearest neighbors", metavar="NUMBER", type="int")
    (options, args) = parser.parse_args()

    es = example.ExampleSet()
    es.initialize_from_file(options.filename)
    es.transfer_to_numerical()

    print("PCT of correctly classified = {}".format(leave_one_out(es,options.k)*100))
