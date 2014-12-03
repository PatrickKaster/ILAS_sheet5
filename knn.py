#!/usr/bin/env python3
import example
import random
from optparse import OptionParser

class ranking:
    def __init__(self,k):
        self.contents = [None]*k 

    def insert(self,value,obj):
        for idx in range(len(self.contents)):
            if self.contents[idx] == None:
                self.contents[idx] = (value,obj)
                break
            elif value <= self.contents[idx][0]:
                self.contents.insert(idx,(value,obj))
                self.contents = self.contents[0:-1]
                break
                
    def __iter__(self):
        return iter(self.contents)

def knn(k,exampleset,instance):
    rank_list = ranking(k)
    # execute a linear search over all examples
    for ex in exampleset:
        # insert the instance into the ranking of the k nearest points
        if ex.active:
            rank_list.insert(ex.euclidian_distance_to(instance),ex)
    return rank_list

def voting(rank_list):
    true  = 0
    false = 0
    # let every example in the rank_list take a vote
    for value, example in rank_list:
        if example.outcome:
            true += 1
        else:
            false += 1
    return true >= false

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename", help="file with examples", metavar="FILE")
    parser.add_option("-k", dest="k", help="number of nearest neighbors", metavar="NUMBER", type="int")
    (options, args) = parser.parse_args()

    es = example.ExampleSet()
    es.initialize_from_file(options.filename)
    es.transfer_to_numerical()

    nbr_of_instances = 5
    instance_list = []
    for idx in random.sample(range(len(es.examples)),nbr_of_instances):
        instance_list.append(es.examples[idx])
        es.examples[idx].active = False
        es.examples.remove(es.examples[idx])

    for instance in instance_list:
        rl = knn(options.k,es,instance)
        print("++++++++++  {}  +++++++++".format(instance.idx))
        for value, example in rl:
            print("IDX: {}, Distance {}".format(example.idx, value))
