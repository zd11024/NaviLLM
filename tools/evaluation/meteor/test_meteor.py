from meteor import Meteor


meteor_eval = Meteor()
ref = {0: [u'a shoe rack with some shoes and a dog sleeping on them .', 
        u'a small dog is curled up on top of the shoes .', 
        u'various slides and other footwear rest in a metal basket outdoors .', 
        u'a dog sleeping on a show rack in the shoes .', 
        u'this wire metal rack holds several pairs of shoes and sandals .'], }
hypo = {0: [u'a large white plate with a white plate with a white plate .'],}
result = meteor_eval.compute_score(ref, hypo)
print(result)