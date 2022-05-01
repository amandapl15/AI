# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

# To solve this exercise we only have to modify one value, which is the answerNoise, this is how often an agent ends
# up in an unintended successor state, so if we change that to 0, the agent would cross the bridge with no doubt

def question2():
    answerDiscount = 0.9
    answerNoise = 0
    return answerDiscount, answerNoise

# With a considerable livingReward, the agent would prefer to obtain 1, because the reward of reaching 10 is almost
# the same but with more probability to risk the cliff, and avoiding the cliff would not be that good because the
# reward is too low
def question3a():
    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -3
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# With these values is more optimal to reach the close exit and avoiding the cliff, because the livingReward is not
# that high but the answerDiscount is low so the more distant exit would not be the most optimal policy
def question3b():
    answerDiscount = 0.5
    answerNoise = 0.2
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# Like in the previous question (2) here we have no noise and no livingReward and the discount almost 1 to have
# the optimal policy of prefer the distant exit
def question3c():
    answerDiscount = 0.9
    answerNoise = 0
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# Like in c but now we add some noise to not risk the cliff
def question3d():
    answerDiscount = 0.9
    answerNoise = 0.3
    answerLivingReward = 0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

# With this optimal policy we can add a very high LivingReward, that makes optimal to not reach the exits.
# And with a very high noise so the agent never risks the cliff
def question3e():
    answerDiscount = 0.01
    answerNoise = 0.9
    answerLivingReward = 20
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = None
    answerLearningRate = None
    return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
