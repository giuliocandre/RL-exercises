
# An RL agent that learns how to play blackjack
# Infinite deck, every card can be 1 - 10
# The density of each card in the deck is: 1/13 except for 10 that happens 4/13 of the times
# an episode ends with :
# +1 if match is won
# -1 if it's lost or drawn
# possible actions are:
# hit or stop
# the state is a combo of
# my_sum, usable_ace, dealer_card,

# the policy of the dealer is always:
# if dealer_sum < 17: hit
# else: stop

# We use first-visit MC 
# if a state is new: calculate if it's terminal -> assign the right Value 
# if it's not terminal -> assign 0.5
# at the end of the episode backpropagate 

import random

ACTION_STOP = 0
ACTION_HIT = 1
explore_threshold = 0.1
actions = [ACTION_STOP, ACTION_HIT]
cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
value_map = {}

def get_card():
    return random.choice(cards)

def get_random_state():
    my_cards = tuple(get_card() for _ in range(2))
    usable_ace = int(1 in my_cards)
    my_sum = sum(my_cards) + usable_ace*10

    dealer_card = get_card()
    state = (my_sum, usable_ace, dealer_card)
    return state

def value(state, action):
    v = value_map.get((state,action), 0.5)
    return v

def policy(state):
    # greedy algorithm that optimizies value(s, a)
    my_sum, _, _ = state
    if my_sum < 11:
        return ACTION_HIT
    
    if my_sum == 21:
        return ACTION_STOP
    
    if random.random() < explore_threshold:
        return random.choice(actions)
    a = max(actions, key=lambda a : value(state, a))
    return a

def get_dealer_sum(dealer_card):
    has_ace = dealer_card == 1
    dealer_sum = dealer_card + 10*has_ace
    while dealer_sum < 17:
        new_card = get_card()
        has_ace = has_ace or new_card == 1

        dealer_sum += new_card
        if dealer_sum > 21 and has_ace:
            has_ace = 0
            dealer_sum -= 10

        # print(f"dealer got {new_card}, cur sum = {dealer_sum}")

    return dealer_sum
 
# This effectively interacts with the environment
def reward(state, action):
    my_sum = state[0]
    usable_ace = state[1]
    dealer_card = state[2]

    assert(my_sum <= 21)

    if action == ACTION_HIT:
        assert(my_sum < 21)

        new_card = get_card()
        # First usable ace? 
        if new_card == 1 and not usable_ace:
                my_sum += 11
                usable_ace = 1

        else:
            my_sum += new_card
        
        if my_sum > 21 and not usable_ace:
            return -1, None # Game over
        
        # Use the ace to reduce the sum and keep going
        if my_sum > 21 and usable_ace:
            my_sum -= 10
            usable_ace = 0
            
        return 0, (my_sum, usable_ace, dealer_card)

    # If action == ACTION_STOP, we need to let the dealer play
    final_dealer_sum = get_dealer_sum(dealer_card)
    if final_dealer_sum > 21 or my_sum > final_dealer_sum:
        return 1, None # Game over, we won
    
    return -1, None # Game over, we lost (or drawn)

def episode(s = None):
    if not s:
        s = get_random_state()
    r = 0
    history = []
    while r == 0:
        a = policy(s)
        history.append((r, s, a))
        r, s = reward(s, a)
        if r != 0: # Game ended
            history.append((r, None, None))
    
    return history

counts = {}
def policy_update(n_episodes = 5000):
    for i in range(n_episodes): # 5000 games
        h = episode()
        g = 0
        for i in range(len(h) - 2, -1, -1):
            next_r, _, _ = h[i + 1]
            _, s, a = h[i]
            g += next_r
            count = counts.get((s,a), 0)
            value_map[(s,a)] = count/(count + 1)*value_map.get((s,a), 0) + g/(count + 1)
            counts[(s,a)] = counts.get((s,a), 0) + 1
    