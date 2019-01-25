import json
import numpy as np
import random

max_price = 0
min_price = 999
max_rank = 26

with open('~/tmp/tickets.json', 'r') as read_file:
    data = json.load(read_file)

train_tickets = []
test_tickets = []

train_outcome = []
test_outcome = []

for ticket in data:
    win_pctg = 0
    osu_win_pctg = 0
    games_played = 0
    osu_rank = 0
    rank = 0
    price = ticket['price']
    rival = 0
    conference = 0

    if max_price < price and price < 800:
        max_price = price

    if min_price > price:
        min_price = price

    if ticket['game'] == 'Oregon State Beavers':
        win_pctg = 0.5 # no games played
        osu_win_pctg = 0.5 # no games played
        games_played = 0 # no games played
        osu_rank = 5
        rank = 26 # unranked
        rival = 0 # not a rival
        conference = 0 # not in our conference

    elif ticket['game'] == 'Rutgers Scarlet Knights':
        win_pctg = 1.0 # 1-0
        osu_win_pctg = 1.0 # 1-0
        games_played = 1
        osu_rank = 4
        rank = 26 # unranked
        rival = 0 # not a rival
        conference = 1 # in our conference

    elif ticket['game'] == 'Tulane Green Wave':
        win_pctg = 1.0 / 3.0 # 1-2
        osu_win_pctg = 1.0 # 3-0
        games_played = 3
        osu_rank = 4
        rank = 26 # unranked
        rival = 0 # not a rival
        conference = 0 # not in our conference

    elif ticket['game'] == 'Indiana Hoosiers':
        win_pctg = 4.0 / 5.0 # 4-1
        osu_win_pctg = 1.0 # 6-0
        games_played = 5
        osu_rank = 3
        rank = 26 # unranked
        rival = 0 # not a rival
        conference = 1 # in our conference

    elif ticket['game'] == 'Minnesota Golden Gophers':
        win_pctg = 3.0 / 5.0 # 3-2
        osu_win_pctg = 1.0 # 6-0
        games_played = 6
        osu_rank = 3
        rank = 26 # unranked
        rival = 0 # not a rival
        conference = 1 # in our conference

    elif ticket['game'] == 'Nebraska Cornhuskers':
        win_pctg = 2.0 / 7.0 # 2-5
        osu_win_pctg = 7.0 / 8.0 # 7-1
        games_played = 8
        osu_rank = 10
        rank = 26 # unranked
        rival = 0 # not a rival
        conference = 1 # in our conference

    else: # Michigan
        win_pctg = 10.0 / 11.0 # 10-1
        osu_win_pctg = 10.0 / 11.0 # 10-1
        games_played = 11
        osu_rank = 10
        rank = 4 # 'best defense in the league LOL'
        rival = 1 # not a rival
        conference = 1 # in our conference


    new_ticket = [
        win_pctg,
        osu_win_pctg,
        rank,
        osu_rank,
        ticket['price'],
        rival,
        conference,
        # games_played
    ]

    if ticket['flag'] == 'Complete':
        outcome = 2
    elif ticket['flag'] == 'Available':
        outcome = 0
    else:
        outcome = 1

    if random.random() > 0.1:
        train_tickets.append(new_ticket)
        train_outcome.append(outcome)
    else:
        test_tickets.append(new_ticket)
        test_outcome.append(outcome)

test_tickets = np.array(test_tickets)
train_tickets = np.array(train_tickets)

test_outcome = np.array(test_outcome)
train_outcome = np.array(train_outcome)

np.savez_compressed(
    '~/tmp/data.npz',
    x_train=train_tickets,
    y_train=train_outcome,
    x_test=test_tickets,
    y_test=test_outcome
)

print(train_tickets[0])

for i in range(train_tickets.shape[0]):
    train_tickets[i][2] = max_rank - train_tickets[i][2] # rank
    train_tickets[i][3] = max_rank - train_tickets[i][3] # osu_rank

    train_tickets[i][2] /= max_rank # rank
    train_tickets[i][3] /= max_rank # osu_rank
    train_tickets[i][4] = (train_tickets[i][4] - min_price) / (max_price - min_price) # price
    # train_tickets[i][7] /= 11 # games_played

for i in range(test_tickets.shape[0]):
    test_tickets[i][2] = max_rank - test_tickets[i][2] # rank
    test_tickets[i][3] = max_rank - test_tickets[i][3] # osu_rank

    test_tickets[i][2] /= max_rank # rank
    test_tickets[i][3] /= max_rank # osu_rank
    test_tickets[i][4] = (test_tickets[i][4] - min_price) / (max_price - min_price) # price
    # test_tickets[i][7] /= 11 # games_played

print(train_tickets[0])

np.savez_compressed(
    '~/tmp/normalized_data.npz',
    x_train=train_tickets,
    y_train=train_outcome,
    x_test=test_tickets,
    y_test=test_outcome
)
