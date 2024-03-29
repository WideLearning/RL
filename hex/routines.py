from copy import deepcopy

from board import Board


def contest(size, first, second):
    board = Board(size, size)
    first.clear_history()
    second.clear_history()
    while not board.winner():
        player = first if board.player == 1 else second
        temp = player.make_move(board)
        if temp:
            board = temp
        else:
            board.rollback()
            board.rollback()
    return board.winner()


def tournament(size, first, second, n=1000):
    w = 0
    first.random_rate = 0
    second.random_rate = 0
    first.learning = False
    second.learning = False
    for game in range(n):
        if game % 2 == 0:
            w += contest(size, first=first, second=second) == 1
        else:
            w += contest(size, first=second, second=first) == -1
    return w / n


def training_camp(size, model, n=10):
    old = deepcopy(model)
    model.learning = True
    model.random_rate = 0.01
    for game in range(n):
        contest(size, model, model)
    new = deepcopy(model)
    old.learning = False
    new.learning = False
    boost = tournament(new, old, n = 100)
    print(f'boosted by {boost}', flush=True)
    return model
