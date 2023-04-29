import unittest

import numpy as np
from envs.hex import HexBoard, HexEnv


class TestHexBoard(unittest.TestCase):
    def test_init(self):
        board = HexBoard(inner_size=2, frame_size=1)

        self.assertEqual(board.inner_size, 2)
        self.assertEqual(board.frame_size, 1)
        self.assertEqual(board.total_size, 4)
        self.assertEqual(board.current_player(), 1)

    def test_inside_total(self):
        board = HexBoard(inner_size=2, frame_size=1)

        self.assertTrue(board.inside_total(-1, -1))
        self.assertTrue(board.inside_total(0, 0))
        self.assertTrue(board.inside_total(1, -1))
        self.assertTrue(board.inside_total(2, 2))

        self.assertFalse(board.inside_total(-2, 0))
        self.assertFalse(board.inside_total(0, 3))
        self.assertFalse(board.inside_total(4, 4))

    def test_inside_inner(self):
        board = HexBoard(inner_size=2, frame_size=1)

        self.assertTrue(board.inside_inner(0, 0))
        self.assertTrue(board.inside_inner(1, 0))

        self.assertFalse(board.inside_inner(-1, -1))
        self.assertFalse(board.inside_inner(2, 2))
        self.assertFalse(board.inside_inner(-2, 0))
        self.assertFalse(board.inside_inner(0, 3))
        self.assertFalse(board.inside_inner(4, 4))

    def test_str(self):
        board = HexBoard(inner_size=2, frame_size=1)
        board.put(0, 1)
        self.assertListEqual(
            str(board).splitlines(),
            [
                ". O O .   ",
                "          ",
                " + . O +  ",
                "          ",
                "  + . . + ",
                "          ",
                "   . O O .",
            ],
        )

    def test_win(self):
        board = HexBoard(inner_size=2, frame_size=1).put_list(
            [
                (0, 0, 1),
                (1, 1, 1),
            ]
        )
        self.assertEqual(board.win(), 0)

        board = HexBoard(inner_size=2, frame_size=1).put_list(
            [
                (0, 1, 1),
                (1, 1, -1),
                (1, 0, 1),
            ]
        )
        self.assertEqual(board.win(), 1)

        board = HexBoard(inner_size=2, frame_size=1).put_list(
            [
                (0, 0, 1),
                (1, 1, -1),
                (1, 0, -1),
            ]
        )
        self.assertTrue(board.win(), -1)

    def test_bfs(self):
        board = HexBoard(inner_size=2, frame_size=1).put_list(
            [
                (0, 0, 1),
                (1, 1, 1),
                (0, 1, -1),
            ]
        )
        self.assertEqual(board.top, [(0, 0), (0, 1)])
        result = board.bfs(board.top, 1)
        self.assertEqual(
            result.tolist(),
            [
                [False, True, True, False],
                [False, True, False, False],
                [False, False, False, False],
                [False, False, False, False],
            ],
        )

    def test_features(self):
        board = HexBoard(inner_size=2, frame_size=1).put_list(
            [
                (0, 0, 1),
                (1, 1, 1),
                (0, 1, -1),
            ]
        )
        features = board.to_features()
        self.assertEqual(features.shape, (6, 4, 4))
        self.assertEqual(
            features[0].tolist(),
            [
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ],
        )
        self.assertEqual(
            features[1].tolist(),
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        )
        self.assertEqual(
            features[2].tolist(),
            [
                [0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        )
        self.assertEqual(
            features[3].tolist(),
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
            ],
        )
        self.assertEqual(
            features[4].tolist(),
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        )
        self.assertEqual(
            features[5].tolist(),
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        )

    def test_to_pieces(self):
        board = HexBoard(inner_size=2, frame_size=1).put_list(
            [
                (0, 0, 1),
                (1, 1, 1),
                (0, 1, -1),
            ]
        )
        self.assertCountEqual(board.to_pieces(), [(0, 0, 1), (0, 1, -1), (1, 1, 1)])


class TestHexEnv(unittest.TestCase):
    def test_reset(self):
        env = HexEnv(inner_size=2, frame_size=1)
        obs, info = env.reset(seed=42)
        self.assertEqual(obs.shape, (1, 6, 4, 4))
        self.assertEqual(obs.dtype, np.float32)
        self.assertEqual(obs.sum(), 8)
        self.assertEqual(info, {})

    def test_win(self):
        env = HexEnv(inner_size=2, frame_size=1)
        _obs, _info = env.reset(seed=42)
        _obs, reward, terminated, truncated, _info = env.step((0, 0))
        self.assertEqual((reward, terminated, truncated), (0.0, False, False))
        _obs, reward, terminated, truncated, _info = env.step((0, 1))
        self.assertEqual((reward, terminated, truncated), (0.0, False, False))
        _obs, reward, terminated, truncated, _info = env.step((1, 0))
        self.assertEqual((reward, terminated, truncated), (1.0, True, False))


if __name__ == "__main__":
    pass
