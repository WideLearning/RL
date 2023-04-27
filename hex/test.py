import unittest
from board import HexBoard


class TestHexBoard(unittest.TestCase):
    def test_init(self):
        board = HexBoard(inner_size=2, frame_size=1)

        self.assertEqual(board.inner_size, 2)
        self.assertEqual(board.frame_size, 1)
        self.assertEqual(board.total_size, 4)
        self.assertEqual(board.current_player(), 1)

    def test_inside(self):
        board = HexBoard(inner_size=2, frame_size=1)

        self.assertTrue(board.inside(0, 0))
        self.assertTrue(board.inside(1, 1))
        self.assertTrue(board.inside(2, 0))
        self.assertTrue(board.inside(3, 3))

        self.assertFalse(board.inside(-1, 0))
        self.assertFalse(board.inside(1, 4))
        self.assertFalse(board.inside(5, 5))

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


if __name__ == "__main__":
    board = HexBoard(inner_size=2, frame_size=1)
    board.put(0, 1)
    print(str(board))
