import unittest
from poker_hand_function import CheckHand


class TestPokerHandFunction(unittest.TestCase):

    def setUp(self) -> None:
        self.ch = CheckHand()

    def test_case_1(self):
        self.assertEqual(self.ch(hand=["AH", "KH", "QH", "JH", "10H"]), "Royal flush")

    def test_case_2(self):
        self.assertEqual(self.ch(hand=["AC", "KC", "QC", "JC", "10C"]), "Royal flush")

    def test_case_3(self):
        self.assertEqual(self.ch(hand=["AD", "KD", "QD", "JD", "10D"]), "Royal flush")

    def test_case_4(self):
        self.assertEqual(self.ch(hand=["AS", "QS", "KS", "10S", "JS"]), "Royal flush")

    def test_case_5(self):
        self.assertEqual(self.ch(hand=["8C", "6C", "10C", "9C", "7C"]), "Straight flush")

    def test_case_6(self):
        self.assertEqual(self.ch(hand=["8H", "10H", "9H", "7H", "JH"]), "Straight flush")

    def test_case_7(self):
        self.assertEqual(self.ch(hand=["AH", "AC", "6C", "AD", "AS"]), "Four of kind")

    def test_case_8(self):
        self.assertEqual(self.ch(hand=["10H", "10C", "6C", "10D", "10S"]), "Four of kind")

    def test_case_9(self):
        self.assertEqual(self.ch(hand=["10C", "KH", "10D", "10H", "KD"]), "Full house")

    def test_case_10(self):
        self.assertEqual(self.ch(hand=["9D", "9H", "JH", "JS", "9S"]), "Full house")

    def test_case_11(self):
        self.assertEqual(self.ch(hand=["KC", "9C", "10C", "6C", "AC"]), "Flush")

    def test_case_12(self):
        self.assertEqual(self.ch(hand=["10C", "8H", "JS", "9C", "QD"]), "Straight")

    def test_case_13(self):
        self.assertEqual(self.ch(hand=["AH", "AD", "AC", "KS", "QH"]), "Three of a kind")

    def test_case_14(self):
        self.assertEqual(self.ch(hand=["AH", "AC", "KD", "KS", "7H"]), "Two pair")

    def test_case_15(self):
        self.assertEqual(self.ch(hand=["AH", "AC", "KD", "JS", "7H"]), "Pair")

    def test_case_16(self):
        self.assertEqual(self.ch(hand=["AH", "KC", "QD", "9S", "7H"]), "Hight card")


if __name__ == '__main__':
    unittest.main()
