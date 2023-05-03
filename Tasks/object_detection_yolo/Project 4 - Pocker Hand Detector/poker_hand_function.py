import re
from typing import List, Tuple, Literal, Union
from collections import Counter


def parse_hand(hand: Union[List[str], str], 
               pattern: str = r"([AKQJ]|[0-9]{1,2})([HCDS])") -> Tuple[List[Union[int, str]]]:
    ranks = []
    suits = []

    for card in hand:
        rank, suit = parse_card(card=card, pattern=pattern)
        ranks.append(rank)
        suits.append(suit)

    return ranks, suits

def parse_card(card: str, pattern: str = r"([AKQJ]|[0-9]{1,2})([HCDS])") -> Tuple[str, str]:
    # Ranks:
    #   A - туз
    #   K - король
    #   Q - королева
    #   J - валет
    #   4 ... 10 - численные значения
    #
    # Suits
    #   H - черви  (Hearts)
    #   С - крести (Clubs)
    #   D - бубны  (Diamonds)
    #   S - пики   (spades)
    parsed = re.search(pattern=pattern, string=card, flags=re.IGNORECASE)
    rank = parsed.group(1)
    if rank.isdigit():
        rank = int(rank)
    suit = parsed.group(2)
    return rank, suit


class CheckHand:
    def __init__(self, pattern: str = r"([AKQJ]|[0-9]{1,2})([HCDS])") -> None:
        self.pattern = pattern

    def __call__(self, hand: List[str]):
        ranks, suits = parse_hand(hand=hand, pattern=self.pattern)
        combs = [
            "Royal flush",
            "Straight flush",
            "Four of kind",
            "Full house",
            "Flush",
            "Straight",
            "Three of a kind",
            "Two pair",
            "Pair",
            ]
        for comb in combs:
            check_func = getattr(self, "check_" + comb.lower().replace(" ", "_"))
            check_res = check_func(ranks=ranks, suits=suits)

            if check_res:
                return comb

        return "Hight card"
    
    def check_royal_flush(self, ranks: List[Union[int, str]], suits: List[Union[int, str]]) -> bool:
        cr = Counter(ranks)
        if (cr.get("A") == 1) & (cr.get("K") == 1) & (cr.get("Q") == 1) & \
           (cr.get("J") == 1) & (cr.get(10)  == 1):
            return self.check_flush(suits=suits)
        return False
    
    def check_straight_flush(self, ranks: List[Union[int, str]], suits: List[Union[int, str]]) -> bool:
        return self.check_flush(suits=suits) and self.check_straight(ranks=ranks)
    
    def check_four_of_kind(self, ranks: List[Union[int, str]], **args) -> bool:
        cr = Counter(ranks)
        return 4 in cr.values()
    
    def check_full_house(self, ranks: List[Union[int, str]], **args) -> bool:
        cr = Counter(ranks)
        return (3 in cr.values()) and (2 in cr.values())
    
    def check_flush(self, suits: List[Union[int, str]], **args) -> bool:
        cs = Counter(suits)
        return len(cs) == 1
    
    def check_straight(self, ranks: List[Union[int, str]], **args) -> bool:
        values = {
            "A": 14,
            "K": 13,
            "Q": 12,
            "J": 11,
            }
        
        arr = []
        min_value = float("inf")
        for v in ranks:
            if isinstance(v, str):
                v = values.get(v)
            arr.append(v)
            min_value = min(v, min_value)
        arr = [v - min_value for v in arr]
        return {0, 1, 2, 3, 4}.issubset(set(arr))
    
    def check_three_of_a_kind(self, ranks: List[Union[int, str]], **args) -> bool:
        cr = Counter(ranks)
        return 3 in cr.values()
    
    def check_two_pair(self, ranks: List[Union[int, str]], **args) -> bool:
        cr = Counter(ranks)
        return Counter(cr.values())[2] == 2
    
    def check_pair(self, ranks: List[Union[int, str]], **args) -> bool:
        cr = Counter(ranks)
        return Counter(cr.values())[2] == 1
    