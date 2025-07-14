from __future__ import annotations
from collections import deque

def create_circular_queue(n: int) -> deque[int]:
    """1부터 n까지의 숫자로 deque를 생성합니다."""
    return deque(range(1, n + 1))

def rotate_and_remove(queue: deque[int], k: int) -> int:
    """
    큐에서 k번째 원소를 제거하고 반환합니다.

    Args:
        - queue(deque[int]): 현재 카드 덱
    """
    # 맨 위 카드 버리기
    queue.popleft()
    # 그 다음 카드 맨 뒤로 보내기 (있으면)
    if queue:
        queue.append(queue.popleft())
    return 0

def simulate_card_game(n: int) -> int:
    """
    카드2 문제의 시뮬레이션
    맨 위 카드를 버리고, 그 다음 카드를 맨 아래로 이동

    Args:
        - n(int): 카드의 개수

    Returns:
        - int: 마지막으로 남은 카드 번호
    """
    queue = create_circular_queue(n)
    while len(queue) > 1:
        rotate_and_remove(queue, 1)
    return queue[0]

def solve_card2() -> None:
    """입, 출력 format"""
    n: int = int(input())
    result: int = simulate_card_game(n)
    print(result)

if __name__ == "__main__":
    solve_card2()