from __future__ import annotations
import copy
from collections import deque
from collections import defaultdict
from typing import DefaultDict, List


"""
TODO:
- __init__ 구현하기
- add_edge 구현하기
- dfs 구현하기 (재귀 또는 스택 방식 선택)
- bfs 구현하기
"""


class Graph:
    def __init__(self, n: int) -> None:
        """
        그래프 초기화
        n: 정점의 개수 (1번부터 n번까지)
        adj (DefaultDict[int, list[int]]): 인접 리스트
        """
        self.n = n
        self.adj: DefaultDict[int, list[int]] = defaultdict(list)

    
    def add_edge(self, u: int, v: int) -> None:
        """
        양방향 간선 추가
        u(int): 첫 번째 정점 번호
        v(int): 두 번쨰 정점 번호
        """
        self.adj[u].append(v)
        self.adj[v].append(u)
        self.adj[u].sort()
        self.adj[v].sort()
    
    def dfs(self, start: int) -> list[int]:
        """
        깊이 우선 탐색 (DFS)
        
        구현 방법 선택:
        - 재귀 방식: 함수 내부에서 재귀 함수 정의하여 구현

        start(int): 탐색을 시작할 정점 번호
        list[int]: 방문 순서대로 정점 번호가 담긴 리스트
        """
        visited: list[int] = []
        seen: set[int] = set()

        def _dfs(v: int) -> None:
            visited.append(v)
            seen.add(v)
            for u in self.adj[v]:
                if u not in seen:
                    _dfs(u)

        _dfs(start)
        return visited
    
    def bfs(self, start: int) -> list[int]:
        """
        너비 우선 탐색 (BFS)
        큐를 사용하여 구현
        """
        visited: list[int] = []
        seen: set[int] = set()
        q = deque([start])
        seen.add(start)

        while q:
            v = q.popleft()
            visited.append(v)
            for u in self.adj[v]:
                if u not in seen:
                    seen.add(u)
                    q.append(u)
        return visited

    
    def search_and_print(self, start: int) -> None:
        """
        DFS와 BFS 결과를 출력
        """
        dfs_result = self.dfs(start)
        bfs_result = self.bfs(start)
        
        print(' '.join(map(str, dfs_result)))
        print(' '.join(map(str, bfs_result)))



from typing import Callable
import sys


"""
-아무것도 수정하지 마세요!
"""


def main() -> None:
    intify: Callable[[str], list[int]] = lambda l: [*map(int, l.split())]

    lines: list[str] = sys.stdin.readlines()

    N, M, V = intify(lines[0])
    
    graph = Graph(N)  # 그래프 생성
    
    for i in range(1, M + 1): # 간선 정보 입력
        u, v = intify(lines[i])
        graph.add_edge(u, v)
    
    graph.search_and_print(V) # DFS와 BFS 수행 및 출력


if __name__ == "__main__":
    main()
