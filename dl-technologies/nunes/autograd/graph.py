import typing as tp
from collections import defaultdict

import graphviz
import numpy as np
from toposort import toposort_flatten

from .functional import Function, ReduceFunction

IdType = int


class Graph:
    """
    Base class which stores forward operations in graph structure.
    Allows to call backward method for applying backpropagation algorithm
    """

    def __init__(self):
        self.nodes_ids: tp.Set[int] = set()
        self.operations: tp.TypedDict[IdType, Function] = dict()
        self.operands: tp.TypedDict[IdType, tp.Any] = dict()
        self.edges: tp.TypedDict[IdType, tp.List[IdType]] = defaultdict(list)

    def add(self, operation: Function):
        node_id = id(operation.result)
        self.nodes_ids.add(node_id)

        for operand in operation.operands():
            operand_id = id(operand)
            self.nodes_ids.add(operand_id)
            self.operands[operand_id] = operand
            self.edges[node_id].append(operand_id)

        self.operations[node_id] = operation
        self.operands[node_id] = operation.result

    def clear(self):
        self.nodes_ids.clear()
        self.operands.clear()
        self.operations.clear()
        self.edges.clear()

    def is_leaf(self, node_id: int) -> bool:
        return node_id not in self.edges

    def backward(self):
        assert len(self.edges) > 0, "No edges in back-propagation graph"

        root, *vertices = toposort_flatten(self.edges)[::-1]
        reduction = self.operations[root]

        assert isinstance(reduction, ReduceFunction), "Only scalar tensors can be back-propagated"

        [loss_gradient] = reduction.backward(output=np.array(1.0))
        operand = reduction.tensor
        operand.grad = loss_gradient

        for v in vertices:
            if self.is_leaf(v):
                continue

            op = self.operations[v]
            output_grad = self.operands[v].grad
            grads = op.backward(output_grad)

            for operand, g in zip(op.operands(), grads):
                if operand is None or g is None:
                    continue

                if operand.requires_grad:
                    if operand.grad is None:
                        operand.grad = g
                    else:
                        operand.grad += g

        self.clear()

    def graphviz(self):
        g = graphviz.Graph("Computational Graph")

        for node in self.nodes_ids:
            if self.is_leaf(node):
                g.node(str(node))
            else:
                g.node(str(node), repr(self.operations[node]))

        for to, in_comming in self.edges.items():
            for arg in in_comming:
                g.edge(head_name=str(arg), tail_name=str(to))

        return g


__all__ = [
    "Graph"
]
