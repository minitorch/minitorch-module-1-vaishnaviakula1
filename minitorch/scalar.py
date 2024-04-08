from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple, Type, Union
import numpy as np
from .autodiff import Context, Variable, backpropagate, central_difference
from .scalar_functions import EQ, LT, Add, Exp, Inv, Log, Mul, Neg, ReLU, ScalarFunction, Sigmoid

ScalarLike = Union[float, int, "Scalar"]

@dataclass
class ScalarHistory:
    last_fn: Optional[Type[ScalarFunction]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Scalar] = ()

_var_count = 0

class Scalar:
    history: Optional[ScalarHistory]
    derivative: Optional[float]
    data: float
    unique_id: int
    name: str

    def __init__(self, v: float, back: ScalarHistory = ScalarHistory(), name: Optional[str] = None):
        global _var_count
        _var_count += 1
        self.unique_id = _var_count
        self.data = float(v)
        self.history = back
        self.derivative = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

    def __repr__(self) -> str:
        return "Scalar(%f)" % self.data

    def __mul__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, b)

    def __truediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(self, Inv.apply(b))

    def __rtruediv__(self, b: ScalarLike) -> Scalar:
        return Mul.apply(b, Inv.apply(self))

    def __add__(self, b: ScalarLike) -> Scalar:
        return Add.apply(self, b)

    def __sub__(self, b: ScalarLike) -> Scalar:
        return Add.apply(self, Neg.apply(b))

    def __neg__(self) -> Scalar:
        return Neg.apply(self)

    def __lt__(self, b: ScalarLike) -> Scalar:
        return LT.apply(self, b)

    def __gt__(self, b: ScalarLike) -> Scalar:
        return LT.apply(b, self)

    def __eq__(self, b: ScalarLike) -> Scalar:
        return EQ.apply(self, b)

    def __radd__(self, b: ScalarLike) -> Scalar:
        return self + b

    def __rmul__(self, b: ScalarLike) -> Scalar:
        return self * b

    def log(self) -> Scalar:
        return Log.apply(self)

    def exp(self) -> Scalar:
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        return ReLU.apply(self)

    def accumulate_derivative(self, x: Any) -> None:
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.derivative is None:
            self.derivative = 0.0
        self.derivative += x

    def is_leaf(self) -> bool:
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        grads = h.last_fn.backward(h.ctx, d_output)
        return zip(h.inputs, grads)

    def backward(self, d_output: Optional[float] = None) -> None:
        if d_output is None:
            d_output = 1.0
        backpropagate(self, d_output)
