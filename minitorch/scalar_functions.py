from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch
from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple
    from .scalar import Scalar, ScalarLike


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


def unwrap_tuple(x):  # type: ignore
    "Turn a singleton tuple into a value"
    if len(x) == 1:
        return x[0]
    return x


class ScalarFunction:
    """
    A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)

    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        ctx = Context(False)
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), f"Expected return type float got {type(c)}"

        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    "Addition function $f(x, y) = x + y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        return d_output, d_output


class Mul(ScalarFunction):
    "Multiplication function $f(x, y) = x * y$"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        ctx.save_for_backward(a, b)
        return operators.mul(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        a, b = ctx.saved_values
        return d_output * b, d_output * a


class Inv(ScalarFunction):
    "Inverse function $f(x) = 1 / x$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return -1 * operators.inv(a) ** 2 * d_output


class Neg(ScalarFunction):
    "Negation function $f(x) = -x$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        return operators.neg(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        return -1 * d_output


class Sigmoid(ScalarFunction):
    "Sigmoid function $f(x) = 1 / (1 + exp(-x))$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        sig = operators.sigmoid(a)
        ctx.save_for_backward(sig)
        return sig

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (sig,) = ctx.saved_values
        return sig * (1 - sig) * d_output


class ReLU(ScalarFunction):
    "ReLU function $f(x) = max(0, x)$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (a,) = ctx.saved_values
        return d_output if a > 0 else 0


class Exp(ScalarFunction):
    "Exp function $f(x) = exp(x)$"

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        result = operators.exp(a)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        (result,) = ctx.saved_values
        return result * d_output


class LT(ScalarFunction):
    "Less-than function $f(x, y) =$ 1.0 if x < y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return 0.0, 0.0  # Gradient is not defined for less-than operator


class EQ(ScalarFunction):
    "Equal function $f(x, y) =$ 1.0 if x == y else 0.0"

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        return 0.0, 0.0  # Gradient is not defined for equality operator
