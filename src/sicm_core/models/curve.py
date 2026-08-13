from dataclasses import dataclass, field
from typing import Mapping, Sequence

@dataclass(frozen=True, slots=True)
class Curve:
    curve_id: str
    name: str
    x_variable: str
    y_variable: str
    x_values: Sequence[float]
    y_values: Sequence[float]
    equation_id: str | None = None
    equilibrium_x: float | None = None
    equilibrium_y: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.curve_id.strip() or not self.name.strip():
            raise ValueError("curve_id and name are required.")
        if len(self.x_values) != len(self.y_values):
            raise ValueError("x_values and y_values must have equal length.")
        if not self.x_values:
            raise ValueError("A curve requires at least one point.")

    def to_dict(self) -> dict:
        return {"curve_id": self.curve_id, "name": self.name,
                "x_variable": self.x_variable, "y_variable": self.y_variable,
                "x_values": list(self.x_values), "y_values": list(self.y_values),
                "equation_id": self.equation_id,
                "equilibrium_x": self.equilibrium_x,
                "equilibrium_y": self.equilibrium_y,
                "metadata": dict(self.metadata)}
