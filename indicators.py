from abc import ABC, abstractmethod
import numpy as np


class Indicator(ABC):
    def __init__(self, name, unit):
        self.name = name
        self.unit = unit

    @abstractmethod
    def process(self, data):
        """
        Process data values if needed.
        Data should already be in correct physical units.
        """
        pass


class TemperatureIndicator(Indicator):
    def __init__(self):
        super().__init__("Temperature", "°C")

    def process(self, data):
        # DataLoader already converts Kelvin → Celsius if needed
        return np.asarray(data, dtype=float)


class PrecipitationIndicator(Indicator):
    def __init__(self):
        super().__init__("Precipitation", "mm")

    def process(self, data):
        # DataLoader already converts meters → mm if needed
        return np.asarray(data, dtype=float)
