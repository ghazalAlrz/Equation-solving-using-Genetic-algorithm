import numpy as np
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Constants
INITIAL_LOW = 0.0
INITIAL_HIGH = 10000.0
TOP_SELECTION = 100
MULTIPLICATION_FACTOR = 100


class GeneticEquationSolver:
    """
    A solver for finding the solution to a nonlinear equation using a genetic algorithm.

    Attributes:
        coefficients (tuple): The coefficients of the nonlinear equation.
        accuracy (float): The threshold for determining if the solution is accurate enough.
        population_size (int): The number of individuals in the population.
        mutation_range (tuple): The range within which mutations occur.
        generation_number (int): The current generation number in the genetic algorithm.
    """

    def __init__(
        self,
        coefficients: Tuple[float, float, float, float],
        accuracy: float = 1e-6,
        population_size: int = 1000,
        mutation_range: Tuple[float, float] = (0.9, 1.1),
        initial_low: float = INITIAL_LOW,
        initial_high: float = INITIAL_HIGH,
    ):
        """
        Initializes the GeneticEquationSolver with the given parameters.

        Args:
            coefficients (tuple): The coefficients of the nonlinear equation.
            accuracy (float): The threshold for determining if the solution is accurate enough.
            population_size (int): The number of individuals in the population.
            mutation_range (tuple): The range within which mutations occur.
            initial_low (float): The lower bound for initial population values.
            initial_high (float): The upper bound for initial population values.
        """
        self.coefficients = coefficients
        self.population_size = population_size
        self.mutation_range = mutation_range
        self.accuracy = accuracy
        self.generation_number = 1
        self.initial_low = initial_low
        self.initial_high = initial_high

    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluates the given population based on the nonlinear equation defined by the coefficients.

        Args:
            population (numpy.ndarray): The population to evaluate.

        Returns:
            numpy.ndarray: The values of the equation for each individual in the population.
        """
        x, y, z = population[:, 0], population[:, 1], population[:, 2]
        return (
            self.coefficients[0] * x
            + self.coefficients[1] * y**2
            + self.coefficients[2] * z**3
            - self.coefficients[3]
        )

    def generate_initial_population(self) -> np.ndarray:
        """
        Generates the initial population of solutions with random values.

        Returns:
            numpy.ndarray: The initial population.
        """
        return np.random.uniform(
            low=self.initial_low, high=self.initial_high, size=(self.population_size, 3)
        )

    def mutate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Applies mutations to the population within the defined mutation range.

        Args:
            population (numpy.ndarray): The population to mutate.

        Returns:
            numpy.ndarray: The mutated population.
        """
        mutation_factors = np.random.uniform(
            low=self.mutation_range[0],
            high=self.mutation_range[1],
            size=population.shape,
        )
        return population * mutation_factors

    def solve(self) -> np.ndarray:
        """
        Executes the genetic algorithm to find the solution to the equation.

        Returns:
            numpy.ndarray: The best solution found by the algorithm.
        """
        population = self.generate_initial_population()
        while True:
            equation_values = np.abs(self.evaluate_population(population))
            sorted_indices = np.argsort(equation_values)
            best_equation_values = equation_values[sorted_indices[:TOP_SELECTION]]
            best_population = population[sorted_indices[:TOP_SELECTION]]

            logging.info(f"Generation number: {self.generation_number}")
            logging.info(f"Best value (lower is better): {best_equation_values[0]}")
            self.generation_number += 1

            if best_equation_values[0] <= self.accuracy:
                return best_population[0]

            new_population = np.vstack([best_population] * MULTIPLICATION_FACTOR)
            population = self.mutate_population(new_population)


if __name__ == "__main__":
    # Example usage
    coefficients = (7, 4, 6, 10)
    solver = GeneticEquationSolver(coefficients)

    # Print configurations
    logging.info(f"Coefficients: {coefficients}")
    logging.info(f"Population size: {solver.population_size}")
    logging.info(f"Mutation range: {solver.mutation_range}")
    logging.info(f"Initial range: ({solver.initial_low}, {solver.initial_high})")
    logging.info(f"Accuracy: {solver.accuracy}")

    # Execute genetic algorithm to find the solution
    solution = solver.solve()
    print(f"\nResult X: {solution[0]}  Y: {solution[1]}  Z: {solution[2]}")
