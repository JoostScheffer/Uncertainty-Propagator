from uncertainty_propagator import Propagator
import numpy as np

# Define the equation
# The propagator will automatically create the variables for you
prop = Propagator("I_0 * cos(theta + theta_0) + I_background")

# We can use the `set_variables` method to set the value and error of multiple variables at once
prop.set_variables(
    {
        "theta": 0,
        "theta_0": 0,
        "I_background": 0.1,
    }
)

# Evaluate the function
y = prop.evaluate_function()

# Evaluate the error
y_err = prop.evaluate_error_function()