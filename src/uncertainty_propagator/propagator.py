import sympy as sp
from tqdm.notebook import tqdm
from uncertainties.core import Variable as ufVar



def symbol_to_error_symbol(symbol: sp.Symbol) -> sp.Symbol:
    """Converts a symbol to its corresponding error symbol.

    Parameters
    ----------
    symbol (sp.Symbol):
        The symbol to convert.

    Returns
    -------
    sp.Symbol:
        The error symbol corresponding to the input symbol.
    """
    return sp.Symbol(f"\sigma_{symbol}")


def generate_error_propagation_function(function: sp.Expr) -> sp.Expr:
    """Generates an error propagation function for a given mathematical expression.

    Uses the formula:
    $$
    \sqrt{\left(\frac{\partial f}{\partial x_1} \cdot \sigma_{x_1}\right)^2 + \left(\frac{\partial f}{\partial x_2} \cdot \sigma_{x_2}\right)^2 + \ldots}
    $$
    Which assumes uncorrelated errors.

    Parameters
    ----------
    function: sp.Expr
        The mathematical expression for which the error propagation function is generated.

    Returns
    -------
    sp.Expr:
        The error propagation function.
    """
    symbols = list(function.free_symbols)
    error_symbols = [symbol_to_error_symbol(symbol) for symbol in symbols]

    error_function = sp.sqrt(
        sum(
            [
                (sp.diff(function, symbol) * error_symbol) ** 2
                for symbol, error_symbol in zip(symbols, error_symbols)
            ]
        )
    )

    return error_function


class Propagator:
    def __init__(
        self,
        func: sp.Expr | str,
        defaults: dict[sp.Symbol : ufVar | list[ufVar]] = {},
    ) -> ...:
        """Initialize the UncertaintyPropagationWidget.

        Parameters
        ----------
        func : sp.Expr or str
            The mathematical function to be evaluated. It can be either a sympy
            expression or a string representing the expression.
        defaults : dict[sp.Symbol : ufVar | list[ufVar]], optional
            A dictionary containing default values for the symbols in the function.
            The keys are sympy symbols and the values can be either ufVar objects
            or lists of ufVar objects.

        Raises
        ------
        ValueError:
            If the function does not contain any symbols or if the defaults contain
            symbols that are not in the function.
        """
        # input sanitation
        if isinstance(func, str):
            func = sp.sympify(func)

        # input sanitation
        if not func.free_symbols:
            raise ValueError("function does not contain any symbols")

        symbols = list(func.free_symbols)

        # input sanitation
        defaults_symbols = set(defaults.keys())
        if defaults_symbols - set(symbols):
            raise ValueError("defaults contains symbols that are not in symbols")

        self.symbols: list[sp.Symbol] = symbols
        self.function: sp.Expr = func
        self.error_function: sp.Expr = generate_error_propagation_function(
            self.function
        )

        # sort self.symbols by name
        self.symbols.sort(key=lambda x: x.name)

    def get_vars(self) -> list[dict[sp.Symbol : ufVar]]:
        """Get a list of dictionaries containing symbols and their corresponding uncertainty variables.

        Each dictionary in the list represents a row of symbols and uncertainty variables.
        If a row contains a list of uncertainty variables, multiple dictionaries will be created,
        each representing a different combination of values from the list.

        Returns
        -------
        list[dict[sp.Symbol : ufVar]]:
            A list of dictionaries containing symbols and uncertainty variables.
        """
        # for each row, call the get function which should return either a list of ufs or a single uf
        row_uf_dict = {row.symbol: row.uf for row in self.rows}
        row_values = list(row_uf_dict.values())
        row_contains_list = [isinstance(x, list) for x in row_values]

        current_dict: dict[sp.Symbol : ufVar]
        list_of_symbols_and_values: list[dict[sp.Symbol : ufVar]]

        if any(row_contains_list):
            list_lengths = {len(x) for x in row_values if isinstance(x, list)}
            if len(list_lengths) > 1:
                raise ValueError(
                    "not all inputs have the same length, AND this should have been caught earlier..."
                )

            if len(list_lengths) == 0:
                raise ValueError(
                    "the row was expected to contain a list, but apparently it does not"
                )

            list_length = list_lengths.pop()

            list_of_symbols_and_values = []
            for i in tqdm(range(list_length), "building replacement dicts"):
                current_dict = {}
                for symbol, uf_val in row_uf_dict.items():
                    error_symbol = symbol_to_error_symbol(symbol)
                    if isinstance(uf_val, list):
                        current_dict[symbol] = uf_val[i].n
                        current_dict[error_symbol] = uf_val[i].s
                    else:
                        current_dict[symbol] = uf_val.n
                        current_dict[error_symbol] = uf_val.s

                list_of_symbols_and_values.append(current_dict)

        else:
            current_dict = {}
            for symbol, uf_val in row_uf_dict.items():
                error_symbol = symbol_to_error_symbol(symbol)
                current_dict[symbol] = uf_val.n
                current_dict[error_symbol] = uf_val.s

            list_of_symbols_and_values = [current_dict]

        return list_of_symbols_and_values

    def evaluate_function(self) -> list[float]:
        """Evaluates the function using the provided variable values and returns a list of floats.

        Returns
        -------
        list[float]:
            The evaluated function values as a list of floats.

        Raises
        ------
        ValueError:
            If not all results are floats, indicating a possible type error in the function.
        """
        replacement_dict_list = self.get_vars()
        result = [
            self.function.evalf(subs=replacement_dict)
            for replacement_dict in tqdm(replacement_dict_list, "evaluating function")
        ]
        if not all([isinstance(x, sp.Float) for x in result]):
            raise ValueError(
                "not all results are floats, you might have a type in your function"
            )
        result = [float(x) for x in result]

        return result

    def evaluate_error_function(self) -> list[float]:
        """Evaluates the error function for a list of replacement dictionaries.

        Returns
        -------
        list[float]:
            A list of floats representing the evaluated error function for each replacement dictionary.

        Raises
        ------
        ValueError:
            If not all results are floats, indicating a possible type error in the function.
        """
        replacement_dict_list = self.get_vars()
        result = [
            self.error_function.evalf(subs=replacement_dict)
            for replacement_dict in tqdm(
                replacement_dict_list, "evaluating error function"
            )
        ]
        if not all([isinstance(x, sp.Float) for x in result]):
            raise ValueError(
                "not all results are floats, you might have a type in your function"
            )
        result = [float(x) for x in result]

        return result
    
    def update_rows(self, _change=None) -> ...:
        """Updates the input rows with the current values."""
        # update the rows
        for row in self.rows:
            row.update()

    def validate(self, raise_exceptions=True) -> ...:
        """
        Validates the inputs and updates the state of the evaluate button accordingly.

        If all inputs are valid and have the same length, the evaluate button is enabled and
        styled as success. Otherwise, the evaluate button is disabled and styled as danger.
        """
        everything_is_valid = False

        # check if all inputs are valid
        if not all(row.valid for row in self.rows):
            if raise_exceptions:
                raise ValueError("not all inputs report to be valid")
            everything_is_valid = False
        else:
            everything_is_valid = True

        # check if all inputs have the same length
        # for each row ask it to return its length
        lengths = {len(row.uf) for row in self.rows if isinstance(row.uf, list)}
        # ignore all literal constants
        if len(lengths) > 1:
            if raise_exceptions:
                raise ValueError("not all inputs have the same length")
            everything_is_valid = False

        if not everything_is_valid:
            if raise_exceptions:
                raise ValueError("not everything is valid")
            return False
        else:
            return True

