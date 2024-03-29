import inspect
from collections.abc import Iterable
from datetime import datetime

import ipywidgets as widgets
from ipywidgets.widgets.widget import Widget
import sympy as sp
from IPython.display import display
from tqdm.notebook import tqdm
from uncertainties import ufloat as uf
from uncertainties.core import Variable as ufVar
from .propagator import Propagator

blue = "#6382f2"
red = "#ff9696"
gray = "#f3f4f5"



class VariableInputRow(widgets.HBox):
    """A class that represents a row of widgets for a variable.

    Attributes
    ----------
    * symbol (sp.Symbol): The symbol representing the variable.
    * post_update (callable): A function to be called after updating the variable.
    * n (float): The nominal value of the variable.
    * s (float): The standard deviation of the variable.
    * format_specifier (str): The format specifier for displaying the variable value.
    * valid (bool): Indicates if the variable input is valid.
    * n_is_valid (bool): Indicates if the nominal value input is valid.
    * s_is_valid (bool): Indicates if the standard deviation input is valid.
    * n_is_iter (bool): Indicates if the nominal value is an iterable.
    * n_is_uf (bool): Indicates if the nominal value is an instance of ufVar.
    * s_is_iter (bool): Indicates if the standard deviation is an iterable.

    Methods
    -------
    * update(): Updates the variable.
    * _update_self(_change=None): Updates the variable and its visuals.
    * _process_n(): Processes the nominal value input.
    * _process_s(): Processes the standard deviation input.
    * _process_and_visualize(): Processes the inputs and visualizes the variable.

    Notes
    -----
    goals:
    * manage a row of widgets for a variable
    * manage the visuals, the logic and the variable itself
    """

    def __init__(
        self,
        symbol: sp.Symbol,
        post_update: callable = lambda: None,
        n: float = 1,
        s: float = 0.1,
        format_specifier: str = ".2ueP",
    ) -> ...:
        """Initializes the VariableInputRow.

        Parameters
        ----------
        symbol : sp.Symbol
            The symbol representing the variable.
        post_update : callable (optional)
            A function to be called after updating the variable. Defaults to an empty function.
        n : float (optional)
            The nominal value of the variable. Defaults to 1.
        s : float (optional)
            The standard deviation of the variable. Defaults to 0.1.
        format_specifier : str (optional)
            The format specifier for displaying the variable value. Defaults to ".2ueP".
        """
        self.symbol: sp.Symbol = symbol
        self.name: str = self.symbol.name
        self.n: float = n
        self.s: float = s
        self.uf: ufVar | list[ufVar] = uf(self.n, self.s, self.name)
        self.format_specifier: str = format_specifier

        self.widgets: dict[str:Widget] = {
            "name": widgets.Label(self.name),
            "value": widgets.Label(f"{self.uf:.2ueP}"),
            "n": widgets.Text(str(self.uf.n)),
            "s": widgets.Text(str(self.uf.s)),
            "n-state": widgets.Label("const"),
            "s-state": widgets.Label("const"),
        }

        # when the user edits the input, run the update function
        self.widgets["n"].observe(lambda change: self._update_self(change), "value")
        self.widgets["s"].observe(lambda change: self._update_self(change), "value")

        # run the update function once
        self._update_self(None)

        self.widgets_list: list[widgets.widget.Widget] = [
            x for x in self.widgets.values() if isinstance(x, Widget)
        ]

        self.post_update = post_update

        self.valid: bool = True
        self.n_is_valid: bool = True
        self.s_is_valid: bool = True
        self.n_is_iter: bool = False
        self.n_is_uf: bool = False
        self.s_is_iter: bool = False

        super().__init__(self.widgets_list)

    def update(self) -> ...:
        """Updates the variable."""
        self._update_self()

    def _update_self(self, _change=None) -> ...:
        """Updates the variable and its visuals.

        This is a support function and should not be called directly.
        """
        # set default values
        self.widgets["n"].style.background = None
        self.widgets["s"].set_trait("disabled", False)
        self.widgets["s"].style.background = None
        self.widgets["n-state"].value = "-----"
        self.widgets["s-state"].value = "-----"

        # assume the input is invalid
        self.valid = False
        self.n_is_valid = False
        self.s_is_valid = False

        self.n_is_iter = False
        self.n_is_uf = False
        self.s_is_iter = False
        self.s_is_func = False

        self._process_n()
        self._process_s()
        self._process_and_visualize()

        self.post_update()

    def _process_n(self) -> ...:
        """Processes the nominal value input.

        This is a support function and should not be called directly.
        """
        # if the input has 0 length
        if len(self.widgets["n"].value) == 0:
            self.widgets["n-state"].value = "field is empty"

            self.n_is_valid = False
            return

        # try to evaluate the input
        try:
            val = eval(self.widgets["n"].value)

        except (NameError, SyntaxError, TypeError) as e:
            self.widgets["n-state"].value = f"{type(e).__name__}"

            self.n_is_valid = False
            return
        except Exception as e:
            self.widgets["n-state"].value = f"{type(e).__name__}"

            self.n_is_valid = False

            print("-------------")
            print("An unexpected error occured")
            print(e)
            print("-------------")
            return

        # if the input is a uf, disable the s field
        if isinstance(val, ufVar):
            # check if the value is a uf, if so disable the uncertainty

            self.widgets["n-state"].value = "uf"
            self.widgets["s-state"].value = "uf"

            self.n = val.n
            self.s = val.s

            self.n_is_uf = True
            self.n_is_valid = True
            self.s_is_valid = True

        elif isinstance(val, float | int):
            self.widgets["n-state"].value = f"{type(val).__name__}"

            self.n = val

            self.n_is_valid = True

        elif isinstance(val, Iterable):
            if all([isinstance(x, ufVar) for x in val]):
                self.widgets["n-state"].value = f"iter (len: {len(val)})"
                self.widgets["s-state"].value = f"iter (len: {len(val)})"

                self.n = [x.n for x in val]
                self.s = [x.s for x in val]

                self.n_is_valid = True
                self.s_is_valid = True

                self.n_is_iter = True
                self.s_is_iter = True
                return
            elif all([isinstance(x, float | int) for x in val]):
                self.n = val

                self.widgets["n-state"].value = f"iter (len: {len(val)})"

                self.n_is_iter = True
                self.n_is_valid = True
        else:
            self.n_is_valid = False

    def _process_s(self) -> ...:
        """Processes the standard deviation input.

        This is a support function and should not be called directly.
        """
        # if s is already set to valid it does not need to be checked
        # s can be valid because it was define by something in the n-field
        if self.s_is_valid:
            return

        # if the input has 0 length, set the background to red and return
        if len(self.widgets["s"].value) == 0:
            self.widgets["s-state"].value = "field is empty"

            self.s_is_valid = False
            return

        # try to evaluate the input
        try:
            val = eval(self.widgets["s"].value)

        except (NameError, SyntaxError, TypeError) as e:
            self.widgets["s-state"].value = f"{type(e).__name__}"

            self.s_is_valid = False
            return
        except Exception as e:
            self.widgets["s-state"].value = f"{type(e).__name__}"

            self.s_is_valid = False

            print("-------------")
            print("An unexpected error occured")
            print(e)
            print("-------------")
            return

        if isinstance(val, float | int):
            if val < 0:
                self.s_is_valid = False
                self.widgets["s-state"].value = "negative value"
                self.valid = False
                return
            elif val == 0:
                self.widgets["s-state"].value = "exact 0"
            # else:
            # self.widgets["s-state"].value = "const"

            self.widgets["s-state"].value = f"{type(val).__name__}"
            self.s_is_valid = True
            self.s = val

        elif isinstance(val, Iterable):
            if any([x < 0 for x in val]):
                self.widgets["s-state"].value = "itterable with negative value"

                self.s_is_valid = False
                return

            self.s = val
            self.widgets["s-state"].value = f"iter (len: {len(val)})"

            self.s_is_iter = True
            self.s_is_valid = True

        elif inspect.isfunction(val):
            # check how many arguments the function takes
            # the function may only take one argument
            # todo check if the function returns a float or int
            # todo check if the input parameter is of type float, int, Iterable
            # opmerking: ik zou niet eens weten hoe ik dit zou moeten doen
            if len(inspect.signature(val).parameters) != 1:
                self.widgets["s-state"].value = "function, too many parameters"

                self.s_is_valid = False
                return

            self.widgets["s-state"].value = "func"
            self.s_is_func = True
            self.s_is_valid = True

            self.s = val
        else:
            self.s_is_valid = False

    def _process_and_visualize(self) -> ...:
        """Processes the inputs and visualizes the variable.

        This is a support function and should not be called directly.

        - n is a uf
        - n and s are constants
        - n is an iterable and s is constant
        - n is an iterable and s is itterable
        - n is an iterable and s is a function
        """
        self.valid = self.n_is_valid and self.s_is_valid

        if not self.n_is_valid:
            self.widgets["n"].style.background = red
        if not self.s_is_valid:
            self.widgets["s"].style.background = red

        if not self.valid:
            self.widgets["value"].value = "..."
            # TODO not sure if this is the best way to do this
            return

        # if they are both valid and are none
        if self.valid and (self.n is None or self.s is None):
            raise ValueError(
                "n or s is None but was determined to be valid, this should not happen"
            )

        if self.n_is_uf:
            self.widgets["value"].value = f"{self.uf:{self.format_specifier}}"

            self.widgets["s"].set_trait("disabled", True)
            self.widgets["s"].style.background = blue

            self.uf = self.n

        elif not (self.n_is_iter or self.s_is_iter or self.s_is_func):
            # n and s are constants
            self.uf = uf(self.n, self.s, self.name)
            self.widgets["value"].value = f"{self.uf:{self.format_specifier}}"

        elif self.n_is_iter and self.s_is_iter:
            # check if lengths match
            if len(self.n) != len(self.s):
                self.widgets["value"].value = "XXX"
                self.widgets["n"].style.background = red
                self.widgets["s"].style.background = red
                self.valid = False
                return

            self.widgets["value"].value = "[...]"

            self.uf = [uf(n, s, self.name) for n, s in zip(self.n, self.s)]

        # n is iterable and s is const
        elif self.n_is_iter and not (self.s_is_func or self.s_is_iter):
            self.widgets["value"].value = "[...]"
            self.uf = [uf(n, self.s, self.name) for n in self.n]

        elif self.n_is_iter and self.s_is_func:
            self.widgets["value"].value = "[...]"

            self.uf = [uf(n, self.s(n), self.name) for n in self.n]

        elif self.s_is_func:
            self.uf = uf(self.n, self.s(self.n), self.name)
            self.widgets["value"].value = f"{self.uf:{self.format_specifier}}"

        else:
            raise ValueError("This should not happen")


class PropagatorWidget(Propagator):
    """A class that represents a propagator for uncertainty propagation.

    Attributes
    ----------
    func (sp.Expr | str): The mathematical function to propagate uncertainty for.
    defaults (dict[sp.Symbol : ufVar | list[ufVar]]): Default values for the symbols in the function.

    Methods
    -------
    __init__(self, func: sp.Expr | str, defaults: dict[sp.Symbol : ufVar | list[ufVar]] = {}) -> None:
        Initializes the Propagator object.
    get_vars(self) -> list[dict[sp.Symbol : ufVar]]:
        Returns a list of dictionaries containing the symbols and their corresponding values.
    evaluate_function(self) -> list[float]:
        Evaluates the function for each set of symbol values and returns the results.
    evaluate_error_function(self) -> list[float]:
        Evaluates the error function for each set of symbol values and returns the results.
    update_rows(self, _change=None) -> None:
        Updates the input rows with the current values.
    write_to_output(self, text: str) -> None:
        Writes the given text to the output widget.
    validate(self) -> None:
        Validates the inputs and updates the state of the evaluate button.
    """

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
        super().__init__(func, defaults)
        self.evaluate_btn = widgets.Button(
            description="Evaluate",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Evaluate",
            icon="check",  # (FontAwesome names without the `fa-` prefix)
        )
        # if pressed call the update_rows function
        self.update_btn = widgets.Button(
            description="Update",
            disabled=False,
            button_style="info",  # 'success', 'info', 'warning', 'danger' or ''
            tooltip="Update fields with the current values",
            icon="",  # (FontAwesome names without the `fa-` prefix)
        )
        self.update_btn.on_click(self.update_rows)

        self.rows: list[VariableInputRow] = []
        for symbol in self.symbols:
            if symbol in defaults:
                self.rows.append(
                    VariableInputRow(
                        symbol,
                        self.validate,
                        n=defaults[symbol].n,
                        s=defaults[symbol].s,
                    )
                )
            else:
                self.rows.append(VariableInputRow(symbol, self.validate))

        self.output = widgets.Output(
            layout={"border": "1px solid black", "padding": "0.25em"}
        )
        self.write_to_output("Status messages will appear here")

        hbox = widgets.HBox([self.evaluate_btn, self.update_btn])
        self.widgets = [hbox, self.output]

        self.box = widgets.VBox(self.rows + self.widgets)
        display(self.function)
        display(self.error_function)
        display(self.box)

    
    def update_rows(self, _change=None) -> ...:
        """Updates the input rows with the current values."""
        # update the rows
        super().update()

        self.write_to_output("updated")

    def write_to_output(self, text: str) -> ...:
        """Writes the given text to the output widget."""
        with self.output:
            print(datetime.now().strftime("%H:%M:%S"), text)

    def validate(self) -> ...:
        """
        Validates the inputs and updates the state of the evaluate button accordingly.

        If all inputs are valid and have the same length, the evaluate button is enabled and
        styled as success. Otherwise, the evaluate button is disabled and styled as danger.
        """
        try:
            everything_is_valid = super().validate()
        except ValueError as e:
            everything_is_valid = False
            self.write_to_output(str(e))

        if not everything_is_valid:
            self.evaluate_btn.disabled = True
            self.evaluate_btn.button_style = "danger"
            self.evaluate_btn.icon = "exclamation"
        else:
            self.evaluate_btn.disabled = False
            self.evaluate_btn.button_style = "success"
            self.evaluate_btn.icon = "check"
