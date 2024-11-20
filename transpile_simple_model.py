#!/usr/bin/env python3

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
import joblib
import numpy as np
import numpy.typing
import sys


def generate_float_array(name: str, values: numpy.typing.NDArray) -> str:
    floats = values.tolist()
    floats = map(str, floats)
    floats = map(lambda value: f"{value}F", floats)
    floats = ", ".join(list(floats))
    return f"""
        const float {name}[] = {{ {floats} }};
    """


def generate_linear_regression(model: LinearRegression) -> str:
    thetas = generate_float_array("thetas", model.coef_)
    return f"""
        {thetas}

        float prediction(const float features[], int n_parameters) {{
            float sum = thetas[0];
            for (int param = 1; param < n_parameters; ++param)
                sum += thetas[param] * features[param - 1];
            return sum;
        }}
    """


def generate_logistic_regression(model: LogisticRegression) -> str:
    thetas = generate_float_array("thetas", model.coef_)
    return f"""
        {thetas}

        float linear_regression_prediction(const float features[], int n_parameters) {{
            float sum = thetas[0];
            for (int param = 1; param < n_parameters; ++param)
                sum += thetas[param] * features[param - 1];
            return sum;
        }}

        float exp_approx(float x, int n_term) {{
            float result = 1.0F;
            int factorial = 1;
            for (int term = 1; term <= n_term; ++term) {{
                result += x / factorial;
                x *= x;
                factorial *= (1 + term);
            }}
            return result;
        }}

        float sigmoid(float x) {{
            int exp_x = exp_approx(x, 10);
            return exp_x / (1.0F + exp_x);
        }}

        float prediction(const float features[], int n_parameters) {{
            return sigmoid(linear_regression_prediction(features, n_parameters));
        }}
    """


def generate_model(model: BaseEstimator) -> str:
    if isinstance(model, LinearRegression):
        return generate_linear_regression(model)
    if isinstance(model, LogisticRegression):
        return generate_logistic_regression(model)
    raise RuntimeError("Unsupported model")


def generate_main(n_parameters: int) -> str:
    features = np.random.random(n_parameters)
    features = generate_float_array("features", features)
    return f"""
        #include <stdio.h>

        int main(void) {{
            {features}
            float output = prediction(features, {n_parameters});
            printf("%f\\n", output);
        }}
    """


def dump(code: str, path: str) -> None:
    with open(path, "w") as file:
        print(code, file=file)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: ./transpile_simple_model.py <joblib> <output>", file=sys.stderr)
        return 1
    model = joblib.load(sys.argv[1])
    n_parameters = len(model.coef_)
    code = generate_model(model) + generate_main(n_parameters)
    dump(code, sys.argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
