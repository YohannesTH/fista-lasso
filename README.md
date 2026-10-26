# ISTA and FISTA for Lasso Regression

A from-scratch NumPy implementation of ISTA and FISTA for the Lasso objective

\[
\min_\beta \frac{1}{2n}\|X\beta-y\|_2^2+\lambda\|\beta\|_1.
\]

The project includes the gradient, soft-thresholding proximal operator, synthetic sparse-data generation, convergence plots, tests, and a LaTeX report template.

## Requirements

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows
python -m pip install -r requirements.txt
```

## Run the experiments

From the project root:

```bash
python main.py
```

Figures and a CSV summary are saved under `results/`.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Compile the report

With a LaTeX distribution installed:

```bash
cd report
pdflatex main.tex
pdflatex main.tex
```

The report is a template: insert the generated figures and results after running the experiments.

## Implementation notes

- The algorithms are implemented directly using NumPy; no Lasso solver from a machine-learning package is used.
- The step size is `1 / L`, where `L = ||X||_2^2 / n`.
- The objective-gap plot uses the best objective value found by either run as a practical reference, not a certified exact optimum. For publication-quality experiments, use a high-accuracy reference solution and report the reference method and stopping criterion.
- Standard FISTA's objective values need not decrease at every iteration. Monotone FISTA is a different variant and is not silently substituted here.
