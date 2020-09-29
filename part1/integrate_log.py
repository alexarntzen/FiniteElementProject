import sympy as sy
x, y = sy.symbols("x y")

def get_log_integral():
    return (float(sy.integrate(sy.log(x + y), (y, sy.Rational(1,2)*x-sy.Rational(1, 2), x-1), (x, 1, 3 ))))

