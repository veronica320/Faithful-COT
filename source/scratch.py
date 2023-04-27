from sympy import Symbol, Eq, simplify, solve
import math
from math import *
def lcm(a,b): # for backwards compatibility with Python<3.9
	return (a * b) // math.gcd(a,b)
def solve_it(equation, variable):
	solution = solve(equation, variable, dict=True)
	if not solution:
		if isinstance(variable, list):
			solution = {v: None for v in variable}
		else:
			solution = {variable: None}
		return solution
	else:
		solution = solution[0]
		return solution
original_price = Symbol('original_price', positive=True)
discounted_price = original_price * 0.78
coupon_discount = 20
final_price = discounted_price - coupon_discount
solution = solve_it(final_price - 0.5 * original_price - 1.9, original_price)
answer = solution[original_price]
print(answer)