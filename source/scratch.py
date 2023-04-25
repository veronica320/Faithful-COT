from sympy import Symbol, Eq, simplify, solve
import math
from math import *

def lcm(a,b):
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

# 1. How much did Adi pay? (independent, support: [\"Adi bought a pencil for 35 cents\"])
paid = 0.35
# 2. How much is a one-dollar bill worth? (independent, support: [\"He paid with a one-dollar bill\"])
dollar_bill_worth = 1.0
# 3. How much change will Adi get? (depends on 1 and 2, support: [])
change = dollar_bill_worth - paid
# 4. Final Answer: How much change will he get? (depends on 3, support: [])
answer = change

print(answer)