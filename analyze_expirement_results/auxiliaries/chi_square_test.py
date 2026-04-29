import numpy as np
from scipy.stats import chi2_contingency

WINTER_DEAD = 33
WINTER_ALIVE = 28
SUMMER_DEAD = 38
SUMMER_ALIVE = 41

contingency_table = np.array([
    [WINTER_DEAD, WINTER_ALIVE],
    [SUMMER_DEAD, SUMMER_ALIVE],
])

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("Contingency table (rows=season, cols=outcome):")
print(f"           Dead   Alive")
print(f"Winter:    {WINTER_DEAD:4d}    {WINTER_ALIVE:4d}")
print(f"Summer:    {SUMMER_DEAD:4d}    {SUMMER_ALIVE:4d}")
print()
print(f"Chi-square statistic : {chi2:.4f}")
print(f"Degrees of freedom   : {dof}")
print(f"P-value              : {p:.4f}")
print()
print("Expected frequencies under H0:")
print(f"           Dead      Alive")
print(f"Winter:    {expected[0, 0]:7.2f}   {expected[0, 1]:7.2f}")
print(f"Summer:    {expected[1, 0]:7.2f}   {expected[1, 1]:7.2f}")
