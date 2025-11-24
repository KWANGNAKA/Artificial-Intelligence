# Pattern Recognition - Probability Theory
# Two boxes: Red and Blue
# Fruits: Apple (a) and Orange (o)

# 1) PRIOR PROBABILITIES
# P(Box = red) and P(Box = blue)
p_red = 0.40      # โอกาสหยิบจากกล่องแดง 40%
p_blue = 0.60     # โอกาสหยิบจากกล่องน้ำเงิน 60%

# 2) NUMBER OF FRUITS IN EACH BOX
# red box: 2 apples (green), 4 oranges (orange)
red_apple = 2
red_orange = 6

# blue box: 3 apples, 1 orange
blue_apple = 3
blue_orange = 1

# Total fruits in red box and blue box
total_red = red_apple + red_orange
total_blue = blue_apple + blue_orange

# 3) LIKELIHOOD: P(F | Box)

p_a_given_red = red_apple / total_red# P(F = apple | red box)
p_o_given_red = red_orange / total_red# P(F = orange | red box)

p_a_given_blue = blue_apple / total_blue# P(F = apple | blue box)
p_o_given_blue = blue_orange / total_blue# P(F = orange | blue box)

# 4) MARGINAL PROBABILITY: P(F)
# Law of total probability
# P(F = a) = P(a|red)*P(red) + P(a|blue)*P(blue)
p_a = p_a_given_red * p_red + p_a_given_blue * p_blue

# P(F = o) = P(o|red)*P(red) + P(o|blue)*P(blue)
p_o = p_o_given_red * p_red + p_o_given_blue * p_blue

# 5) POSTERIOR: P(red | orange)
# Bayes' rule:
# P(red | o) = P(o|red) * P(red) / P(o)
p_red_given_o = p_o_given_red * p_red / p_o

# 6) PRINT RESULTS
print("=== Given information ===")
print(f"P(Box = red)   = {p_red:.2f}")
print(f"P(Box = blue)  = {p_blue:.2f}")
print(f"Red box  : {red_apple} apples, {red_orange} oranges")
print(f"Blue box : {blue_apple} apples, {blue_orange} oranges")
print()

print("=== Marginal probabilities of fruit ===")
print(f"P(F = apple)  = {p_a:.4f}  ({p_a*100:.2f}%)")
print(f"P(F = orange) = {p_o:.4f}  ({p_o*100:.2f}%)")
print()

print("=== Posterior probability ===")
print(f"P(Box = red | F = orange) = {p_red_given_o:.4f}  ({p_red_given_o*100:.2f}%)")