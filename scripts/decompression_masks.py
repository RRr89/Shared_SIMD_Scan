import math

compression = int(input("compression factor: "))

pattern = "{:>12} {:>12} {:>20} {:>20}"
print(pattern.format("index", "first byte", "offset from load", "padding"))
print(pattern.format("", "", "(shuffle mask)", "(shift mask)"))
for i in range(0, 32):
	# global index of the first byte
	first_byte = 4 * i * compression // 8

	shuffle_mask = [
		(compression * (i*4 + 0) // 8) - first_byte,
		(compression * (i*4 + 1) // 8) - first_byte,
		(compression * (i*4 + 2) // 8) - first_byte,
		(compression * (i*4 + 3) // 8) - first_byte
	]

	shifting_mask = [
		compression * (i*4 + 0) % 8,
		compression * (i*4 + 1) % 8,
		compression * (i*4 + 2) % 8,
		compression * (i*4 + 3) % 8
	]

	mask_pattern = "({} {} {} {})"
	print(pattern.format(i*4, first_byte, mask_pattern.format(*shuffle_mask), mask_pattern.format(*shifting_mask)))