import math

compression = int(input("compression factor: "))

pattern = "{:>12} {:>12} {:>20} {:>20}"
print(pattern.format("index", "first byte", "offset from load", "padding"))
print(pattern.format("", "", "(shuffle mask)", "(shift mask)"))
for i in range(0, 32):
	# global index of the first byte
	first_byte = 8 * i * compression // 8

	shuffle_mask = [
		(compression * (i*8 + 0) // 8) - first_byte,
		(compression * (i*8 + 1) // 8) - first_byte,
		(compression * (i*8 + 2) // 8) - first_byte,
		(compression * (i*8 + 3) // 8) - first_byte,

		(compression * (i*8 + 4) // 8) - first_byte,
		(compression * (i*8 + 5) // 8) - first_byte,
		(compression * (i*8 + 6) // 8) - first_byte,
		(compression * (i*8 + 7) // 8) - first_byte
	]

	shifting_mask = [
		compression * (i*8 + 0) % 8,
		compression * (i*8 + 1) % 8,
		compression * (i*8 + 2) % 8,
		compression * (i*8 + 3) % 8,

		compression * (i*8 + 4) % 8,
		compression * (i*8 + 5) % 8,
		compression * (i*8 + 6) % 8,
		compression * (i*8 + 7) % 8
	]

	mask_pattern = "({} {} {} {} {} {} {} {})"
	print(pattern.format(i*8, first_byte, mask_pattern.format(*shuffle_mask), mask_pattern.format(*shifting_mask)))