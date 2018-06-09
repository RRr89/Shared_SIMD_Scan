import math

compression = int(input("compression factor: "))

pattern = "{:>12} {:>12} {:>12}"
print(pattern.format("index", "input offset", "padding"))
for i in range(0, 32):
	partial = i * compression
	first_byte = math.floor(partial / 8)
	padding = partial % 8
	print(pattern.format(i, first_byte, padding))