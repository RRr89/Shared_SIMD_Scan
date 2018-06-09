n = int(input("compression factor: "))
max_index = int(input("max_index: "))

print()

k = 0
for i in range(0, max_index//128 + 1):
	print("i = {}".format(i))
	for j in range(0, 16):
		print("    parallel_load b_a from input[{}]".format(k*16 + j*n))
		print("    shuffle b_a to c_a using shuffle_mask(...)")
		print("    parallel_shift c_a by (...)")
		print("    parallel_store c_a in output[{}]".format(i*16 + j*8))

		print("    parallel_load b_b from input[{}]".format(k*16 + j*n + n//2))
		print("    shuffle b_b to c_b using shuffle_mask(...)")
		print("    parallel_shift c_b by (...)")
		print("    parallel_store c_b in output[{}]".format(i*16 + j*8 + 4))
		print("    ---")
	k += n
	print("k = {}".format(k))
	print("---")