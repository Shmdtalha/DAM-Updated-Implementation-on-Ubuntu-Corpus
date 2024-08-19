def calculate_top1_precision(file_path):
	with open(file_path, 'r') as file:
		lines = file.readlines()

	data = [(float(line.split()[0]), int(line.split()[1])) for line in lines]

	total_contexts = 0
	correct_top1 = 0

	for i in range(0, len(data), 10):
		context = data[i : i+10]
		_, max_valid = max(context, key=lambda x: x[0])
		total_contexts += 1
		if max_valid == 1:
			correct_top1 += 1

	# Calculate Top-1 Precision
	top1_precision = correct_top1 / total_contexts
	return top1_precision

precision = calculate_top1_precision('output/ubuntu/temp/score.test'  )
print("Top-1 Precision: {}".format(precision))
