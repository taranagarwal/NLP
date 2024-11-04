import matplotlib.pyplot as plt

err_rates = [10.05, 12.55, 7.53, 2.51, 2.09, 1.67]
size_K = [1, 2, 5, 10, 20, 50]

plt.figure(figsize=(8, 6))
plt.scatter(size_K, err_rates, color='blue', s=100)

plt.title('Error Rates vs. Size')
plt.xlabel('Size (K)')
plt.ylabel('Error Rate (%)')

plt.grid(True)
plt.show()