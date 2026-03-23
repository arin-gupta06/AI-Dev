import matplotlib.pyplot as  plt

#Basic chart
#Method1:
# x = [1, 3, 4, 5]
# y = [5, 45, 60,75]
# plt.plot(x, y)
# plt.show()

#Method2:
# plt.plot([1,2,3,4],[10,20,30,40], label="Study")
# plt.title("Line graph")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.legend()
# plt.show()

#Bar Chart
# classes = ["Blue", "Red", "Yellow"]
# y = [10, 30, 50]
# plt.bar(classes, y, color=["blue", "red", "yellow"])
# plt.title("Bar Chart")
# plt.show()

#Histogram
data = [1,2,2,3,4,5,3,4,3,2,4,1]
plt.hist(data, bins = 4, color = "green", edgecolor = "black")
plt.title("Histogram")
plt.show()