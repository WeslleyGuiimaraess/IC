import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

y = [2.0, 4.0, 6.0, 3.0, 2.0, 6.0, 7.0, 1.0, 7.0, 2.0, 1.0, 3.0, 4.0, 3.0, 6.0, 6.0, 7.0, 10.0, 17.0, 12.0, 12.0, 14.0, 17.0, 13.0, 20.0, 26.0, 27.0, 24.0, 27.0, 27.0]

# plotting the points  
plt.plot(x, y) 
    
# naming the x axis 
plt.xlabel('Episódios') 
# naming the y axis 
plt.ylabel('Total da recompensa') 
    
# function to show the plot 
plt.show() 