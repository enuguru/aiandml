import matplotlib.pyplot as plt
import numpy as np

# Data
years = np.arange(2012, 2020)
boys =   [110, 185, 240, 285, 305, 310, 315, 315]
girls =  [85, 175, 225, 295, 275, 315, 305, 320]

bar_width = 0.4
index = np.arange(len(years))

plt.figure(figsize=(10, 6))

# Bars
plt.bar(index, boys, bar_width, color='#FFA54F', label='Boys', edgecolor='black')
plt.bar(index + bar_width, girls, bar_width, color='#FFFFB3', label='Girls', edgecolor='black')

# Titles and labels
plt.title('Chart 5.2.2\nStudents who own a smartphone at Redwood School, by gender, 2012 to 2019', 
          loc='left', fontsize=12, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of students', fontsize=12)
plt.xticks(index + bar_width / 2, years)
plt.yticks(np.arange(0, 351, 50))

# Legend
plt.legend(loc='upper left', bbox_to_anchor=(0.25, -0.08), ncol=2, frameon=False)

# Layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('redwood_school_smartphones.png')
plt.close()