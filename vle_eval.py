import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# datasets to be used
studentVle = pd.read_csv("studentVle.csv",  delimiter=',', index_col=False, na_values='?')
vle = pd.read_csv("vle.csv",  delimiter=',', index_col=False, na_values='?')
studentInfo = pd.read_csv("studentInfo.csv",  delimiter=',', index_col=False, na_values='?')


vle_merged = studentVle.merge(vle, on='id_site').merge(studentInfo, on='id_student')
vle_average = vle_merged[['code_module_x', 'activity_type', 'final_result', 'sum_click']].\
    groupby(['code_module_x', 'activity_type', 'final_result'])\
    .mean().reset_index()


for course in vle_average['code_module_x'].unique():
    d1 = vle_average[(vle_average['code_module_x'] == course) &
                     (vle_average['final_result'] == 'Fail')]['activity_type']
    n1 = range(len(d1))
    plt.bar(n1, vle_average[(vle_average['code_module_x'] == course) &
            (vle_average['final_result'] == 'Fail')]['sum_click'], color=(0.4, 0.1, 0.9),align='center')
    plt.xticks(n1, d1, rotation=45, fontsize=20)
    plt.title(course + '  Fail', fontsize=30)
    plt.show()

    d2 = vle_average[(vle_average['code_module_x'] == course) & (vle_average['final_result'] == 'Pass')]['activity_type']
    n2 = range(len(d2))
    plt.bar(n2, vle_average[(vle_average['code_module_x'] == course) &
                            (vle_average['final_result'] == 'Pass')]['sum_click'], color=(0.4, 0.4, 0.9),align='center')

    plt.xticks(n2, d2, rotation=45, fontsize=20)
    plt.title(course + '  Pass', fontsize=30)
    plt.show()

    d3 = vle_average[(vle_average['code_module_x'] == course) &
                     (vle_average['final_result'] == 'Withdrawn')]['activity_type']

    n3 = range(len(d3))
    plt.bar(n3, vle_average[(vle_average['code_module_x'] == course) &
                            (vle_average['final_result'] == 'Withdrawn')]['sum_click'], color=(0.4, 0.7, 0.9),align='center')

    plt.xticks(n3, d3, rotation=45, fontsize=20)
    plt.title(course + ' Withdrawn', fontsize=30)
    plt.show()

    d4 = vle_average[(vle_average['code_module_x'] == course) &
                     (vle_average['final_result'] == 'Distinction')]['activity_type']
    n4 = range(len(d4))
    plt.bar(n4, vle_average[(vle_average['code_module_x'] == course) &
                            (vle_average['final_result'] == 'Distinction')]['sum_click'],  color=(0.4, 0.9, 0.9), align='center')

    plt.xticks(n4, d4, rotation=45, fontsize=20)
    plt.title(course + ' Distinction', fontsize=30)
    plt.show()




