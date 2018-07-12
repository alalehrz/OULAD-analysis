import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

studentVle = pd.read_csv("studentVle.csv",  delimiter=',')
courses = pd.read_csv("courses.csv",  delimiter=',')
vle = pd.read_csv("vle.csv",  delimiter=',')
assessments = pd.read_csv("assessments.csv",  delimiter=',')
studentInfo = pd.read_csv("studentInfo.csv",  delimiter=',',na_values='?')
studentAssessment = pd.read_csv("studentAssessment.csv",  delimiter=',', na_values='?')
studentAssessment = studentAssessment.apply(pd.to_numeric, errors='coerce')

student_assess_merged = studentAssessment.merge(assessments, on='id_assessment')

# count of withdrawal between courses
withdrawal_count = studentInfo[studentInfo['final_result'] == 'Withdrawn'].groupby('code_module').count()\
    ['final_result'].sort_values()
wc = range(len(withdrawal_count.index))
plt.xticks(wc, withdrawal_count.index)
p1 = plt.bar(wc, withdrawal_count, align='center', color=(0.7, 0.2, 0.5))
plt.xlabel('Number of withdrawals per course')
plt.show()

# number of registration per course based on gender
course_reg_count = studentInfo[['code_module', 'gender']].groupby(['code_module', 'gender']).size()
print(course_reg_count)
crc = range(len(course_reg_count))
plt.xticks(crc, course_reg_count.index, rotation=45)
p2 = plt.bar(crc, course_reg_count, align='center',  color='rbrbrbrbrbrbrb')
plt.xlabel('Count of registration based on gender', fontsize=30)
plt.show()

# number of registration per course per location
course_reg_count = studentInfo[['code_module', 'age_band']].groupby(['code_module', 'age_band']).size()
print(course_reg_count)
crc = range(len(course_reg_count))
plt.xticks(crc, course_reg_count.index, rotation=45)
p3 = plt.bar(crc, course_reg_count, align='center',  color='g')
plt.xlabel('Count of registration based on age group', fontsize=30)
plt.show()

# number of registration per course per age group
course_reg_count = studentInfo[['code_module', 'region']].groupby(['code_module', 'region']).size()
print(course_reg_count)
crc = range(len(course_reg_count))
plt.xticks(crc, course_reg_count.index, rotation=90)
p4 = plt.bar(crc, course_reg_count, align='center',  color='rgbyrcm')
plt.xlabel('Count of registration based on location', fontsize=30)
plt.show()

# Rate of P, F, W, D of courses
course_pres_count = studentInfo[['code_module', 'code_presentation', 'final_result']].\
                 groupby(['code_module', 'code_presentation', 'final_result'], as_index=False)['final_result'].size()

rates = course_pres_count / course_pres_count.groupby(level=[0,1]).transform(sum)
cpc = range(len(rates))
plt.xticks(cpc, rates.index, rotation='vertical')
p5 = plt.bar(cpc, rates, align='center', color = 'gryb')
plt.xlabel('Rate of final results per course', fontsize=30)
plt.show()


# assessment score among students between different courses
score_comparison = student_assess_merged.groupby(['code_module'])['score'].mean().sort_values()
sc = range(len(score_comparison.index))
p6 = plt.bar(sc, score_comparison, align='center', color=(0.1, 0.2, 0.5, 0.3))
plt.xlabel('Mean of score', fontsize=20)
plt.xticks(sc, score_comparison.index)
plt.show()

