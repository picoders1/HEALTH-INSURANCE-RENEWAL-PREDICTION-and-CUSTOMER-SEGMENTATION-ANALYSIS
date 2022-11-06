import pandas as pd
import numpy as np

insurance = pd.read_csv(r"C:\Users\AKANSHA\Downloads\Final_Dataset (1).csv")
insurance.dtypes
#Graphical Representation
#sex
plt.bar(height = insurance.sex, x = np.arange(1,1001,1))
plt.hist(insurance.sex)
plt.boxplot(insurance.sex)
plt.show()

#age
plt.bar(height = insurance.Age, x = np.arange(1,1001,1))
plt.hist(insurance.Age)
plt.boxplot(insurance.Age)
plt.show()

# children
plt.bar(height = insurance.children, x = np.arange(1,1001,1))
plt.hist(insurance.children)
plt.boxplot(insurance.children)
plt.show()

# smoker
plt.bar(height = insurance.Smoker, x = np.arange(1,1001,1))
plt.hist(insurance.Smoker)
plt.boxplot(insurance.Smoker)
plt.show()

# bmi
plt.bar(height = insurance.bmi, x = np.arange(1,1001,1))
plt.hist(insurance.bmi)
plt.boxplot(insurance.bmi)
plt.show()

# education
plt.bar(height = insurance.education, x = np.arange(1,1001,1))
plt.hist(insurance.education)
plt.boxplot(insurance.education)
plt.show()

#income
plt.bar(height = insurance.income, x = np.arange(1,1001,1))
plt.hist(insurance.income)
plt.boxplot(insurance.income)
plt.show()

#Distinct_Parties_on_Claim
plt.bar(height = insurance.Distinct_Parties_on_Claim, x = np.arange(1,1001,1))
plt.hist(insurance.Distinct_Parties_on_Claim)
plt.boxplot(insurance.Distinct_Parties_on_Claim)
plt.show()

#Type_of_insurance
plt.bar(height = insurance.Type_of_insurance, x = np.arange(1,1001,1))
plt.hist(insurance.Type_of_insurance)
plt.boxplot(insurance.Type_of_insurance)
plt.show()

#Policy_Length
plt.bar(height = insurance.Policy_Length, x = np.arange(1,1001,1))
plt.hist(insurance.Policy_Length)
plt.boxplot(insurance.Policy_Length)
plt.show()

#Claim_Type
plt.bar(height = insurance.Claim_Type, x = np.arange(1,1001,1))
plt.hist(insurance.Claim_Type)
plt.boxplot(insurance.Claim_Type)
plt.show()

#Payment_method_
plt.bar(height = insurance.Payment_method_, x = np.arange(1,1001,1))
plt.hist(insurance.Payment_method_)
plt.boxplot(insurance.Payment_method_)
plt.show()

#Number_of_declarations
plt.bar(height = insurance.Number_of_declarations, x = np.arange(1,1001,1))
plt.hist(insurance.Number_of_declarations)
plt.boxplot(insurance.Number_of_declarations)
plt.show()

#Number_of_authorizations_
plt.bar(height = insurance.Number_of_authorizations_, x = np.arange(1,1001,1))
plt.hist(insurance.Number_of_authorizations_)
plt.boxplot(insurance.Number_of_authorizations_)
plt.show()

#Handling_time_of_authorizations_and_declarations_
plt.bar(height = insurance.Handling_time_of_authorizations_and_declarations_, x = np.arange(1,1001,1))
plt.hist(insurance.Handling_time_of_authorizations_and_declarations_)
plt.boxplot(insurance.Handling_time_of_authorizations_and_declarations_)
plt.show()

#Duration_of_current_insurance_contract
plt.bar(height = insurance.Duration_of_current_insurance_contract, x = np.arange(1,1001,1))
plt.hist(insurance.Duration_of_current_insurance_contract)
plt.boxplot(insurance.Duration_of_current_insurance_contract)
plt.show()

#Elapsed_time_since_last_contact_moment
plt.bar(height = insurance.Elapsed_time_since_last_contact_moment, x = np.arange(1,1001,1))
plt.hist(insurance.Elapsed_time_since_last_contact_moment)
plt.boxplot(insurance.Elapsed_time_since_last_contact_moment)
plt.show()

#Elapsed_time_since_the_last_complaint
plt.bar(height = insurance.Elapsed_time_since_the_last_complaint, x = np.arange(1,1001,1))
plt.hist(insurance.Elapsed_time_since_the_last_complaint)
plt.boxplot(insurance.Elapsed_time_since_the_last_complaint)
plt.show()

#Duration_of_current_insurance_contract
plt.bar(height = insurance.Duration_of_current_insurance_contract, x = np.arange(1,1001,1))
plt.hist(insurance.Duration_of_current_insurance_contract)
plt.boxplot(insurance.Duration_of_current_insurance_contract)
plt.show()

#Elapsed_time_since_last_contact_moment
plt.bar(height = insurance.Elapsed_time_since_last_contact_moment, x = np.arange(1,1001,1))
plt.hist(insurance.Elapsed_time_since_last_contact_moment)
plt.boxplot(insurance.Elapsed_time_since_last_contact_moment)
plt.show()

#Elapsed_time_since_last_contact_moment
plt.bar(height = insurance.Elapsed_time_since_last_contact_moment, x = np.arange(1,1001,1))
plt.hist(insurance.Elapsed_time_since_last_contact_moment)
plt.boxplot(insurance.Elapsed_time_since_last_contact_moment)
plt.show()

#Product_usage_
plt.bar(height = insurance.Product_usage_, x = np.arange(1,1001,1))
plt.hist(insurance.Product_usage_)
plt.boxplot(insurance.Product_usage_)
plt.show()

#Elapsed_time_since_the_last_complaint
plt.bar(height = insurance.Elapsed_time_since_the_last_complaint, x = np.arange(1,1001,1))
plt.hist(insurance.Elapsed_time_since_the_last_complaint)
plt.boxplot(insurance.Elapsed_time_since_the_last_complaint)
plt.show()

#Policy_Claim_Day_Diff
plt.bar(height = insurance.Policy_Claim_Day_Diff, x = np.arange(1,1001,1))
plt.hist(insurance.Policy_Claim_Day_Diff)
plt.boxplot(insurance.Policy_Claim_Day_Diff)
plt.show()

#Claim_History
plt.bar(height = insurance.Claim_History, x = np.arange(1,1001,1))
plt.hist(insurance.Claim_History)
plt.boxplot(insurance.Claim_History)
plt.show()

#Renewal_History
plt.bar(height = insurance.Renewal_History, x = np.arange(1,1001,1))
plt.hist(insurance.Renewal_History)
plt.boxplot(insurance.Renewal_History)
plt.show()

#Customer_Complaint
plt.bar(height = insurance.Customer_Complaint, x = np.arange(1,1001,1))
plt.hist(insurance.Customer_Complaint)
plt.boxplot(insurance.Customer_Complaint)
plt.show()

#Number_of_complaints
plt.bar(height = insurance.Number_of_complaints, x = np.arange(1,1001,1))
plt.hist(insurance.Number_of_complaints)
plt.boxplot(insurance.Number_of_complaints)
plt.show()

#Customer_mentioned_that_they_are_going_to_switch
plt.bar(height = insurance.Customer_mentioned_that_they_are_going_to_switch, x = np.arange(1,1001,1))
plt.hist(insurance.Customer_mentioned_that_they_are_going_to_switch)
plt.boxplot(insurance.Customer_mentioned_that_they_are_going_to_switch)
plt.show()

#Switching_barrier_
plt.bar(height = insurance.Switching_barrier_, x = np.arange(1,1001,1))
plt.hist(insurance.Switching_barrier_)
plt.boxplot(insurance.Switching_barrier_)
plt.show()

#Claim_Cancellation
plt.bar(height = insurance.Claim_Cancellation, x = np.arange(1,1001,1))
plt.hist(insurance.Claim_Cancellation)
plt.boxplot(insurance.Claim_Cancellation)
plt.show()

#Brand_credibility_
plt.bar(height = insurance.Brand_credibility_, x = np.arange(1,1001,1))
plt.hist(insurance.Brand_credibility_)
plt.boxplot(insurance.Brand_credibility_)
plt.show()

#Claim_After_Renewal
plt.bar(height = insurance.Claim_After_Renewal, x = np.arange(1,1001,1))
plt.hist(insurance.Claim_After_Renewal)
plt.boxplot(insurance.Claim_After_Renewal)
plt.show()

#Contracted_care_
plt.bar(height = insurance.Contracted_care_, x = np.arange(1,1001,1))
plt.hist(insurance.Contracted_care_)
plt.boxplot(insurance.Contracted_care_)
plt.show()

#Experience_during_contact_moment
plt.bar(height = insurance.Experience_during_contact_moment, x = np.arange(1,1001,1))
plt.hist(insurance.Experience_during_contact_moment)
plt.boxplot(insurance.Experience_during_contact_moment)
plt.show()

#Premium_price_
plt.bar(height = insurance.Premium_price_, x = np.arange(1,1001,1))
plt.hist(insurance.Premium_price_)
plt.boxplot(insurance.Premium_price_)
plt.show()

#Outstanding_charges
plt.bar(height = insurance.Outstanding_charges, x = np.arange(1,1001,1))
plt.hist(insurance.Outstanding_charges)
plt.boxplot(insurance.Outstanding_charges)
plt.show()

#Discount
plt.bar(height = insurance.Discount, x = np.arange(1,1001,1))
plt.hist(insurance.Discount)
plt.boxplot(insurance.Discount)
plt.show()

#Deductible_excess_
plt.bar(height = insurance.Deductible_excess_, x = np.arange(1,1001,1))
plt.hist(insurance.Deductible_excess_)
plt.boxplot(insurance.Deductible_excess_)
plt.show()

#Renewal
plt.bar(height = insurance.Renewal, x = np.arange(1,1001,1))
plt.hist(insurance.Renewal)
plt.boxplot(insurance.Renewal)
plt.show()


#####################Bivariate plots#########################################################
import matplotlib.pyplot as plt # visualization
import seaborn as sns  # visualization
import seaborn as sb # visualization

plt.figure(figsize= (6, 6))
sns.heatmap(insurance.corr())

plt.figure(figsize=(12,10))
cor = insurance.corr()
sns.heatmap(cor,annot=True, cmap= plt.cm.CMRmap_r)
plt.show()

#######scatter & line plots high correlation value##################################

insurance.plot.scatter('Product_usage_', 'income')
insurance.plot.line('Product_usage_', 'income')

insurance.plot.scatter('Brand_credibility_', 'Handling_time_of_authorizations_and_declarations_')
insurance.plot.line('Brand_credibility_', 'Handling_time_of_authorizations_and_declarations_')

insurance.plot.scatter('Deductible_excess_', 'Claim_Type')
insurance.plot.line('Deductible_excess_', 'Claim_Type')

insurance.plot.scatter('Premium_price_', 'Number_of_authorizations_')
insurance.plot.line('Premium_price_', 'Number_of_authorizations_')

insurance.plot.scatter('children', 'Product_usage_')
insurance.plot.line('children', 'Product_usage_')

insurance.plot.scatter('Renewal', 'Policy_Claim_Day_Diff')
insurance.plot.line('Renewal', 'Policy_Claim_Day_Diff')

insurance.plot.scatter('Product_usage_', 'Renewal_History')
insurance.plot.line('Product_usage_', 'Renewal_History')

insurance.plot.scatter('Outstanding_charges', 'Number_of_complaints')
insurance.plot.line('Outstanding_charges', 'Number_of_complaints')

insurance.plot.scatter('Outstanding_charges', 'Discount')
insurance.plot.line('Outstanding_charges', 'Discount')

insurance.plot.scatter('Outstanding_charges', 'Deductible_excess_')
insurance.plot.line('Outstanding_charges', 'Deductible_excess_')

insurance.plot.scatter('Contracted_care_', 'Outstanding_charges')
insurance.plot.line('Contracted_care_', 'Outstanding_charges')

insurance.plot.scatter('Contracted_care_', 'Discount')
insurance.plot.line('Contracted_care_', 'Discount')

insurance.plot.scatter('Contracted_care_', 'Deductible_excess_')
insurance.plot.line('Contracted_care_', 'Deductible_excess_')

###################other type's bivariate plots########################################
sns.pairplot(insurance, height=5.5)
plt.triplot(insurance.Contracted_care_, insurance.Renewal)
plt.bar(insurance.Contracted_care_ ,insurance.Renewal)
plt.barbs(insurance.Contracted_care_ ,insurance.Renewal)
insurance['Duration_of_current_insurance_contract'].corr(insurance['Renewal'])
insurance.plot.line('Duration_of_current_insurance_contract', 'Renewal')
insurance.plot.hexbin('Duration_of_current_insurance_contract', 'Renewal')