#Data Preparation - Version 3


import pandas as pd
import numpy as np
import random
df = pd.read_excel(r"C:\Users\sprav\Desktop\Data Science\Live Project\360\Feature set.xlsx")
df.columns = df.columns.str.replace(' ','_')

df.drop(['Date_of_policy','Outstanding_charges','Claim_before_Premium_Paid','Customer_Locality','Customer_Region','Claim_Description','charges'], axis = 1, inplace=True)


'''smoker,Discount,Deductible excess,Payment method ,Switching barrier ,Contracted care ,
Customer mentioned that they are going to switch,Claim Cancellation'''


df["smoker"]=np.random.binomial(n=1,size=1000, p=0.25)    #25% of indian adults smoke  
df['smoker'] = df['smoker'].replace([0,1],["No","Yes"])
df['smoker'].value_counts()

df["education"]=np.random.binomial(n=1,size=1000, p=0.85)    #85% of Literate indians involve health insurance
df['education'] = df['education'].replace([0,1],["Illiterate","Literate"])
df['education'].value_counts()

df["Payment_method_"]=np.random.binomial(n=1,size=1000, p=0.5)    #50% of indians pay through online
df['Payment_method_'] = df['Payment_method_'].replace([0,1],["Offline","Online"])
df['Payment_method_'].value_counts()

df["Switching_barrier_"]=np.random.binomial(n=1,size=1000, p=0.3)    #30% of indian adults smoke  
df['Switching_barrier_'] = df['Switching_barrier_'].replace([0,1],["No","Yes"])
df['Switching_barrier_'].value_counts()

df["sex"]=np.random.binomial(n=1,size=1000, p=0.52)     #52% of indian adults are male 
df['sex'] = df['sex'].replace([0,1],["Female","Male"])
df['sex'].value_counts()

df["Discount"]=np.random.binomial(n=1,size=1000, p=0.1)    #10% of customer avail discount 
df['Discount'] = df['Discount'].replace([0,1],["No","Yes"])
df['Discount'].value_counts()

df["Deductible_excess_"]=np.random.binomial(n=1,size=1000, p=0.05) #Rare scenario so just 5% payable case   
df['Deductible_excess_'] = df['Deductible_excess_'].replace([0,1],["Non-Payable","Payable"])
df['Deductible_excess_'].value_counts()

df["Customer_Complaint"]=np.random.binomial(n=1,size=1000, p=0.35) # Around 35% people complaint    
df['Customer_Complaint'] = df['Customer_Complaint'].replace([0,1],["No","Yes"])
df['Customer_Complaint'].value_counts()

df["Customer_mentioned_that_they_are_going_to_switch"]=np.random.binomial(n=1,size=1000, p=0.4) #40% of people have said 
df['Customer_mentioned_that_they_are_going_to_switch'] = df['Customer_mentioned_that_they_are_going_to_switch'].replace([0,1],["No","Yes"])
df['Customer_mentioned_that_they_are_going_to_switch'].value_counts()

#######3 should depend on claim history 
'''
df["Claim_before_Premium_Paid"]=np.random.binomial(n=1,size=1000, p=0.5)     #15% times claim is cancelled
df['Claim_before_Premium_Paid'] = df['Claim_before_Premium_Paid'].replace([0,1],["No","Yes"])
df['Claim_before_Premium_Paid'].value_counts()'''

df["Distinct_Parties_on_Claim"]=np.random.binomial(n=1,size=1000, p=0.5)
df['Distinct_Parties_on_Claim'] = df['Distinct_Parties_on_Claim'].replace([0,1],["TPA","In-House"])
df['Distinct_Parties_on_Claim'].value_counts()

df["Claim_Type"]=np.random.binomial(n=1,size=1000, p=0.8)    #80% of type of claim is cashless  
df['Claim_Type'] = df['Claim_Type'].replace([0,1],["Cashless","Reimbursement"])
df['Claim_Type'].value_counts()

df["Renewal"]=np.random.binomial(n=1,size=1000, p=0.5) 
df['Renewal'] = df['Renewal'].replace([0,1],["No","Yes"])
df['Renewal'].value_counts()


#Age

grp2 = []
for i in range(0,360):
    i = random.randint(18,40) #Group-2: Age b/w 19-40 years for 24% data
    grp2.append(i)
grp3 = []
for i in range(0,360):
    i = random.randint(41,60) #Group-3: Age b/w 41-60 years for 36% data
    grp3.append(i)
grp4 = []
for i in range(0,280):
    i = random.randint(61,65) #Group-4: Age b/w 61-65 years for 28% data
    grp4.append(i)

age=grp2+grp3+grp4
random.shuffle(age)
df["age"] = age

bmi1=[]
for i in range(0,160):
    i=round(random.uniform(18.5,24.9), 1) # this function rounds the random number between 18.5 to 24.9 upto 1 decimal place   
    # BMI of humans lies b/w 18.5-24.9 only 16% ppl have ideal bmi
    bmi1.append(i)
bmi2=[]
for i in range(0,160):
    i=round(random.uniform(10,18.4), 1) # this is for underweight 16% ppl 
    bmi2.append(i)
bmi3=[]
for i in range(0,350):
    i=round(random.uniform(25, 29.9), 1) # this is for 35% who fall in obese category 
    bmi3.append(i)
bmi4=[]
for i in range(0,330):
    i=round(random.uniform(30, 100), 1) # this is for 33% who fall in overweight category 
    bmi4.append(i)
bmi= bmi1 + bmi2 + bmi3 + bmi4
random.shuffle(bmi)
df["bmi"] = bmi

claim_history=[]
for i in range(0,1000):
    n = random.randint(0,5)     #Condiering 0-5 Claim_History
    claim_history.append(n)
df["Claim_History"]=claim_history

renewal_history=[]
for i in range(0,1000):
    n = random.randint(0,10)     #Condiering 0-10 renewal_History
    renewal_history.append(n)
df["Renewal_History"]=renewal_history
### Age & children and Age & Income dependent variables

for i in range(len(df.age)):
    if (df.age[i]<27):
        df.children[i]=0
    elif (26<df.age[i]<39):
         n = random.randint(1,2)    
         df.children[i]=int(n)
    else:
        df.children[i]=3

for i in range(len(df.age)):
    if (df.age[i]<27):
        n=random.randrange(250000,1500000,25000)
        df.income[i]=n
    elif (27<df.age[i]<39):
        n=random.randrange(800000,2000000,25000)
        df.income[i]=n   
    else:
         n=random.randrange(1500000,2500000,25000)
         df.income[i]=n 

Policy_Length1=[]
for i in range(0,600):
    i =1 #Need to confirm the range of policy length
    Policy_Length1.append(i)
Policy_Length2=[]
for i in range(0,300):
    i =2 #Need to confirm the range of policy length
    Policy_Length2.append(i)
Policy_Length3=[]
for i in range(0,100):
    i =3 #Need to confirm the range of policy length
    Policy_Length3.append(i)
Policy_Length=Policy_Length1+Policy_Length2+Policy_Length3
df["Policy_Length"] = Policy_Length

Elapsed_time_since_last_contact_moment= []
for i in range(0,1000):
    n = random.randint(7,30)     #Considering elapsed time since last contact min.7 days and max. of 15 days
    Elapsed_time_since_last_contact_moment.append(n)
df["Elapsed_time_since_last_contact_moment"]=Elapsed_time_since_last_contact_moment

Experience_during_contact_moment= []
for i in range(0,1000):
    n = random.randint(1,5)     #Star rating 1-5
    Experience_during_contact_moment.append(n)
df["Experience_during_contact_moment"]=Experience_during_contact_moment

Number_of_declarations= []
for i in range(0,1000):
    n = random.randint(0,5)     #Considering min.6 declarations and max. of 10 declarations 
    Number_of_declarations.append(n)
df["Number_of_declarations"]=Number_of_declarations

Number_of_authorizations_= []
for i in range(0,1000):
    n = random.randint(3,10)     #Considering min.6 authorizations and max. of 10 authorizations
    Number_of_authorizations_.append(n)
df["Number_of_authorizations_"]=Number_of_authorizations_

Duration_of_current_insurance_contract= [] 
for i in range(0,1000):
    n = random.randint(365,548)     #range for the duration is 1yr to 1.5yr
    Duration_of_current_insurance_contract.append(n)
df["Duration_of_current_insurance_contract"]=Duration_of_current_insurance_contract

Type_of_insurance= []
for i in range(0,1000):
    n = random.randint(1,4)     #Need to specify the 4 different types of insurance here
    Type_of_insurance.append(n)
df["Type_of_insurance"]=Type_of_insurance
df["Type_of_insurance"]=df.Type_of_insurance.replace(to_replace=[1,2,3,4],value=["Individual","Floater","Senior Citizen","Critical Illness"])

Product_usage_= []
for i in range(0,1000):
    n = random.randint(0,100)     #Product usage is 0-100%
    Product_usage_.append(n)
df["Product_usage_"]=Product_usage_

for i in range(0,1000):
    if (df.Claim_Type[i]=="Reimbursement"):
        n = random.randint(30,90) #Considering min.0 days of Policy Claim Day Difference and till max. of 15 days
        df.Policy_Claim_Day_Diff[i]=n
    else:
        df.Policy_Claim_Day_Diff[i]=0

#df["smoker","education","Payment_method_"]=df[""].apply(pd.to_numeric)
##Customer complaint , Number of complaints, Cancellation , Switch barrier , Brand credibility

##Handling time of authorizations & declaration depend on Number of declarations and authorisation 

for i in range(0,1000):
    if ((df.Number_of_authorizations_[i])<6):
        c=round(random.uniform(0.5,2), 1)
        df.Handling_time_of_authorizations_and_declarations_[i]=c
    else:
        y=round(random.uniform(2,4), 1)
        df.Handling_time_of_authorizations_and_declarations_[i]=y 

#If only the customer has complaint, there will be elasped time of last complaint
for i in range(0,1000):
    if (df.Customer_Complaint[i]=="Yes"):
        a=random.randint(0,15) # Elaspsed time is 0-15 range
        df.Elapsed_time_since_the_last_complaint[i]=a
    else:
        df.Elapsed_time_since_the_last_complaint[i]=0


# If only the customer has complaint , we can have some number in number of complaints column
for i in range(0,1000):
    if (df.Customer_Complaint[i]=="Yes"):
        a=random.randint(1,5) #Maximum number of complaints is 5
        df.Number_of_complaints[i]=a
    else:
        df.Number_of_complaints[i]=0

for i in range(0,1000):
    if(df.Claim_History[i]==0):
         df.Claim_Cancellation[i]="No"
    else:
        df["Claim_Cancellation"]=np.random.binomial(n=1,p=0.15)     #15% times claim is cancelled
        df['Claim_Cancellation'] = df['Claim_Cancellation'].replace([0,1],["No","Yes"])
        df['Claim_Cancellation'].value_counts()

# If someone cancels the claim, they tend to give lesser brand rating 
for i in range(0,1000):
    if (df.Claim_Cancellation[i]=="Yes"):
        n=random.randint(1,2)
        df.Brand_credibility_[i]=n
    else:
        s=random.randint(3,5)
        df.Brand_credibility_[i]=s
 ##############################################################################
############################################################################       
''' Premium Price, calculated depending on type of policy'''

for i in range(0,1000):
    if (df.Type_of_insurance[i]== "Individual"):
        n=random.randrange(3000,10000,2000) 
        df.Premium_price_[i]=n
    elif (df.Type_of_insurance[i]== "Floater"):
        n=random.randrange(11000,20000,2000) 
        df.Premium_price_[i]=n
    elif (df.Type_of_insurance[i]== "Senior Citizen"):
        n=random.randrange(21000,35000,2000) 
        df.Premium_price_[i]=n
    else:
        n=random.randrange(36000,50000,2000) 
        df.Premium_price_[i]=n
        
''' Contracted care can depend on elapsed time since last complaint'''
for i in range(0,1000):
    if(str(df.Customer_Complaint[i])=="No"):
        df['Contracted_care_'][i]="Not Applicable"
    else:
        df["Contracted_care_"][i]=np.random.binomial(n=1,p=0.8)    #80% of type of claim is cashless  
        df['Contracted_care_'] = df['Contracted_care_'].replace([0,1],["Not Resolved","Resolved"])
        df['Contracted_care_'].value_counts()
########### Claim After Renewal can depend on renewal history
for i in range(0,1000):
    if(df.Renewal_History[i]==0):
       df["Claim_After_Renewal"][i]=0
    else:
        n=random.randint(1,5)
        df["Claim_After_Renewal"][i]=n
        
######## Claim Cancellation should depend on claim history
df[["sex","smoker","education",'Distinct_Parties_on_Claim',
    'Type_of_insurance','Claim_Type', 'Payment_method_', 
    'Customer_Complaint','Customer_mentioned_that_they_are_going_to_switch',
    'Switching_barrier_', 'Claim_Cancellation', 'Contracted_care_',
    'Discount', 'Renewal']]=df[["sex","smoker","education",'Distinct_Parties_on_Claim',
                                'Type_of_insurance','Claim_Type', 'Payment_method_', 'Customer_Complaint',
                                'Customer_mentioned_that_they_are_going_to_switch','Switching_barrier_',
                                'Claim_Cancellation', 'Contracted_care_','Discount',
                                'Renewal']].astype(str)
                                
df[["children","income","Handling_time_of_authorizations_and_declarations_",
    'Elapsed_time_since_the_last_complaint', 
    'Policy_Claim_Day_Diff','Number_of_complaints','Brand_credibility_',
    'Claim_After_Renewal','Premium_price_']]=df[["children","income","Handling_time_of_authorizations_and_declarations_",
                                                                      'Elapsed_time_since_the_last_complaint', 'Policy_Claim_Day_Diff',
                                                                      'Number_of_complaints','Brand_credibility_','Claim_After_Renewal',
                                                                      'Premium_price_']].astype(int)

df.to_csv(r'C:\Users\sprav\Desktop\Data Science\Live Project\360\Data3.csv', index=False)
