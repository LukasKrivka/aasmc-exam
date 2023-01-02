# aasmc-exam

Dataset: https://www.kaggle.com/datasets/uciml/adult-census-income

Question / Hypothesis:
Does this data indicate discrimination or racism?

Supporting hypothesis to be tested:
- is the attribute "race" significant as a predictor when explaining "income"
- is the attribute "gender" significant as a predictor when explaining "income"

Methods:
- Logistic Regression - to get the estimators of the predictors' coefficient
- t-Test - to test the significance of the coefficient (H0: coefficient = 0)

_________________________________________________________________________________
---------------------------------------------------------------------------------
Change the frame of the project / paper to dimensionality reduction
- we are interested in explaining what attributes influence whether a person makes 50k
- with categorical variables encoded as dummy variables, the number of features grows very fast
  - logistic regression including around half of attributes has issues with multicollinearity (singular matrix and no convergence)
- we can test their association (categorical x categorical) with chi-squared test of independence and remove highly correlated
- we can fit a regression and then from the estimators of the coefficients mean and std test their significance (do they differ from 0)
- we can fit the LASSO regression instead using regularization to get rid of less important variables
- we can also use PCA (or MCA) to find which attributes contribute to the most significant components

_________________________________________________________________________________
---------------------------------------------------------------------------------
Second dataset(s): Folktables

Filtered by: 
AGEP > 16,  PINCP > 100, WKHP > 0, PWGTP (personal weight) >= 1

feature_names: 
- 'AGEP' # Age (Continuous)
- 'SEX' # Gender (Binary, 1: male, 2: female)
- 'WKHP' # Hours-per-week (Continuous)
- 'COW' # Workclass (Categorical, nominal)
- 'SCHL' # Educational attainment (Categorical, ordinal)
- 'MAR' # Marital status (Categorical, nominal)
- 'RAC1P' # Recoded detailed race code (Categorical, nominal)
- 'WAOB' # World area of birth (Categorical, nominal) (native-country)
- 'OCCP' # Occupation (Categorical, nominal)
- 'INTP' # Interests, dividends and net rental income (Continuous) (capital gain - capital loss)
- 'RELSHIPP' # Relationship (Categorical, nominal)
- 'CIT' # Citizenship status (Categorical, nominal) (extra)

target_name:
- 'PINCP' # Total person's income (Continuous)

### conversions

sex = {1: "Male", 2: "Female"}

race = {1: "White", 2:"African-American", 3:"American-Indian", 4:"Alaska Native", 5:"Am. Indian and Alaska Nat.", 
        6: "Asian", 7:"Native Hawaiian or Other Pacific Islander", 8: "Other Races (alone)", 
        9: "Two or More Races"}
        
cow = {1: "Private (for profit)", 2:"Private (non-profit)", 3: "Local gov.", 4: "State gov.", 5:"Federal gov.",  
       6: "Self-employed (not inc.)", 7: "Self-employed (inc.)", 8:"Without Pay", 9: "Unemployed" }
       
cit = {1: "Born in US", 2: "Born in unincorporated territory", 3:"Born abroad (US parents)", 
       4: "Citizen by Naturalisation", 5:"Not a citizen"}
       
mar = {1: "Married", 2:"Widowed", 3:"Divorced", 4:"Separated", 5:"Never Maried"}

schl = {1: "No schooling completed", 2: "Nursery school, preschool", 3: "Kindergarten", 4: "Grade 1", 5: "Grade 2",
        6: "Grade 3", 7: "Grade 4", 8: "Grade 5", 9: "Grade 6", 10: "Grade 7", 11: "Grade 8", 12: "Grade 9",
        13: "Grade 10", 14: "Grade 11", 15: "12th grade - no diploma", 16: "Regular high school diploma", 
        17: "GED or alternative credential", 18: "Some college, but less than 1 year", 
        19: "1 or more years of college credit, no degree", 20: "Associate's degree", 21: "Bachelor's degree",
        22: "Master's degree", 23: "Professional degree beyond a bachelor's degree", 24: "Doctorate degree"}
        
occp = {MGR: "Management Occupations", BUS: "Business Occupations", FIN: "Financial Operations Occupations",
        CMM: "Computer and Mathematical Occupations", ENG: "Architecture and Engineering Occupations",
        SCI: "Life, Physical, and Social Science Occupations", CMS: "Community and Social Service Occupations",
        LGL: "Legal Occupations", EDU: "Educational Instruction and Library Occupations",
        ENT: "Arts, Design, Entertainment, Sports, and Media Occupations", 
        MED: "Healthcare Practitioners and Technical Occupations",
        HLS: "Healthcare Support Occupations", PRT: "Protective Service Occupations",
        EAT: "Food Preparation and Serving Related Occupations", 
        CLN: "Building and Grounds Cleaning and Maintenance Occupations",
        PRS: "Personal Care and Service Occupations", SAL: "Sales and Related Occupations",
        OFF: "Office and Administrative Support Occupations", FFF: "Farming, Fishing, and Forestry Occupations",
        CON: "Construction Occupations", EXT: "Extraction Occupations", 
        RPR: "Installation, Maintenance, and Repair Occupations", PRD: "Production Occupations",
        TRN: "Transportation and Material Moving Occupations", MIL: "Military Specific Occupations",
        UEP: "Unemployed, With No Work Experience In The Last 5 Years Or Earlier Or Never Worked"}
        
waob =  {1: "US state", 2: "PR and US Island Areas", 3: "Latin America", 4: "Asia", 5: "Europe", 6: "Africa",
         7: "Northern America", 8: "Oceania and at Sea"}
         
relshipp = {20: "Reference person", 21: "Opposite-sex husband/wife/spouse", 22: "Opposite-sex unmarried partner",
            23: "Same-sex husband/wife/spouse", 24: "Same-sex unmarried partner", 25: "Biological son or daughter",
            26: "Adopted son or daughter", 27: "Stepson or stepdaughter", 28: "Brother or sister",
            29: "Father or mother", 30: "Grandchild", 31: "Parent-in-law", 32: "Son-in-law or daughter-in-law",
            33: "Other relative", 34: "Roommate or housemate", 35: "Foster child", 36: "Other nonrelative",
            37: "Institutionalized group quarters population", 38: "Noninstitutionalized group quarters population"}
