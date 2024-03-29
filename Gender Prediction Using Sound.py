
# coding: utf-8

# ## 1. Sound it out!
# <p>Grey and Gray. Colour and Color. Words like these have been the cause of many heated arguments between Brits and Americans. Accents (and jokes) aside, there are many words that are pronounced the same way but have different spellings. While it is easy for us to realize their equivalence, basic programming commands will fail to equate such two strings. </p>
# <p>More extreme than word spellings are names because people have more flexibility in choosing to spell a name in a certain way. To some extent, tradition sometimes governs the way a name is spelled, which limits the number of variations of any given English name. But if we consider global names and their associated English spellings, you can only imagine how many ways they can be spelled out. </p>
# <p>One way to tackle this challenge is to write a program that checks if two strings sound the same, instead of checking for equivalence in spellings. We'll do that here using fuzzy name matching.</p>

# In[227]:


# Importing the fuzzy package
import fuzzy 

# Exploring the output of fuzzy.nysiis
fuzzy.nysiis('Michael')
fuzzy.nysiis('Michelle')

# Testing equivalence of similar sounding words
fuzzy.nysiis('aluminum')
fuzzy.nysiis('aluminium')


# In[228]:


get_ipython().run_cell_magic('nose', '', "import sys\n\ndef test_fuzzy_is_loaded():\n    assert 'fuzzy' in sys.modules, \\\n    'The fuzzy module should be loaded'")


# ## 2. Authoring the authors
# <p>The New York Times puts out a weekly list of best-selling books from different genres, and which has been published since the 1930’s.  We’ll focus on Children’s Picture Books, and analyze the gender distribution of authors to see if there have been changes over time. We'll begin by reading in the data on the best selling authors from 2008 to 2017.</p>

# In[229]:


# Importing the pandas module
import pandas as pd


# Reading in datasets/nytkids_yearly.csv, which is semicolon delimited.
author_df = pd.read_csv('datasets/nytkids_yearly.csv', delimiter = ';')

# Looping through author_df['Author'] to extract the authors first names
first_name = []
for name in author_df['Author']:
    first_name.append(name.split()[0])

# Adding first_name as a column to author_df
author_df['first_name'] = first_name

# Checking out the first few rows of author_df
author_df.head()


# In[230]:


get_ipython().run_cell_magic('nose', '', "    \ndef test_check_authors():\n    len_auth = len(author_df['first_name'])\n    all_names = list(author_df['first_name'])\n    assert ('Shel' in all_names and len_auth==603), \\\n    'first_name column does not contan the correct first names of authors'")


# ## 3. It's time to bring on the phonics... _again_!
# <p>When we were young children, we were taught to read using phonics; sounding out the letters that compose words. So let's relive history and do that again, but using python this time. We will now create a new column or list that contains the phonetic equivalent of every first name that we just extracted. </p>
# <p>To make sure we're on the right track, let's compare the number of unique values in the <code>first_name</code> column and the number of unique values in the nysiis coded column. As a rule of thumb, the number of unique nysiis first names should be less than or equal to the number of actual first names.</p>

# In[231]:


# Importing numpy
import numpy as np

# Looping through author's first names to create the nysiis (fuzzy) equivalent
nysiis_name = []
for name in first_name:
    nysiis_name.append(fuzzy.nysiis(name))

# Adding nysiis_name as a column to author_df
author_df['nysiis_name'] = nysiis_name

# Printing out the difference between unique firstnames and unique nysiis_names:
print(len(np.unique(first_name, nysiis_name)))



# In[232]:


get_ipython().run_cell_magic('nose', '', "\nimport numpy as np\n\ndef test_check_nysiis_list():\n    assert len( np.unique(author_df['nysiis_name']) ) == 145, \\\n        'The nysiis_name column does not contan the correct entries'")


# ## 4. The inbetweeners
# <p>We'll use <code>babynames_nysiis.csv</code>, a dataset that is derived from <a href="https://www.ssa.gov/oact/babynames/limits.html">the Social Security Administration’s baby name data</a>, to identify author genders. The dataset contains unique NYSIIS versions of baby names, and also includes the percentage of times the name appeared as a female name (<code>perc_female</code>) and the percentage of times it appeared as a male name (<code>perc_male</code>). </p>
# <p>We'll use this data to create a list of <code>gender</code>. Let's make the following simplifying assumption: For each name, if <code>perc_female</code> is greater than <code>perc_male</code> then assume the name is female, if <code>perc_female</code> is less than <code>perc_male</code> then assume it is a male name, and if the percentages are equal then it's a "neutral" name.</p>

# In[233]:


# Reading in datasets/babynames_nysiis.csv, which is semicolon delimited.
babies_df = pd.read_csv('datasets/babynames_nysiis.csv', delimiter = ';')

# Looping through babies_df to and filling up gender
gender = []
for x,y in zip(babies_df['perc_female'], babies_df['perc_male']):
    if x > y:
        gender.append('F')
    elif x < y:
        gender.append('M')
    else :
        gender.append('N')

# Add\Finvng a gender column to babies_df
babies_df['gender'] = gender

# Printing out the first few rows of babies_df
babies_df.head()


# In[234]:


get_ipython().run_cell_magic('nose', '', "\ndef test_gender_distribution():\n    assert len([i for i, x in enumerate(babies_df['gender']) if x == 'N']) == 1170,\\\n        'gender column does not contain the correct number of Male, Female and Neutral names, which are 7031, 8939 and 1170 respectively'")


# ## 5. Playing matchmaker
# <p>Now that we have identified the likely genders of different names, let's find author genders by searching for each author's name in the <code>babies_df</code> DataFrame, and extracting the associated gender. </p>

# In[235]:


# This function returns the location of an element in a_list.
# Where an item does not exist, it returns -1.
def locate_in_list(a_list, element):
    loc_of_name = a_list.index(element) if element in a_list else -1
    return(loc_of_name)

# Looping through author_df['nysiis_name'] and appending the gender of each
# author to author_gender.
author_gender = []
for author in author_df['nysiis_name'] : 
    idx = locate_in_list(list(babies_df['babynysiis']), author)
    if idx < 0:
        author_gender.append('unknown')
    else:
        author_gender.append(babies_df['gender'][locate_in_list(list(babies_df['babynysiis']),author)])
print(author_gender)

# Adding author_gender to the author_df
author_df['author_gender'] = author_gender


# Counting the author's genders

author_df['author_gender'].value_counts()


# In[236]:


get_ipython().run_cell_magic('nose', '', '\ndef len_authors():\n    return len(author_df[author_df.author_gender == "M"])\n\ndef test_num_males():\n    assert len_authors() == 191, \\\n        \'The number of Males (M) and Females (F) appear to be wrong. These are 191 and 395 respectively\'')


# ## 6. Tally up
# <p>From the results above see that there are more female authors on the New York Times best seller's list than male authors. Our dataset spans 2008 to 2017. Let's find out if there have been changes over time.</p>

# In[237]:


# Creating a list of unique years, sorted in ascending order.
years = sorted(author_df['Year'].unique())

# Initializing lists
males_by_yr = []
females_by_yr = []
unknown_by_yr = []

# Looping through years to find the number of male, female and unknown authors per year
for year in years:
    df = author_df[author_df['Year'] == year]
    females_by_yr.append(len( df[ df['author_gender'] == 'F']))
    males_by_yr.append(len(df[df['author_gender'] == 'M']))
    unknown_by_yr.append(len(df[df['author_gender'] == 'unknown']))

# Printing out yearly values to examine changes over time
print(males_by_yr, females_by_yr, unknown_by_yr)


# In[238]:


get_ipython().run_cell_magic('nose', '', '\ndef test_years():\n    correct_years = list(np.unique(author_df.Year))\n    assert list(years) == correct_years, \\\n    \'years should be the unique years in author_df["Year"] sorted in ascending order.\'\n\ndef test_gender_by_yr():\n    assert sum(males_by_yr)==191, \\\n    \'At least one of the lists (males_by_yr, females_by_yr, unknown_by_yr) contains an incorrect value.\'')


# ## 7. Foreign-born authors?
# <p>Our gender data comes from social security applications of individuals born in the US. Hence, one possible explanation for why there are "unknown" genders associated with some author names is because these authors were foreign-born. While making this assumption, we should note that these are only a subset of foreign-born authors as others will have names that have a match in <code>baby_df</code> (and in the social security dataset). </p>
# <p>Using a bar chart, let's explore the trend of foreign-born authors with no name matches in the social security dataset.</p>

# In[239]:


# Importing matplotlib
import matplotlib.pyplot as plt

# This makes plots appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Plotting the bar chart
plt.bar(years, unknown_by_yr)

# [OPTIONAL] - Setting a title, and axes labels


plt.xlabel('Year')
plt.ylabel('Number of Unknowns')
plt.show()


# In[240]:


get_ipython().run_cell_magic('nose', '', '\n# It\'s hard to test plots.\ndef test_nothing():\n    assert True, ""\n\n#def test_pos():\n#    assert  pos ==list(range(len(unknown_by_yr))) or pos== range(len(unknown_by_yr)) or pos==years, \\\n#    \'pos should be a list containing integer values with the same length as unknown_by_yr \'')


# ## 8. Raising the bar
# <p>What’s more exciting than a bar chart is a grouped bar chart. This type of chart is good for displaying <em>changes</em> over time while also <em>comparing</em> two or more groups. Let’s use a grouped bar chart to look at the distribution of male and female authors over time.</p>

# In[241]:


# Creating a new list, where 0.25 is added to each year
years_shifted = [x+.25 for x in years]

# Plotting males_by_yr by year
plt.bar(years, males_by_yr, width=.25, color='lightblue')

# Plotting females_by_yr by years_shifted
plt.bar(years_shifted, females_by_yr, width=.25, color='pink')

# [OPTIONAL] - Adding relevant Axes labels and Chart Title

plt.xlabel('year')
plt.ylabel('num of authors')
plt.show()


# In[242]:


get_ipython().run_cell_magic('nose', '', "\ndef test_years_shifted():\n    correct_years_shifted = [year + 0.25 for year in years]\n    assert list(years_shifted) == correct_years_shifted, \\\n    'years_shifted should be like years but with 0.25 added to each year.'")

