# -*- coding: utf-8 -*-

#This script: 
#	Reads Yelp's Academic Dataset (JSON)
#	Ouputs the following CSV files to be ingested by MatLab:
#		A list of all the schools
#		For each school, a list of all nearby businesses 
#		For each school, a matrix of all star rankings associated with users' reviews of nearby businesses

import os
import json
import csv
import pickle
filePath = './data/yelp_academic_dataset.json'
#json_data=open('dataset_sample.json')
#json_data=open('yelp_academic_dataset.json')
json_data=open(filePath)
reviews={}
schools ={}
names ={}
user_list = []
business_list =[]
school_list = []

print 'Reading Yelp file'
for line in json_data:
    data = json.loads(line)
    
    if(data['type'] == 'review'):
        user_id = data['user_id']
        business_id = data['business_id']
        stars = data['stars']
        
        if business_id not in business_list:
            business_list.append(business_id)
            reviews[business_id]={}
        if user_id not in user_list:  
            user_list.append(user_id)
        
        reviews[business_id][user_id] = stars
            
    if (data['type'] == 'business'):
        school_ids = data['schools']
        business_id = data['business_id']
        
        for school_id in school_ids:
            if school_id not in school_list:
                school_list.append(school_id)
        
        schools[business_id] = school_ids
        names[business_id] = data['name']
        
json_data.close()

print 'Done reading Yelp file'

#Export the list of schools
school_list_file = open('school_list.csv', 'w')
school_list_writer = csv.writer(school_list_file, dialect='excel')
for school_id in school_list:
    school_list_writer.writerow([school_id])
    print school_id
school_list_file.close()

#For each school
for k in range(len(school_list)):
    school_id = school_list[k]
    uni_business_list = [business_id for business_id in business_list if school_id in schools[business_id]]

    print 'Exporting list of businesses of {0}'.format(school_id)
    #Export the list of businesses
    business_list_file = open('business_list{0}.csv'.format(str(k+1)), 'w')
    business_list_writer = csv.writer(business_list_file, dialect='excel')
    for business_id in uni_business_list:
        business_list_writer.writerow([business_id])
    business_list_file.close()

    print 'Exporting reviews of {0}'.format(school_id)
    #Export reviews as a sparse matrix
    reviews_file = open('reviews{0}.csv'.format(str(k+1)), 'w')
    reviews_writer = csv.writer(reviews_file, dialect='excel')
    for i in range(len(uni_business_list)):
        business_id = uni_business_list[i]
        for j in range(len(user_list)):
            user_id= user_list[j]
            if user_id in reviews[business_id]:
                reviews_writer.writerow([i+1,j+1,reviews[business_id][user_id]]) #Don't forget to filter out unnecessary users in Matlab
    reviews_file.close()
    print 'Done with {0}'.format(school_id)
print 'Done'
pickle.dump(names, open( "save_business_names.p", "wb" ))

    


